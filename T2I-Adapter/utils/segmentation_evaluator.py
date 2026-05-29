
import torch
import numpy as np
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor


class SegmentationControlEvaluator:
    def __init__(self, eval_dtype=torch.float16, ignore_index=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_dtype = eval_dtype
        self.processor = MaskFormerImageProcessor.from_pretrained("facebook/maskformer-swin-large-ade", local_files_only=True)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained("facebook/maskformer-swin-large-ade", local_files_only=True).to(self.device, eval_dtype).eval()
        self.num_classes = len(self.model.config.id2label)
        self.ignore_index = ignore_index
        self.metrics = ['mIoU', 'mDice', 'mFscore']

    def preprocess(self, images):
        return self.processor(images, return_tensors="pt")
    
    def postprocess(self, outputs, n_samples, input_size):
        predicted_semantic_map = np.stack([
            pred_sem_seg.cpu().numpy().astype(np.uint8) for pred_sem_seg in self.processor.post_process_semantic_segmentation(outputs, target_sizes=[input_size]*n_samples)
        ])
        return predicted_semantic_map

    @torch.no_grad()
    def predict_control(self, images):
        res = []
        for image in images:
            inputs = self.preprocess(image).to(self.device, self.eval_dtype)
            with torch.cuda.amp.autocast(dtype=self.eval_dtype):
                outputs = self.model(**inputs)
            r = self.postprocess(outputs, 1, image.size[::-1]) + 1 # This is because ids range from [1..N] in ADE20K put [0..N-1] in the MaskFormer
            res.append(r[0])
            del inputs, outputs
        return np.array(res)


    def intersect_and_union(self, pred_label: np.ndarray, label: np.ndarray,
                            num_classes: int, ignore_index: int):
        """Calculate Intersection and Union.

        Args:
            pred_label (np.ndarray): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (np.ndarray): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = np.histogram(intersect.astype(float), bins=(num_classes), range=[1, num_classes])[0]
        area_pred_label = np.histogram(pred_label.astype(float), bins=(num_classes), range=[1, num_classes])[0]
        area_label = np.histogram(label.astype(float), bins=(num_classes), range=[1, num_classes])[0]
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label
    
    def f_score(self, precision, recall, beta=1):
        """calculate the f-score value.

        Args:
            precision (float | np.ndarray): The precision value.
            recall (float | np.ndarray): The recall value.
            beta (int): Determines the weight of recall in the combined
                score. Default: 1.

        Returns:
            [np.ndarray]: The f-score value.
        """
        score = (1 + beta**2) * (precision * recall) / (
            (beta**2 * precision) + recall)
        return score

    def eval(self, gt, pred):
        """
        gt: (b, h, w) or (h, w)
        pred: (b, h, w) or (h, w)
        """
        assert isinstance(gt, np.ndarray)
        assert len(gt.shape) == len(pred.shape)
        if len(gt.shape) == 1:
            # batch size = 1
            gt = gt[None,...]
            pred = gt[None, ...]
        
        assert len(gt.shape) == 3

        ret_metrics = {k: [] for k in self.metrics}
        for gt_, pred_ in zip(gt, pred):
            (
                area_intersect,
                area_union,
                area_pred_label,
                area_label
            ) = self.intersect_and_union(
                pred_,
                gt_,
                self.num_classes,
                self.ignore_index
            )

            for k in self.metrics:
                if k == 'mIoU':
                    iou = area_intersect / area_union
                    ret_metrics[k].append(np.round(np.nanmean(iou) * 100, 2))
                if k == 'mDice':
                    dice =  2 * area_intersect / (area_pred_label + area_label)
                    ret_metrics[k].append(np.round(np.nanmean(dice) * 100, 2))
                if k == 'mFscore':
                    precision = area_intersect / area_pred_label
                    recall = area_intersect / area_label
                    f_score = np.array([
                        self.f_score(x[0], x[1]) for x in zip(precision, recall)
                    ])
                    ret_metrics[k].append(np.round(np.nanmean(f_score) * 100, 2))
            
        return ret_metrics
    
    def __call__(self, gen_samples, controls):
        """
        gen_samples: List[Image]
        controls: np.ndarray
        """
        preds = self.predict_control(gen_samples)
        ret_metrics = self.eval(controls, preds)
        return ret_metrics
