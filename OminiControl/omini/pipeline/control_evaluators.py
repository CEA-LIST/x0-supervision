import torch
import numpy as np
import einops
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from annotator.midas.api import MiDaSInference
from annotator.canny import CannyDetector
from ultralytics import YOLO
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.classification import BinaryF1Score
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any

class DepthControlEvaluator:
    def __init__(self, eval_dtype=torch.float16):
        self.eval_dtype= eval_dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.midas = MiDaSInference(model_type="dpt_hybrid").to(self.device, self.eval_dtype)
        self.metrics = ["RMSE"]

    def preprocess(self, images):
        x = []
        for image in images:
            im = torch.from_numpy(np.array(image)).to(self.device, self.eval_dtype)
            im = im / 127.5 - 1.
            im = einops.rearrange(im, 'h w c -> c h w')
            x.append(im)
        x = torch.stack(x)
        return x
    
    def postprocess(self, x):
        """
        x: (b, h, w)
        """
        assert len(x.shape) == 3

        x -= x.min(dim=-1, keepdim=True).values.min(dim=-2, keepdim=True).values
        x /= x.max(dim=-1, keepdim=True).values.max(dim=-2, keepdim=True).values
        x = (x.cpu().numpy() * 255.0).clip(0, 255)
        return x

    @torch.no_grad()
    def predict_control(self, images):
        x = self.preprocess(images)
        with torch.cuda.amp.autocast(dtype=self.eval_dtype):
            out = self.midas(x)
        depth = self.postprocess(out)
        
        return depth

    def eval(self, gt, pred):
        """
        gt: (b, h, w) or (h, w)
        pred: (b, h, w) or (h, w)
        """
        assert isinstance(gt, np.ndarray)
        assert len(gt.shape) == len(pred.shape), f"Error: prediction and target must have the same shape, got {pred.shape}, {gt.shape}"
        if len(gt.shape) == 2:
            # batch size = 1
            gt = gt[None,...]
            pred = gt[None, ...]
        
        assert len(gt.shape) == 3
        if gt.dtype == np.uint8:
            gt = gt.astype(float)
        ret_metrics = {k: [] for k in self.metrics}
        mse = np.sqrt(np.mean((gt - pred)**2, axis=(-2, -1)))
        ret_metrics["RMSE"] = mse.tolist()
        return ret_metrics
    
    def __call__(self, gen_samples, controls):
        """
        gen_samples: List[Image]
        controls: np.ndarray
        """
        preds = self.predict_control(gen_samples)
        ret_metrics = self.eval(controls, preds)
        return ret_metrics
    
    
class CannyControlEvaluator:
    def __init__(self, eval_dtype=torch.float16):
        self.eval_dtype= eval_dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.canny_detector = CannyDetector()
        model_path = "facebook/maskformer-swin-large-ade"
        self.processor = MaskFormerImageProcessor.from_pretrained(model_path)
        self.segmentator = MaskFormerForInstanceSegmentation.from_pretrained(model_path).to(self.device, eval_dtype).eval()
        self.ade20k_palette = np.load("ade20k_palette.npy")
        self.f1 = BinaryF1Score(multidim_average="samplewise")
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0, reduction="none")
        self.psnr = PeakSignalNoiseRatio(data_range=1.0, dim=(-3, -2, -1), reduction="none")
        self.metrics = ["F1", "SSIM", "PSNR"]

    def preprocess(self, images):
        return self.processor(images, return_tensors="pt")
    
    def postprocess(self, outputs, n_samples, input_size):
        canny = []
        for pred_sem_seg in self.processor.post_process_semantic_segmentation(outputs, target_sizes=[input_size]*n_samples):
            pred_sem_seg = pred_sem_seg.cpu().numpy().astype(np.uint8) + 1 # This is because ids range from [1..N] in ADE20K put [0..N-1] in the MaskFormer
            pred_sem_seg = self.ade20k_palette[pred_sem_seg].astype(np.uint8)
            pred_sem_seg_canny = self.canny_detector(pred_sem_seg, 100, 200)
            canny.append(pred_sem_seg_canny)
        canny = torch.from_numpy(np.stack(canny))
        return canny

    @torch.no_grad()
    def predict_control(self, images):
        inputs = self.preprocess(images).to(self.device, self.eval_dtype)
        with torch.cuda.amp.autocast(dtype=self.eval_dtype):
            outputs = self.segmentator(**inputs)
        res = self.postprocess(outputs, len(images), images[0].size[::-1])
        return res

    def eval(self, gt, pred):
        """
        gt: (b, h, w) or (h, w)
        pred: (b, h, w) or (h, w)
        """
        assert isinstance(gt, np.ndarray)
        assert len(gt.shape) == len(pred.shape)
        if len(gt.shape) == 2:
            # batch size = 1
            gt = gt[None,...]
            pred = gt[None, ...]
        
        assert len(gt.shape) == 3
        # if gt.dtype == np.uint8:
        #     gt = torch.from_numpy(gt)
        
        gt = einops.rearrange(torch.from_numpy(gt), "b h w -> b 1 h w")
        pred = einops.rearrange(pred, "b h w -> b 1 h w")
        
        ret_metrics = {k: [] for k in self.metrics}
        ret_metrics["SSIM"] = self.ssim((pred / 255.0).clip(0, 1), (gt / 255.0).clip(0, 1)).tolist() if gt.shape[0] > 1 else [self.ssim((pred / 255.0).clip(0, 1), (gt / 255.0).clip(0, 1)).tolist()]
        ret_metrics["PSNR"] = self.psnr((pred / 255.0).clip(0, 1), (gt / 255.0).clip(0, 1)).tolist() if gt.shape[0] > 1 else [self.psnr((pred / 255.0).clip(0, 1), (gt / 255.0).clip(0, 1)).tolist()]
        gt[gt == 255] = 1
        pred[pred == 255] = 1
        ret_metrics["F1"] = self.f1(pred.flatten(-3), gt.flatten(-3)).tolist() if gt.shape[0] > 1 else [self.f1(pred.flatten(-3), gt.flatten(-3)).tolist()]
        return ret_metrics
    
    def __call__(self, gen_samples, controls):
        """
        gen_samples: List[Image]
        controls: np.ndarray
        """
        preds = self.predict_control(gen_samples)
        ret_metrics = self.eval(controls, preds)
        return ret_metrics

    
class SegmentationControlEvaluator:
    def __init__(self, eval_dtype=torch.float16, ignore_index=0):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.eval_dtype = eval_dtype
        model_path = "facebook/maskformer-swin-large-ade"
        self.processor = MaskFormerImageProcessor.from_pretrained(model_path)
        self.model = MaskFormerForInstanceSegmentation.from_pretrained(model_path).to(self.device, eval_dtype).eval()
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
        inputs = self.preprocess(images).to(self.device, self.eval_dtype)
        with torch.cuda.amp.autocast(dtype=self.eval_dtype):
            outputs = self.model(**inputs)
        res = self.postprocess(outputs, len(images), images[0].size[::-1]) + 1 # This is because ids range from [1..N] in ADE20K put [0..N-1] in the MaskFormer
        return res


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
    

class PoseEvaluator:
    def __init__(self, eval_dtype=torch.float16):
        self.eval_dtype= eval_dtype
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.yolo = YOLO("yolo11m-pose.pt")
        self.metrics = ["mAP", "AP@50", "AP@75"]

    def preprocess(self, images):
        """
        images: List[PIL.Image]
        """
        x = [np.array(im) for im in images]
        return x
    
    def postprocess(self, out):
        """
        out: List[ultralytics.engine.results.Results]
        """
        cocokp_outputs = []
        for result in out:
            cocokp_output = []
            for kpdts, conf in zip(result.keypoints.xy, result.boxes.conf):
                person_dict = dict(
                    image_id=1,
                    category_id=1,
                    iscrowd=0,
                    score=float(conf)
                )
                keypoints = []
                for kp in kpdts:
                    keypoints.append([kp[0].item(), kp[1].item(), 1])
                
                person_dict["keypoints"] = np.array(keypoints).flatten().tolist()
                cocokp_output.append(person_dict)
            
            cocokp_outputs.append(cocokp_output)

        return cocokp_outputs

    @torch.no_grad()
    def predict_control(self, images):
        x = self.preprocess(images)
        with torch.cuda.amp.autocast(dtype=self.eval_dtype):
            out = self.yolo(x, verbose=False)
        detections = self.postprocess(out)
        
        return detections
    
    def evaluate_pose_estimation(
        self,
        gt: Dict[str, Any],
        pred: List[Dict[str, Any]],
    ):  
        
        gtKps = COCO(gt)
        if len(pred) == 0:
            return 0., 0., 0.
        
        dtKps = gtKps.loadRes(pred)
        coco_eval = COCOeval(gtKps, dtKps, iouType="keypoints")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        
        mAP = coco_eval.stats[0].item()
        AP50 = coco_eval.stats[1].item()
        AP75 = coco_eval.stats[2].item()

        return mAP, AP50, AP75


    def eval(self, gts, preds):
        """
        gts: List[List[Dict]]
        preds: List[List[Dict]]
        """
        assert len(gts) == len(preds)

        ret_metrics = {
            "mAP": [],
            "AP@50": [],
            "AP@75": []
        }

        for gt, pred in zip(gts, preds):
            mAP, AP50, AP75 = self.evaluate_pose_estimation(gt, pred)
            ret_metrics["mAP"].append(mAP * 100)
            ret_metrics["AP@50"].append(AP50 * 100)
            ret_metrics["AP@75"].append(AP75 * 100)
            
        return ret_metrics
    
    def __call__(self, gen_samples, gt_detections):
        """
        gen_samples: List[Image]
        controls: np.ndarray
        """
        pred_detections = self.predict_control(gen_samples)
        ret_metrics = self.eval(gt_detections, pred_detections)
        return ret_metrics