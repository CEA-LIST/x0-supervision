import torch
import numpy as np
import einops
from transformers import MaskFormerForInstanceSegmentation, MaskFormerImageProcessor
from annotator.canny import CannyDetector
from torchmetrics.image import StructuralSimilarityIndexMeasure, PeakSignalNoiseRatio
from torchmetrics.classification import BinaryF1Score


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