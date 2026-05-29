import torch
import numpy as np
import einops
from annotator.midas.api import MiDaSInference
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
        # x = torch.stack(x)
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
        out = []
        with torch.cuda.amp.autocast(dtype=self.eval_dtype):
            for x_ in x:
                out_ = self.midas(x_.unsqueeze(0))
                out.append(out_)
        out = torch.cat(out, dim=0)
        depth = self.postprocess(out)
        
        return depth

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