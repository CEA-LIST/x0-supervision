import torch
import numpy as np
from ultralytics import YOLO
from cocoapi.PythonAPI.pycocotools.coco import COCO
from cocoapi.PythonAPI.pycocotools.cocoeval import COCOeval
from typing import List, Dict, Any


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