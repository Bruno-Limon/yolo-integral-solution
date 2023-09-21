import os
import time

import cv2

import torch

from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import postprocess

IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

class Predictor(object):
    def __init__(
        self,
        model,
        exp,
        cls_names=COCO_CLASSES,
        trt_file=None,
        decoder=None,
        device="cpu",
        fp16=False,
        legacy=False,
    ):
        self.model = model
        self.cls_names = cls_names
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        self.preproc = ValTransform(legacy=legacy)

        if trt_file is not None:
            from torch2trt import TRTModule

            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))

            x = torch.ones(1, 3, exp.test_size[0], exp.test_size[1]).cuda()
            self.model(x)
            self.model = model_trt

    def inference(self, img):

        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        ratio = min(self.test_size[0] / img.shape[0], self.test_size[1] / img.shape[1])
        img_info["ratio"] = ratio

        img, _ = self.preproc(img, None, self.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()

        if self.device == "gpu":
            img = img.cuda()
            if self.fp16:
                img = img.half()  # to FP16

        with torch.no_grad():
            t0 = time.time()
            outputs = self.model(img)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes, self.confthre,
                                  self.nmsthre, class_agnostic=True)

            print("Infer time: {:.4f}s".format(time.time() - t0))

        return outputs, img_info

    def visual(self, output, img_info, cls_conf=0.35):

        class Object(object):
            pass

        detected_object = Object()

        ratio = img_info["ratio"]
        img = img_info["raw_img"]
        if output is None:
            return img
        output = output.cpu().numpy()

        bboxes_xyxy = output[:, 0:4]
        bboxes_xyxy /= ratio # preprocessing: resize
        bboxes_xyxy = bboxes_xyxy.astype(int).tolist()
        cls = (output[:, 6]).astype(int).tolist()
        scores = (output[:, 4] * output[:, 5]).tolist()

        detected_object.bbox = bboxes_xyxy
        detected_object.cls = cls
        detected_object.conf = scores

        return detected_object

def detect(model_name, exp, frame):

    # define arguments
    experiment_name = exp.exp_name
    device = "cpu"
    fp16 = False
    trt = False
    fuse = False
    legacy = False
    exp.test_conf = 0.4
    exp.nmsthre = 0.45
    exp.test_size = (640, 640)
    model = exp.get_model()

    file_name = os.path.join(exp.output_dir, experiment_name)
    # os.makedirs(file_name, exist_ok=True)

    if device == "gpu":
        model.cuda()
        if fp16:
            model.half()  # to FP16
    model.eval()

    ckpt_file = model_name
    if not trt:
        ckpt = torch.load(ckpt_file, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])

    if trt:
        assert not fuse, "TensorRT model is not support model fusing!"
        trt_file = os.path.join(file_name, "model_trt.pth")
        assert os.path.exists(
            trt_file
        ), "TensorRT model is not found!\n Run python3 tools/trt.py first!"
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model=model, exp=exp, cls_names=COCO_CLASSES, trt_file=trt_file, decoder=decoder, device=device, fp16=fp16, legacy=legacy)

    img = [frame]
    outputs, img_info = predictor.inference(img=img[0])
    result_image = predictor.visual(output=outputs[0], img_info=img_info, cls_conf=predictor.confthre)

    return result_image