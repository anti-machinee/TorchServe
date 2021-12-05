from dotmap import DotMap

import cv2
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

from src.logger import logger
from src.line_detection.model import ResDCN
from src.line_detection.utils import Utils


class LineDetector:
    def __init__(self, **kwargs):
        self.config = DotMap(kwargs)
        self.num_layer = 50
        self.num_classes = 20
        self.input_size = (511, 511)
        self.transform = A.Compose([
            A.Normalize(mean=(0., 0., 0.), std=(1, 1, 1)),
            ToTensorV2()])

        self.model = self.load_model()
        self.utils = Utils()
        self.image0 = None
        self.scale_x, self.scale_y = None, None

    def load_model(self):
        logger.info("*** LOADING DETECTION MODEL" + self.config.model_path)
        model = ResDCN(num_layer=self.num_layer, num_classes=self.num_classes)
        state_dict = torch.load(self.config.model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.config.device)
        return model

    def preprocess(self, data):
        self.image0 = data
        image = self.image0.copy()
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, self.input_size)
        image = self.transform(image=image)["image"]
        image = torch.unsqueeze(image, dim=0)
        image = image.to(self.config.device)

        h0, w0 = self.image0.shape[:2]
        self.scale_x, self.scale_y = self.input_size[1] / w0, self.input_size[0] / h0
        return image

    def inference(self, data):
        with torch.no_grad():
            y_pred = self.model(data)
        return y_pred

    def postprocess(self, data):
        heatmap = data["hm"].sigmoid_()
        width_height = data["wh"]
        offset = data["reg"]
        decoded_output = self.utils.decode(heatmap, width_height, offset, K=30)[0]

        boxes = []
        pred_indexes = []
        scores = []
        for det in decoded_output:
            det[:4] = (det[:4] * (511 / 128))
            xmin = int(det[0] / self.scale_x)
            ymin = int(det[1] / self.scale_y)
            xmax = int(det[2] / self.scale_x)
            ymax = int(det[3] / self.scale_y)
            score = det[4]
            pred_idx = det[5]
            if score > self.config.score:
                boxes.append([xmin, ymin, xmax, ymax])
                pred_indexes.append(pred_idx)
                scores.append(score)
        boxes, pred_indexes, scores = self.utils.nms_fast(boxes, pred_indexes, scores, self.config.overlap_thresh)
        return boxes, pred_indexes

    def shareprocess(self, boxes):
        self.image0 = cv2.cvtColor(self.image0, cv2.COLOR_BGR2RGB)
        box_images = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box
            assert xmin < xmax
            assert ymin < ymax
            box_images.append(self.image0[ymin: ymax, xmin: xmax])  # RGB
        return box_images

    def process(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        boxes, pred_indexes = self.postprocess(data)
        box_images = self.shareprocess(boxes)
        return boxes, pred_indexes, box_images
