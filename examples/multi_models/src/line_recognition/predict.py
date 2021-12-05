from dotmap import DotMap

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from scipy.special import softmax

from src.logger import logger
from src.line_recognition.model import CRNN


class LineRecognizer:
    def __init__(self, **kwargs):
        self.config = DotMap(kwargs)
        self.input_channel = 3
        self.h_std = 32
        self.hidden_size = 256
        self.num_classes = 117
        self.transform = A.Compose([
            A.Normalize(mean=(0., 0., 0.), std=(1, 1, 1)),
            ToTensorV2()])
        self.label_info_path = "src/line_recognition/label_info.txt"
        self.blank = "---"
        self.label_info = self.index2char()
        self.model = self.load_model()

    def index2char(self):
        with open(self.label_info_path, "r") as f:
            characters = f.read()
            f.close()
        characters = characters.replace("\n", "")
        index_label = {i + 1: char for i, char in enumerate(characters)}
        index_label[0] = self.blank
        return index_label

    def load_model(self):
        logger.info("*** LOADING RECOGNITION MODEL" + self.config.model_path)
        model = CRNN(self.input_channel, self.hidden_size, self.num_classes)
        state_dict = torch.load(self.config.model_path, map_location="cpu")["state_dict"]
        model.load_state_dict(state_dict)
        model.eval()
        model.to(self.config.device)
        return model

    def preprocess(self, data):
        h0, w0 = data.shape[:2]  # RGB
        w_new = int(self.h_std * w0 / h0)
        image = cv2.resize(data, (w_new, self.h_std))
        image = self.transform(image=image)["image"]
        image = torch.unsqueeze(image, dim=0)
        image = torch.FloatTensor(image)
        image = image.to(self.config.device)
        return image

    def inference(self, data):
        with torch.no_grad():
            y_pred = self.model(data)
        return y_pred

    def postprocess(self, data):
        data = data.cpu().numpy()
        data = np.squeeze(data, axis=1)
        scores = softmax(data, axis=1)
        data = np.argmax(data, axis=1)
        scores = np.max(scores, axis=1)
        data = data.tolist()
        text_sequence = []
        text_scores = []
        for i in range(len(data)):
            pred = data[i]
            if i == 0:
                text_sequence.append(self.label_info[pred])
            elif i != 0 and pred != data[i - 1]:
                text_sequence.append(self.label_info[pred])
            text_scores.append(scores[i])
        remove_blank = [t for t in text_sequence if t != self.blank]
        last_sequence = "".join(remove_blank)
        text_scores = np.mean(text_scores)
        return last_sequence, text_scores

    def process(self, data):
        data = self.preprocess(data)
        data = self.inference(data)
        data = self.postprocess(data)
        return data
