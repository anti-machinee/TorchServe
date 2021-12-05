import os
import stat
import io
import zipfile

import cv2
import numpy as np
import torch
from ts.torch_handler.base_handler import BaseHandler


os.environ["CUDA_VISIBLE_DEVICES"] = "1"


class ModelHandler(BaseHandler):
    def __init__(self):
        super(ModelHandler, self).__init__()
        self.initialized = False
        self.metrics = None

    def initialize(self, context):
        print("Start initializing *****")
        # extract zip file
        properties = context.system_properties
        model_dir = properties.get("model_dir")
        try:
            with zipfile.ZipFile(model_dir + "/src.zip", "r") as zip_ref:
                zip_ref.extractall(model_dir)
        except Exception as e:  # if exists
            print(e)
            pass

        # build DCN
        st = os.stat("./src/build_dcn.sh")
        os.chmod("./src/build_dcn.sh", st.st_mode | stat.S_IEXEC)
        os.system("./src/build_dcn.sh")

        # initialize models
        from src.logger import logger
        from src.line_detection.predict import LineDetector
        from src.line_recognition.predict import LineRecognizer
        from src.post_process.post_process import Decoder
        from src import config

        logger.info("Start loading model")
        if not torch.cuda.is_available():
            logger.info("Using cpu device")
            self.device = torch.device("cpu")
        else:
            logger.info("Using gpu device")
            self.device = torch.device("cuda")
        det_path = os.path.join(model_dir, "best-ckp_det.pth")
        rec_path = os.path.join(model_dir, "best-ckp_rec.pth")
        self.det_model = LineDetector(model_path=det_path, device=self.device, score=config.thresh_det,
                                      overlap_thresh=config.overlap_thresh)
        self.rec_model = LineRecognizer(model_path=rec_path, device=self.device)
        self.decoder = Decoder()
        logger.info("Load model successfully")
        self.initialized = True

    def preprocess(self, data):
        row = data[0]
        id = row.get("id")
        image = row.get("image")
        image = io.BytesIO(image)
        image = cv2.imdecode(np.frombuffer(image.getbuffer(), np.uint8), -1)
        image = np.array(image)
        return image

    def inference(self, data, *args, **kwargs):
        # 1. classify id card

        # 2. detect line
        boxes, box_labels, box_images = self.det_model.process(data)

        # 3. recognize line
        texts = []
        text_scores = []
        for b_image in box_images:
            sequence, sequence_score = self.rec_model.process(b_image)
            texts.append(sequence)
            text_scores.append(sequence_score)
        out = {
            "type": "cccd_chip_f",
            "boxes": boxes,
            "box_labels": box_labels,
            "texts": texts,
            "text_scores": text_scores,
        }
        return out

    def postprocess(self, data):
        data = self.decoder.process(data)
        return data

    @staticmethod
    def _pack_return_message(code, data=None):
        from src import config
        response = {"code": code, "message": config.error_msg[code], "data": data}
        return [response]

    def handle(self, data, context):
        from src import config
        from src.logger import logger
        try:
            data = self.preprocess(data)
            data = self.inference(data)
            data = self.postprocess(data)
            print(type(data))
            return self._pack_return_message(config.SUCCESSFULLY, data)
        except Exception as e:
            logger.error(f"Error occurs on inference session: {e}", exc_info=True)
            return self._pack_return_message(config.INTERNAL_SERVER_ERROR)
