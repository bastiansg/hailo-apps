import numpy as np
import degirum as dg

from common.logger import get_logger
from hailo_apps.meta.interfaces import PicamApp, ImageSize


logger = get_logger(__name__)


class FaceDetector(PicamApp["FaceDetector"]):
    def __init__(self):
        super().__init__(image_size=ImageSize())
        self.model = dg.load_model(
            model_name="yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1",
            inference_host_address="@local",
            zoo_url="degirum/hailo",
        )

    def on_frame(self, np_image: np.ndarray) -> None:
        detection = self.model(np_image)
        face_bbox = detection.results[0]["bbox"]  # Only use the first result.

        logger.info(face_bbox)
