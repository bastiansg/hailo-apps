import numpy as np

from common.logger import get_logger

from hailo_apps.servos import ServoAngles
from hailo_apps.meta.interfaces import (
    RotatorApp,
    RotatorParams,
    Centroid,
    ImageSize,
)


logger = get_logger(__name__)


class FaceTracker(RotatorApp["FaceTracker"]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
        image_size: ImageSize,
        model_url: str = "https://hub.degirum.com/zoo/v1/public/models/degirum/hailo/yolov8n_relu6_face--640x640_quant_hailort_hailo8l_1",
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
        detection_history_length: int = 0,
    ):
        super().__init__(
            model_url=model_url,
            image_size=image_size,
            init_servo_angles=init_servo_angles,
            rotator_params=rotator_params,
            debug_mode=debug_mode,
            debug_path=debug_path,
            detection_history_length=detection_history_length,
        )

    def get_centroid(self, np_image: np.ndarray) -> Centroid | None:
        detection = self.model(np_image)
        results = detection.results
        if not len(results):
            return

        x1, y1, x2, y2 = results[0]["bbox"]  # Only use the first result.
        centroid = Centroid(
            x=int((x1 + x2) / 2),
            y=int((y1 + y2) / 2),
        )

        return centroid
