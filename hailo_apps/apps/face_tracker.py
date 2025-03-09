import numpy as np
import degirum as dg

from common.logger import get_logger

from hailo_apps.servos import ServoAngles
from hailo_apps.meta.interfaces import RotatorApp, RotatorParams, Centroid


logger = get_logger(__name__)


class FaceTracker(RotatorApp["FaceTracker"]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params=RotatorParams,
    ):
        super().__init__(
            init_servo_angles=init_servo_angles,
            rotator_params=rotator_params,
        )

        self.model = dg.load_model(
            model_name="yolov8n_relu6_coco--640x640_quant_hailort_hailo8l_1",
            inference_host_address="@local",
            zoo_url="degirum/hailo",
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
