import numpy as np

from hailo_apps.servos import ServoAngles
from hailo_apps.meta.interfaces import (
    RotatorApp,
    RotatorParams,
    Centroid,
    ImageSize,
)


BASE_MODEL_URL = "/usr/share/hailo-models"


class FaceTracker(RotatorApp["FaceTracker"]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
        image_size: ImageSize,
        # model_name: str = "scrfd_10g_h8l.hef",
        model_name: str = "scrfd_2.5g_h8l.hef",
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
        history_length: int = 0,
        min_score: float = 0.0,
    ):
        super().__init__(
            model_url=f"{BASE_MODEL_URL}/{model_name}",
            image_size=image_size,
            init_servo_angles=init_servo_angles,
            rotator_params=rotator_params,
            debug_mode=debug_mode,
            debug_path=debug_path,
            history_length=history_length,
        )

        self.min_score = min_score

    def get_centroid(self, np_image: np.ndarray) -> Centroid | None:
        detection = self.model(np_image)
        results = detection.results
        if not len(results):
            return

        first_result = results[0]  # Only use the first result.
        if first_result["score"] < self.min_score:
            return

        x1, y1, x2, y2 = first_result["bbox"]
        centroid = Centroid(
            x=int((x1 + x2) / 2),
            y=int((y1 + y2) / 2),
        )

        return centroid
