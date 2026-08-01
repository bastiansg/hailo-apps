import numpy as np
from libcamera import Transform

from hailo_apps.config import config
from hailo_apps.meta.interfaces import (
    Centroid,
    HistoryItem,
    ImageSize,
    RotatorApp,
    RotatorParams,
)
from hailo_apps.meta.interfaces.picam_app import CAMERA_CONTROLS
from hailo_apps.servos import ServoAngles

BASE_MODEL_URL = "/usr/share/hailo-models"


class FaceTracker(RotatorApp["FaceTracker"]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
        image_size: ImageSize | None = None,
        capture_size: ImageSize | None = None,
        # model_name: str = "scrfd_10g_h8l.hef",
        model_name: str = "scrfd_2.5g_h8l.hef",
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
        history_length: int = 0,
        min_score: float = 0.0,
    ):
        image_size = image_size or ImageSize()
        if capture_size is not None and history_length == 0:
            raise ValueError(
                "history_length must be positive when capture_size is set"
            )

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
        self.capture_size = capture_size

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

    def before_stop(self) -> None:
        if self.capture_size is None or self.history.maxlen == 0:
            return

        capture_configuration = self.picam.create_still_configuration(
            main={
                "format": config.image_format,
                "size": (
                    self.capture_size.width,
                    self.capture_size.height,
                ),
            },
            transform=Transform(
                hflip=True,
                vflip=True,
            ),
            controls=CAMERA_CONTROLS,
            buffer_count=1,
        )

        np_image = self.picam.switch_mode_and_capture_array(
            capture_configuration,
            "main",
        )

        self.history.append(
            HistoryItem(
                np_image=np_image,
                centroid=self.get_centroid(np_image),
            )
        )
