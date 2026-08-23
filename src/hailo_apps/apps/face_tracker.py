from time import sleep

import numpy as np
from libcamera import Transform

from hailo_apps.config import config
from hailo_apps.meta.interfaces import (
    Centroid,
    ImageSize,
    RotatorApp,
    RotatorParams,
)
from hailo_apps.meta.interfaces.picam_app import CAMERA_CONTROLS
from hailo_apps.servos import ServoAngles


class FaceTracker(RotatorApp["FaceTracker"]):
    def __init__(
        self,
        init_servo_angles: ServoAngles,
        rotator_params: RotatorParams,
        image_size: ImageSize | None = None,
        capture_size: ImageSize | None = None,
        final_capture_y_angle_offset: int = 0,
        # model_name: str = "scrfd_10g_h8l.hef",
        model_name: str = "scrfd_2.5g_h8l.hef",
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
        history_length: int = 0,
        min_score: float = 0.0,
    ):
        image_size = image_size or ImageSize()
        super().__init__(
            model_url=f"{config.base_model_url}/{model_name}",
            image_size=image_size,
            init_servo_angles=init_servo_angles,
            rotator_params=rotator_params,
            debug_mode=debug_mode,
            debug_path=debug_path,
            history_length=history_length,
        )

        self.min_score = min_score
        self.capture_size = capture_size or image_size
        self.final_capture_y_angle_offset = final_capture_y_angle_offset
        self.final_capture: np.ndarray | None = None

    def run(self) -> None:
        self.final_capture = None
        super().run()

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
        final_y_angle = max(
            self.rotator_params.min_y_angle,
            min(
                self.y_angle + self.final_capture_y_angle_offset,
                self.rotator_params.max_y_angle,
            ),
        )

        self.servos.set_angles(
            servo_angles=ServoAngles(
                x=self.x_angle,
                y=final_y_angle,
            )
        )

        sleep(0.5)
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

        self.final_capture = self.picam.switch_mode_and_capture_array(
            capture_configuration,
            "main",
        )
