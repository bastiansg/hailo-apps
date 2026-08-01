from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event, Lock
from time import sleep
from typing import Generic, TypeVar

import numpy as np
from libcamera import Transform, controls
from picamera2 import Picamera2
from PIL import Image
from pydantic import BaseModel, PositiveInt

from .hailo_app import HailoApp
from .utils import threaded

T = TypeVar("T", bound="PicamApp")


class ImageSize(BaseModel):
    width: PositiveInt = 640
    height: PositiveInt = 640


class PicamApp(HailoApp["PicamApp"], ABC, Generic[T]):  # type: ignore
    def __init__(
        self,
        model_url: str,
        image_size: ImageSize,
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
    ):
        super().__init__(
            model_url=model_url,
        )

        self.image_size = image_size
        self.debug_mode = debug_mode

        Path(debug_path).mkdir(parents=True, exist_ok=True)
        self.debug_image_path = f"{debug_path}/debug-image.jpg"

        self.picam = self.get_picam(image_size=image_size)

        self.mutex = Lock()
        self.stop_event = Event()

    def __del__(self) -> None:
        picam = getattr(self, "picam", None)
        if picam is None:
            return

        picam.stop()
        picam.close()

    @staticmethod
    def get_picam(image_size: ImageSize) -> Picamera2:
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={
                "format": "RGB888",
                "size": (
                    image_size.width,
                    image_size.height,
                ),
            },
            transform=Transform(
                hflip=True,
                vflip=True,
            ),
            buffer_count=1,
        )

        picam.configure(config)
        return picam

    @abstractmethod
    def on_frame(self, np_image: np.ndarray) -> None:
        pass

    @threaded
    def run(self) -> None:
        with self.mutex:
            self.stop_event.clear()
            self.picam.start()
            self.picam.set_controls(
                {
                    "AfMode": controls.AfModeEnum.Continuous,
                    "AfRange": controls.AfRangeEnum.Normal,
                    "AfSpeed": controls.AfSpeedEnum.Normal,
                    "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
                    "AwbMode": controls.AwbModeEnum.Auto,
                }
            )

            while not self.stop_event.is_set():
                np_image = self.picam.capture_array()

                if self.debug_mode:
                    pil_image = Image.fromarray(np_image)
                    pil_image.save(self.debug_image_path)
                    break

                self.on_frame(np_image=np_image)  # type: ignore

            self.picam.stop()

    def stop(self) -> None:
        self.stop_event.set()
        sleep(0.1)
