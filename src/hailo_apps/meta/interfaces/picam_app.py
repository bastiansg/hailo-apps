from abc import ABC, abstractmethod
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Generic, TypeVar

import numpy as np
from libcamera import Transform, controls
from picamera2 import Picamera2
from PIL import Image
from pydantic import BaseModel, PositiveInt

from hailo_apps.config import config

from .hailo_app import HailoApp

T = TypeVar("T", bound="PicamApp")

CAMERA_CONTROLS = {
    "AfMode": controls.AfModeEnum.Continuous,
    "AfRange": controls.AfRangeEnum.Normal,
    "AfSpeed": controls.AfSpeedEnum.Normal,
    "AeMeteringMode": controls.AeMeteringModeEnum.CentreWeighted,
    "AwbMode": controls.AwbModeEnum.Auto,
}


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
        self.thread: Thread | None = None
        self.thread_error: Exception | None = None

    def __del__(self) -> None:
        picam = getattr(self, "picam", None)
        if picam is None:
            return

        picam.stop()
        picam.close()

    @staticmethod
    def get_picam(image_size: ImageSize) -> Picamera2:
        picam = Picamera2()
        camera_configuration = picam.create_video_configuration(
            main={
                "format": config.image_format,
                "size": (
                    image_size.width,
                    image_size.height,
                ),
            },
            transform=Transform(
                hflip=True,
                vflip=True,
            ),
            controls=CAMERA_CONTROLS,
            buffer_count=1,
        )

        picam.configure(camera_configuration)
        return picam

    @abstractmethod
    def on_frame(self, np_image: np.ndarray) -> None:
        pass

    def run(self) -> None:
        if self.thread is not None and self.thread.is_alive():
            raise RuntimeError("camera is already running")

        self.stop_event.clear()
        self.thread_error = None
        self.thread = Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self) -> None:
        with self.mutex:
            self.picam.start()

            try:
                while not self.stop_event.is_set():
                    np_image = self.picam.capture_array()

                    if self.debug_mode:
                        pil_image = Image.fromarray(np_image)
                        pil_image.save(self.debug_image_path)
                        break

                    self.on_frame(np_image=np_image)  # type: ignore

                self.before_stop()
            except Exception as error:
                self.thread_error = error
            finally:
                self.picam.stop()

    def before_stop(self) -> None:
        pass

    def stop(self) -> None:
        self.stop_event.set()
        thread = self.thread
        if thread is None:
            return

        thread.join()
        self.thread = None

        if self.thread_error is not None:
            raise self.thread_error
