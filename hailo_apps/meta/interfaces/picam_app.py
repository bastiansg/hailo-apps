import numpy as np

from PIL import Image
from threading import Thread, Lock
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Callable, Any

from picamera2 import Picamera2
from pydantic import BaseModel, PositiveInt

from common.utils.path import create_path

from .hailo_app import HailoApp


T = TypeVar("T", bound="PicamApp")
F = TypeVar("F", bound=Callable[..., None])


# TODO: Move this function to common
def threaded(func: Callable[..., None]) -> Callable[..., None]:
    def wrapper(*args: Any, **kwargs: Any) -> None:
        thread = Thread(target=func, args=args, kwargs=kwargs)
        thread.daemon = True
        thread.start()

    return wrapper


class ImageSize(BaseModel):
    width: PositiveInt = 640
    height: PositiveInt = 640


class PicamApp(HailoApp["PicamApp"], ABC, Generic[T]):
    def __init__(
        self,
        model_url: str,
        image_size: ImageSize,
        rotation_180: bool = True,
        debug_mode: bool = False,
        debug_path: str = "/resources/debug/images",
    ):
        super().__init__(
            model_url=model_url,
        )

        self.image_size = image_size
        self.rotation_180 = rotation_180
        self.debug_mode = debug_mode

        create_path(path=debug_path)
        self.debug_image_path = f"{debug_path}/debug-image.jpg"

        self.picam = self.get_picam(image_size=image_size)

        self.mutex = Lock()
        self.is_active = False

    @staticmethod
    def get_picam(image_size: ImageSize) -> Picamera2:
        picam = Picamera2()
        config = picam.create_video_configuration(
            main={
                "size": (
                    image_size.width,
                    image_size.height,
                ),
            }
        )

        picam.configure(config)
        return picam

    @abstractmethod
    def on_frame(self, np_image: np.ndarray) -> None:
        pass

    @threaded
    def run(self) -> None:
        self.mutex.acquire()
        self.picam.start()

        self.is_active = True
        while self.is_active:
            np_image = self.picam.capture_array()
            np_image = np_image[:, :, :3]  # Remove alpha channel.

            if self.rotation_180:
                np_image = np_image[::-1, ::-1]  # 180-degree rotation.

            if self.debug_mode:
                pil_image = Image.fromarray(np_image)
                pil_image.save(self.debug_image_path)
                self.is_active = False

            self.on_frame(np_image=np_image)

        self.picam.stop()
        self.mutex.release()

    def stop(self) -> None:
        self.is_active = False
