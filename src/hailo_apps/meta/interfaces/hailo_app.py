import os
import pooch

from typing import Generic, TypeVar
from .hailo_detection_model import HailoDetectionModel


T = TypeVar("T", bound="HailoApp")


class HailoApp(Generic[T]):  # type: ignore
    def __init__(
        self,
        model_url: str,
        model_download_path: str = "/resources/cache/models",
    ):
        model_path = self.download_model(
            model_url=model_url,
            model_download_path=model_download_path,
        )

        model_name = os.path.basename(model_url)
        self.model = HailoDetectionModel(
            model_dir=model_path,
            model_name=model_name,
        )

    @staticmethod
    def download_model(model_url: str, model_download_path: str) -> str:
        file_paths = pooch.retrieve(
            url=model_url,
            known_hash=None,
            path=model_download_path,
            processor=pooch.Unzip(),
            progressbar=True,
        )

        return os.path.dirname(file_paths[0])
