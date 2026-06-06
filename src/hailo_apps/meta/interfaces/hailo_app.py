import os
import pooch

from urllib.parse import urlparse
from typing import Generic, TypeVar

from .hailo_detection_model import HailoDetectionModel


HAILO_MODEL_URLS = {
    "scrfd_10g_h8l.hef": (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
        "ModelZoo/Compiled/v2.14.0/hailo8l/scrfd_10g.hef"
    ),
    "scrfd_2.5g_h8l.hef": (
        "https://hailo-model-zoo.s3.eu-west-2.amazonaws.com/"
        "ModelZoo/Compiled/v2.14.0/hailo8l/scrfd_2.5g.hef"
    ),
}

T = TypeVar("T", bound="HailoApp")


class HailoApp(Generic[T]):  # type: ignore
    def __init__(
        self,
        model_url: str,
        model_download_path: str = "/resources/cache/models",
    ):
        model_path = self.get_model_path(
            model_url=model_url,
            model_download_path=model_download_path,
        )

        model_name = os.path.basename(model_url)
        self.model = HailoDetectionModel(
            model_dir=model_path,
            model_name=model_name,
        )

    @staticmethod
    def get_model_path(model_url: str, model_download_path: str) -> str:
        if os.path.exists(model_url):
            return model_url

        model_name = os.path.basename(model_url)
        if model_name in HAILO_MODEL_URLS:
            return pooch.retrieve(
                url=HAILO_MODEL_URLS[model_name],
                known_hash=None,
                fname=model_name,
                path=model_download_path,
            )

        parsed_model_url = urlparse(model_url)
        if parsed_model_url.scheme == "":
            raise FileNotFoundError(f"Model file not found: {model_url}")

        if model_url.endswith(".hef"):
            return pooch.retrieve(
                url=model_url,
                known_hash=None,
                path=model_download_path,
            )

        file_paths = pooch.retrieve(
            url=model_url,
            known_hash=None,
            path=model_download_path,
            processor=pooch.Unzip(),
        )

        return os.path.dirname(file_paths[0])
