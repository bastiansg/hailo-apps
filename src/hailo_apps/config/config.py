from pydantic import StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    image_format: StrictStr = "BGR888"
    base_model_url: StrictStr = "/usr/share/hailo-models"


config = Config()
