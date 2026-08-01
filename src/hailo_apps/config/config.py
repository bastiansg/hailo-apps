from pydantic import StrictStr
from pydantic_settings import BaseSettings


class Config(BaseSettings):
    image_format: StrictStr = "BGR888"


config = Config()
