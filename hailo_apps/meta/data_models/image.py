from pydantic import BaseModel, NonNegativeFloat


class BBox(BaseModel):
    x1: NonNegativeFloat
    y1: NonNegativeFloat
    x2: NonNegativeFloat
    y2: NonNegativeFloat
