from PIL import Image, ImageDraw
from hailo_apps.meta.data_models import BBox


def draw_bboxes(
    pil_image: Image.Image,
    bboxes: list[BBox],
    color: str = "blue",
    width: int = 3,
) -> None:
    draw = ImageDraw.Draw(pil_image)
    for bbox in bboxes:
        draw.rectangle(
            [
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
            ],
            width=width,
            outline=color,
        )
