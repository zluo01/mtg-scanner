from PIL import Image

from typing import Tuple


class PadToSquare:
    def __init__(self, fill_color: Tuple[int, int, int] = (114, 114, 114)) -> None:
        self.fill_color = fill_color

    def __call__(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        max_dim = max(w, h)

        new_img = Image.new("RGB", (max_dim, max_dim), self.fill_color)
        left = (max_dim - w) // 2
        top = (max_dim - h) // 2
        new_img.paste(img, (left, top))

        return new_img
