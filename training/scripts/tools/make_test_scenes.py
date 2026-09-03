"""Compose phone-photo-like test scenes from Scryfall card images.

Pastes one or more reference card images, tilted, onto a wood-like
background so the browser detector (YOLO OBB) and the scan flow can be
exercised without a phone: feed the output to the scanner's photo upload
(see ``app/scripts/screenshot.mjs`` for driving that headlessly).

Usage::

    conda run -n learning python training/scripts/tools/make_test_scenes.py \\
        --out /tmp/scenes --single tsp-157.jpg:12 --single m11-100.jpg:-9 \\
        --binder blb-280.jpg,zen-21.jpg,tsp-157.jpg,10e-100.jpg --empty

``--single NAME[:ANGLE]`` writes ``single-<NAME>.jpg`` with one card tilted
by ANGLE degrees (counter-clockwise positive). ``--binder`` takes four
file names and writes ``binder.jpg`` as a 2x2 page. ``--empty`` writes a
background with no card. Names are files under the Scryfall image
directory (``training/_data/scryfall/images``).
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import config  # noqa: E402

CARD_W, CARD_H = 488, 680


def background(w: int, h: int, rng: random.Random) -> Image.Image:
    bg = Image.new("RGB", (w, h), (112, 84, 60))
    d = ImageDraw.Draw(bg)
    for y in range(0, h, 7):
        d.line(
            [(0, y), (w, y + rng.randint(-3, 3))],
            fill=(rng.randint(95, 125), rng.randint(70, 92), rng.randint(48, 66)),
            width=3,
        )
    return bg.filter(ImageFilter.GaussianBlur(1.2))


def place(bg: Image.Image, image_path: Path, box_w: int, x: int, y: int, angle: float) -> None:
    card = Image.open(image_path).convert("RGB").resize((box_w, int(box_w * CARD_H / CARD_W)))
    rot = card.rotate(angle, expand=True, resample=Image.BICUBIC)
    mask = Image.new("L", card.size, 255).rotate(angle, expand=True, resample=Image.BICUBIC)
    bg.paste(rot, (x, y), mask)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--images", type=Path, default=config.SCRYFALL_IMAGE_PATH, help="Scryfall image directory")
    parser.add_argument("--single", action="append", default=[], metavar="NAME[:ANGLE]")
    parser.add_argument("--binder", metavar="A,B,C,D", help="Four image names for a 2x2 binder page")
    parser.add_argument("--empty", action="store_true", help="Also write a scene with no card")
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--quality", type=int, default=88)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    for spec in args.single:
        name, _, angle = spec.partition(":")
        bg = background(1200, 1600, rng)
        place(bg, args.images / name, 560, 300, 330, float(angle or 0))
        out = args.out / f"single-{Path(name).stem}.jpg"
        bg.save(out, quality=args.quality)
        print(out)

    if args.binder:
        names = args.binder.split(",")
        if len(names) != 4:
            raise SystemExit("--binder needs exactly four names")
        bg = background(1600, 1600, rng)
        slots = [(120, 90, -3), (830, 110, 4), (110, 820, 2), (850, 800, -5)]
        for name, (x, y, angle) in zip(names, slots):
            place(bg, args.images / name, 560, x, y, angle)
        out = args.out / "binder.jpg"
        bg.save(out, quality=args.quality)
        print(out)

    if args.empty:
        out = args.out / "empty.jpg"
        background(1200, 1600, rng).save(out, quality=args.quality)
        print(out)


if __name__ == "__main__":
    main()
