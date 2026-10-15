"""Overlay per-class binary masks onto source images for visual inspection."""
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

CLASS_COLORS = {
    "zp": (255, 0, 0),      # red
    "te": (0, 255, 0),      # green
    "bl": (0, 128, 255),    # blue
    "icm": (255, 255, 0),   # yellow
}
ALPHA = 0.45


def overlay_one(image_path: Path, mask_root: Path, out_path: Path) -> None:
    img = Image.open(image_path).convert("RGB")
    base = np.array(img).astype(np.float32)
    overlay = base.copy()

    for cls, color in CLASS_COLORS.items():
        mask_path = mask_root / cls / image_path.name
        if not mask_path.exists():
            continue
        mask = np.array(Image.open(mask_path).convert("L"))
        binm = mask > 127
        if not binm.any():
            continue
        color_arr = np.array(color, dtype=np.float32)
        overlay[binm] = overlay[binm] * (1 - ALPHA) + color_arr * ALPHA

    out_img = Image.fromarray(overlay.astype(np.uint8))

    legend_h = 24
    canvas = Image.new("RGB", (out_img.width, out_img.height + legend_h), (255, 255, 255))
    canvas.paste(out_img, (0, 0))
    draw = ImageDraw.Draw(canvas)
    x = 4
    for cls, color in CLASS_COLORS.items():
        draw.rectangle([x, out_img.height + 4, x + 14, out_img.height + 18], fill=color)
        draw.text((x + 18, out_img.height + 4), cls, fill=(0, 0, 0))
        x += 18 + 8 * len(cls) + 20

    canvas.save(out_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="dataset/source")
    parser.add_argument("--limit", type=int, default=0, help="0 = all images")
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    image_dir = source_dir / "images"
    mask_root = source_dir / "masks"
    out_dir = source_dir / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(image_dir.glob("*"))
    if args.limit:
        images = images[: args.limit]

    for i, image_path in enumerate(images, 1):
        out_path = out_dir / image_path.name
        overlay_one(image_path, mask_root, out_path)
        if i % 50 == 0 or i == len(images):
            print(f"[{i}/{len(images)}] {image_path.name}")

    print(f"Done. Wrote {len(images)} overlays to {out_dir}")


if __name__ == "__main__":
    main()
