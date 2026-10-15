#!/usr/bin/env python3
"""Print-ready A4 sheets for a second (independent) annotation of the 50 validation images (analysis A3).

Each PDF holds one A4 portrait page with two images stacked vertically, printed as large as the page
allows, labelled with the image file name (public dataset ids, no PHI). A separate instruction sheet explains what to
draw. Output: paper/annotation_sheets/<image1>__<image2>.pdf (25 files), A3_all_sheets.pdf,
A3_instructions.pdf and manifest.csv (pdf, slot, image file).

    python scripts/paper/make_annotation_sheets.py [--split data/splits/source_grouped_v1.json] [--out paper/annotation_sheets]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from PIL import Image

matplotlib.use("Agg")
ROOT = Path(__file__).resolve().parents[2]
A4 = (8.27, 11.69)  # inches, portrait

INSTRUCTIONS = [
    "Independent annotation of human blastocyst images (validation set: 50 images, 13 pages)",
    "",
    "Each image is printed three times in one row: draw ONLY the column's structure on each copy",
    "(left copy ZP, middle copy TE, right copy ICM), with a fine pen:",
    "  1. ZP  - zona pellucida: draw BOTH the outer and the inner edge of the ring.",
    "  2. TE  - trophectoderm: draw its inner edge (boundary with the blastocoel cavity).",
    "           Its outer edge is the inner edge of the ZP you already drew.",
    "  3. ICM - inner cell mass: outline the compact cell mass. Where it meets the TE,",
    "           draw the boundary where you judge the cell membership changes.",
    "  4. Absent / not visible / unsure: write '?' on that copy instead of guessing.",
    "",
    
    "Do not consult the original dataset labels or another annotator. Sheet and slot codes",
    "(top-left of each image) are the only identifiers; keep them visible when scanning.",
    "Scan at 300 dpi or photograph flat, one page per file, named as the PDF file",
    "(the two image names printed on it).",
    "",
    "Annotator: ____________________   Date: ____________   Experience (years): ______",
]

INSTRUCTIONS_VI = [
    "Gán nhãn độc lập ảnh phôi nang người (tập kiểm định: 50 ảnh, 13 trang)",
    "",
    "Mỗi ảnh được in ba lần trên một hàng: CHỈ vẽ cấu trúc của cột đó lên từng bản",
    "(cột trái ZP, cột giữa TE, cột phải ICM), dùng bút nét mảnh:",
    "  1. ZP  - màng trong suốt: vẽ CẢ đường biên ngoài và biên trong của vành.",
    "  2. TE  - lá nuôi phôi: vẽ đường biên trong (ranh giới với khoang phôi).",
    "           Biên ngoài của TE chính là biên trong của ZP đã vẽ.",
    "  3. ICM - khối tế bào trong: khoanh khối tế bào đặc. Chỗ tiếp giáp với TE,",
    "           vẽ ranh giới tại nơi bạn cho rằng tế bào đổi thuộc lớp khác.",
    "  4. Không có / không thấy / không chắc: ghi dấu '?' lên bản đó thay vì đoán.",
    "",
    
    "Không tham khảo nhãn gốc của bộ dữ liệu hay người gán nhãn khác. Tên ảnh in ở góc",
    "trên trái mỗi ảnh là định danh; giữ nguyên khi quét.",
    "Quét 300 dpi hoặc chụp thẳng, mỗi trang một file, đặt tên theo tên file PDF (tên hai ảnh).",
    "",
    "Người gán nhãn: ______________   Ngày: __________   Số năm kinh nghiệm: ______",
]


def load_gray(path: Path) -> np.ndarray:
    return np.array(Image.open(path).convert("L"))


def draw_page(pdf: PdfPages, images: list[tuple[str, np.ndarray]], sheet: int) -> None:
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    # two slots, each ~48 % of the page height, with margins for hand notes on the right
    slots = [(0.06, 0.53, 0.72, 0.43), (0.06, 0.05, 0.72, 0.43)]
    for (code, img), (x, y, w, h) in zip(images, slots):
        ax = fig.add_axes([x, y, w, h])
        ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bicubic", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_linewidth(0.4); s.set_color("#888888")
        fig.text(x, y + h + 0.008, code, fontsize=9, color="#333333", family="monospace")
        fig.text(x + w + 0.02, y + h - 0.02, "Notes:", fontsize=8, color="#333333")
        for i in range(6):
            fig.add_artist(plt.Line2D([x + w + 0.02, 0.97], [y + h - 0.06 - 0.045 * i] * 2, color="#bbbbbb", lw=0.5))
        fig.text(x + w + 0.02, y + 0.01, "ZP ?  TE ?  ICM ?", fontsize=7, color="#666666")
    fig.text(0.5, 0.975, f"Blastocyst annotation   -   page {sheet} of 25   -   see instruction page", ha="center", fontsize=8, color="#555555")
    pdf.savefig(fig); plt.close(fig)


COLS = ("ZP", "TE", "ICM")


def draw_grid_page(pdf: PdfPages, images: list[tuple[str, np.ndarray]], page: int, n_pages: int) -> None:
    """A4 portrait: 4 rows (4 images) x 3 columns (the same image repeated; ZP / TE / ICM header above every row)."""
    fig = plt.figure(figsize=A4)
    fig.patch.set_facecolor("white")
    left, right, top, bottom = 0.05, 0.98, 0.94, 0.02
    rows, cols = 4, 3
    gap_x = 0.012
    header = 0.042  # per-row band holding the image name (left) and the column labels
    rh = (top - bottom) / rows
    ih = rh - header - 0.006
    cw = (right - left - gap_x * (cols - 1)) / cols
    for r, (code, img) in enumerate(images):
        y_top = top - r * rh
        y_img = y_top - header - ih
        fig.text(left, y_top - 0.006, code, fontsize=7.5, color="#333333", family="monospace", va="top")
        for c, name in enumerate(COLS):
            x = left + c * (cw + gap_x)
            fig.text(x + cw / 2, y_img + ih + 0.004, name, ha="center", va="bottom", fontsize=9, weight="bold", color="#222222")
            ax = fig.add_axes([x, y_img, cw, ih])
            ax.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="bicubic", aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_linewidth(0.4); sp.set_color("#888888")
    fig.text(0.5, 0.975, f"Blastocyst annotation   -   page {page} of {n_pages}   -   draw only the column's structure on each copy", ha="center", fontsize=7.5, color="#555555")
    pdf.savefig(fig); plt.close(fig)


def draw_instructions(pdf: PdfPages) -> None:
    for lines in (INSTRUCTIONS, INSTRUCTIONS_VI):
        fig = plt.figure(figsize=A4)
        y = 0.94
        for i, line in enumerate(lines):
            fig.text(0.07, y, line, fontsize=11 if i == 0 else 9, family="DejaVu Sans Mono" if (line.startswith("  ") and lines is INSTRUCTIONS) else "DejaVu Sans",
                     weight="bold" if i == 0 else "normal", va="top")
            y -= 0.03 if line else 0.018
        pdf.savefig(fig); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default=str(ROOT / "data/splits/source_grouped_v1.json"))
    ap.add_argument("--image-dir", default=str(ROOT / "dataset/source/images"))
    ap.add_argument("--out", default=str(ROOT / "paper/annotation_sheets"))
    ap.add_argument("--seed", type=int, default=7, help="shuffle order so consecutive sheets are not consecutive embryos")
    args = ap.parse_args()
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    ids = list(json.loads(Path(args.split).read_text())["val"])
    rng = np.random.default_rng(args.seed); rng.shuffle(ids)
    manifest = []
    with PdfPages(out / "A3_instructions.pdf") as pdf:
        draw_instructions(pdf)
    per_page = 4
    n_pages = (len(ids) + per_page - 1) // per_page
    with PdfPages(out / "A3_all_sheets.pdf") as all_pdf:
        draw_instructions(all_pdf)
        for page, start in enumerate(range(0, len(ids), per_page), start=1):
            group = ids[start:start + per_page]
            images = [(Path(n).stem, load_gray(Path(args.image_dir) / n)) for n in group]
            pdf_name = f"page{page:02d}__" + "__".join(Path(n).stem.replace(" ", "_") for n in group) + ".pdf"
            for row, n in enumerate(group, start=1):
                manifest.append({"pdf": pdf_name, "row": row, "code": Path(n).stem, "image": n})
            with PdfPages(out / pdf_name) as pdf:
                draw_grid_page(pdf, images, page, n_pages)
            draw_grid_page(all_pdf, images, page, n_pages)
    with (out / "manifest.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pdf", "row", "code", "image"]); w.writeheader(); w.writerows(manifest)
    print(f"{len(ids)} images -> {n_pages} PDFs (4 images x 3 copies per page) in {out}; manifest.csv lists pdf -> images")


if __name__ == "__main__":
    main()
