#!/usr/bin/env python3
"""Render benchmark PDFs to page images data/<dataset>/img/{pdf_stem}_{page}.jpg (1-indexed).

Settings used for the paper:
    ViDoSeek     : --backend pymupdf --dpi 150
    MMLongBench  : --backend pdf2image            (poppler, its default 200 dpi)
SlideVQA ships page images (see data_prep/convert_slidevqa_to_vidorag.py).

Usage:
    python data_prep/pdf2images.py --dataset ViDoSeek --backend pymupdf --dpi 150 --workers 8
    python data_prep/pdf2images.py --dataset MMLongBench --backend pdf2image
"""
import argparse
import multiprocessing
import os

from tqdm import tqdm

PROJECT_ROOT = os.environ.get('VIDORAG_PROJECT_ROOT') or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get('VIDORAG_DATA_ROOT', os.path.join(PROJECT_ROOT, 'data'))


def convert_one(item):
    """Convert a single PDF. Returns (pdf_name, num_pages, error)."""
    pdf_dir, img_dir, pdf_name, backend, dpi = item
    pdf_path = os.path.join(pdf_dir, pdf_name)
    stem = pdf_name.rsplit('.pdf', 1)[0]
    try:
        if backend == 'pdf2image':
            from pdf2image import convert_from_path
            images = convert_from_path(pdf_path, thread_count=4)
            for i, image in enumerate(images):
                image.save(os.path.join(img_dir, f'{stem}_{i + 1}.jpg'), 'JPEG')
            return pdf_name, len(images), None
        import fitz  # PyMuPDF, imported per worker to avoid fork issues
        doc = fitz.open(pdf_path)
        n_pages = len(doc)
        mat = fitz.Matrix(dpi / 72, dpi / 72)
        for page_idx in range(n_pages):
            out_path = os.path.join(img_dir, f'{stem}_{page_idx + 1}.jpg')
            if not os.path.exists(out_path):
                doc[page_idx].get_pixmap(matrix=mat).save(out_path)
        doc.close()
        return pdf_name, n_pages, None
    except Exception as e:  # keep going; failures are listed at the end
        return pdf_name, 0, str(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--backend', choices=['pymupdf', 'pdf2image'], default='pymupdf')
    ap.add_argument('--dpi', type=int, default=150, help='pymupdf only')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()

    pdf_dir = os.path.join(DATA_ROOT, args.dataset, 'pdf')
    img_dir = os.path.join(DATA_ROOT, args.dataset, 'img')
    os.makedirs(img_dir, exist_ok=True)
    pdf_files = sorted(f for f in os.listdir(pdf_dir) if f.endswith('.pdf'))
    done = {f.rsplit('_', 1)[0] for f in os.listdir(img_dir) if f.endswith('.jpg')}
    todo = [f for f in pdf_files if f.rsplit('.pdf', 1)[0] not in done]
    print(f'{len(pdf_files)} PDFs, {len(todo)} to convert -> {img_dir}')

    items = [(pdf_dir, img_dir, f, args.backend, args.dpi) for f in todo]
    with multiprocessing.Pool(args.workers) as pool:
        results = list(tqdm(pool.imap_unordered(convert_one, items), total=len(items), desc='PDF->IMG'))
    errors = [(name, err) for name, _, err in results if err]
    for name, err in errors:
        print(f'  ERROR {name}: {err}')
    print(f'{sum(n for _, n, _ in results)} pages written, {len(errors)} errors')


if __name__ == '__main__':
    main()
