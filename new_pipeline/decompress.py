"""
Split a large .zst Reddit dump into smaller .zst files.

Each output file contains at most --max-lines JSON lines,
re-compressed with zstandard.

Usage:
    python split_zst.py RC_2018-01.zst --max-lines 5000000 --output-dir ./split/

This produces:
    ./split/RC_2018-01_part000.zst
    ./split/RC_2018-01_part001.zst
    ./split/RC_2018-01_part002.zst
    ...

Each part is a valid .zst file that collect_instances.py can process.
"""

import argparse
import io
import os
import logging
import zstandard as zstd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def split_zst(
    input_path: str,
    output_dir: str,
    max_lines: int = 5_000_000,
    compression_level: int = 3,
):
    os.makedirs(output_dir, exist_ok=True)
    basename = os.path.splitext(os.path.basename(input_path))[0]
    # Remove .zst extension if double-extension like .jsonl.zst
    if basename.endswith(".jsonl"):
        basename = basename[:-6]

    dctx = zstd.ZstdDecompressor(max_window_size=2147483648)
    fh = open(input_path, "rb")
    reader = dctx.stream_reader(fh, read_size=2**24)
    text_stream = io.TextIOWrapper(reader, encoding="utf-8", errors="ignore")

    part_num = 0
    line_count = 0
    total_lines = 0
    out_fh = None
    cctx = None
    compressor = None

    def open_new_part():
        nonlocal part_num, line_count, out_fh, cctx, compressor
        part_path = os.path.join(output_dir, f"{basename}_part{part_num:03d}.zst")
        logger.info(f"Writing {part_path}...")
        out_fh = open(part_path, "wb")
        cctx = zstd.ZstdCompressor(level=compression_level)
        compressor = cctx.stream_writer(out_fh)
        line_count = 0
        part_num += 1

    def close_current_part():
        nonlocal out_fh, compressor
        if compressor is not None:
            compressor.close()
        if out_fh is not None:
            out_fh.close()

    open_new_part()

    for line in text_stream:
        compressor.write(line.encode("utf-8"))
        line_count += 1
        total_lines += 1

        if line_count >= max_lines:
            close_current_part()
            open_new_part()

        if total_lines % 5_000_000 == 0:
            logger.info(f"  {total_lines:,} lines processed, on part {part_num}")

    close_current_part()
    text_stream.close()
    fh.close()

    logger.info(f"Done: {total_lines:,} lines split into {part_num} parts in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input", help="Input .zst file")
    parser.add_argument("--output-dir", default="./split/")
    parser.add_argument("--max-lines", type=int, default=5_000_000,
                        help="Max lines per output file")
    parser.add_argument("--compression-level", type=int, default=3)
    args = parser.parse_args()

    split_zst(args.input, args.output_dir, args.max_lines, args.compression_level)