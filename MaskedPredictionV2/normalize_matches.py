import json
import argparse
from pathlib import Path

def normalize(input_path, output_path):
    total = 0
    kept = 0

    with open(input_path, "r", encoding="utf-8", errors="replace") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:

        for line in fin:
            total += 1
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except:
                continue

            # Already normalized
            if "sentence" in obj and "canonical_phrase" in obj:
                fout.write(json.dumps(obj) + "\n")
                kept += 1
                continue

            # Raw format
            if "context" in obj:
                sentence = str(obj["context"])
                phrase = str(obj.get("text") or obj.get("word") or "").strip()

                if not phrase:
                    continue

                idx = sentence.lower().find(phrase.lower())
                if idx == -1:
                    continue

                new_obj = {
                    "sentence": sentence,
                    "canonical_phrase": phrase,
                    "char_offset_start": idx,
                    "char_offset_end": idx + len(phrase),
                    "timestamp": obj.get("timestamp", ""),
                    "primary_category": obj.get("category", "")
                }

                fout.write(json.dumps(new_obj) + "\n")
                kept += 1

            if total % 100000 == 0:
                print(f"processed {total} lines, kept {kept}")

    print(f"\nDONE")
    print(f"Total lines: {total}")
    print(f"Kept lines: {kept}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    normalize(Path(args.input), Path(args.output))
