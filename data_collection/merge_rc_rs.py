import os
import glob

INPUT_DIR = "./"   # change if needed
OUTPUT_DIR = "./merged"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def merge_files(rc_path, rs_path, output_path):
    with open(output_path, "w", encoding="utf-8") as out:

        # write comments first
        if os.path.exists(rc_path):
            print(f"Adding RC: {rc_path}")
            with open(rc_path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

        # then submissions
        if os.path.exists(rs_path):
            print(f"Adding RS: {rs_path}")
            with open(rs_path, "r", encoding="utf-8") as f:
                for line in f:
                    out.write(line)

    print(f"Saved → {output_path}\n")


def main():
    # find all RC files
    rc_files = glob.glob(os.path.join(INPUT_DIR, "RC_*.jsonl"))

    for rc_file in rc_files:
        filename = os.path.basename(rc_file)
        date_part = filename.replace("RC_", "").replace(".jsonl", "")

        rs_file = os.path.join(INPUT_DIR, f"RS_{date_part}.jsonl")
        output_file = os.path.join(OUTPUT_DIR, f"merged_{date_part}.jsonl")

        merge_files(rc_file, rs_file, output_file)


if __name__ == "__main__":
    main()
