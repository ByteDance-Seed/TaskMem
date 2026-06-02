import argparse
import glob
import json
import os
import re

from tqdm import tqdm


def invalid_face_tag(desc: str) -> bool:
    if re.search(r"<face>", desc, re.IGNORECASE):
        return True
    suspicious_pattern = re.compile(r"face[\s_]*\d+", re.IGNORECASE)
    for m in suspicious_pattern.finditer(desc):
        s, e = m.span()
        token = m.group()
        if token != f"face_{token.split('_')[-1]}" or not re.fullmatch(r"face_\d+", token):
            if not re.fullmatch(r"face_\d+", token):
                return True
        has_brackets = (
            s > 0 and desc[s - 1] == "<" and e < len(desc) and desc[e] == ">"
        )
        if not has_brackets:
            return True
    return False


def parse_args():
    parser = argparse.ArgumentParser(
        description="Filter pre-encoded episodic training records, dropping "
                    "any record whose `text` field contains a malformed "
                    "face tag (e.g. bare `<face>` or `face_3` not wrapped "
                    "in angle brackets)."
    )
    parser.add_argument("--input_glob", type=str, required=True,
                        help="glob pattern matching the .jsonl shards to scan")
    parser.add_argument("--output_path", type=str, required=True,
                        help="destination .jsonl for the filtered episodic records")
    parser.add_argument("--memory_type", type=str, default="episodic",
                        choices=["episodic"],
                        help="record type to keep (only episodic is open-sourced)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    kept, dropped = 0, 0
    with open(args.output_path, "w") as fout:
        for file_path in tqdm(sorted(glob.glob(args.input_glob))):
            with open(file_path) as f:
                for line in f:
                    data = json.loads(line)
                    if data.get("type") != args.memory_type:
                        continue
                    if invalid_face_tag(data["text"]):
                        dropped += 1
                        continue
                    fout.write(line)
                    kept += 1

    print(f"kept={kept} dropped={dropped}")


if __name__ == "__main__":
    main()
