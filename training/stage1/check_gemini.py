import argparse
import os

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Walk a folder of Gemini rollout log files and print "
                    "the path of every .txt that contains a Python "
                    "traceback (i.e. a failed run worth re-running)."
    )
    parser.add_argument("--target_folder", type=str, required=True,
                        help="root folder to scan recursively")
    parser.add_argument("--suffix", type=str, default=".txt",
                        help="file suffix to inspect")
    parser.add_argument("--needle", type=str, default="Traceback ",
                        help="substring that marks a failed run")
    return parser.parse_args()


def main():
    args = parse_args()
    for root, _, files in tqdm(os.walk(args.target_folder)):
        for file_name in files:
            if not file_name.endswith(args.suffix):
                continue
            file_full_path = os.path.join(root, file_name)
            with open(file_full_path) as f:
                for line in f:
                    if args.needle in line:
                        print(file_full_path)
                        break


if __name__ == "__main__":
    main()
