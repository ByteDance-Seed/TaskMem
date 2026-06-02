import argparse
import json
import os
import random
from typing import List

from tqdm import tqdm


MEM_TYPE = "episodic"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 episodic SFT data assembler: walk every "
                    "video listed in --video_info_dir/train_{split}.jsonl, "
                    "collect all per-clip episodic memory records, shuffle "
                    "within each video, and interleave them across videos "
                    "for round-robin training order."
    )
    parser.add_argument("--video_info_dir", type=str, required=True,
                        help="directory containing train_{split}.jsonl files; "
                             "each line needs fields id / start_time / end_time")
    parser.add_argument("--splits", nargs="+", type=int, required=True,
                        help="which train_{split}.jsonl files to scan")
    parser.add_argument("--memory_root", type=str, required=True,
                        help="root containing {id}/{clip_idx}/{id}_{clip_idx}_{memory_tag}.json")
    parser.add_argument("--memory_tag", type=str, required=True,
                        help="tag suffix used in per-clip memory filenames")
    parser.add_argument("--output_path", type=str, required=True,
                        help="destination .jsonl path")
    parser.add_argument("--clip_seconds", type=int, default=10,
                        help="clip length in seconds")
    parser.add_argument("--num_passes", type=int, default=30,
                        help="how many interleaved passes to write "
                             "(each pass emits one record per video)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    per_video: List[List[dict]] = []
    out_lens, item_lens = [], []

    for split in args.splits:
        info_path = os.path.join(args.video_info_dir, f"train_{split}.jsonl")
        if not os.path.exists(info_path):
            continue
        with open(info_path) as f:
            for line in tqdm(f, desc=f"split={split}"):
                data = json.loads(line)
                clips: List[dict] = []
                for t in range(data["start_time"], data["end_time"], args.clip_seconds):
                    clip_idx = int(t // args.clip_seconds)
                    mem_file = os.path.join(
                        args.memory_root,
                        data["id"],
                        str(clip_idx),
                        f"{data['id']}_{clip_idx}_{args.memory_tag}.json",
                    )
                    memory_io = json.load(open(mem_file))
                    clips.append({
                        "id": f"{data['id']}*{clip_idx}",
                        "type": MEM_TYPE,
                        "input": memory_io[MEM_TYPE]["input"],
                        "output": memory_io[MEM_TYPE]["output"],
                    })
                    out_lens.append(len(memory_io[MEM_TYPE]["output"]))
                    for j in memory_io[MEM_TYPE]["output"]:
                        item_lens.append(len(j))
                random.shuffle(clips)
                per_video.append(clips)

    if out_lens:
        print(f"avg outputs per clip: {sum(out_lens) / len(out_lens):.3f}")
    if item_lens:
        print(f"avg chars per output item: {sum(item_lens) / len(item_lens):.3f}")

    with open(args.output_path, "w") as f:
        for pass_idx in range(args.num_passes):
            for clips in per_video:
                if pass_idx < len(clips):
                    f.write(json.dumps(clips[pass_idx], ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
