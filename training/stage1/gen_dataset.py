import argparse
import json
import os

from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stage-1 episodic data packer: read per-clip "
                    "{id}_{idx}_{memory_tag}.json memory dumps and emit "
                    "an SFT-ready jsonl shard with chat-templated text."
    )
    parser.add_argument("--video_info", type=str, required=True,
                        help="jsonl of video metadata records "
                             "(id / start_time / end_time / memory_path / folder)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="destination jsonl path for this shard")
    parser.add_argument("--memory_tag", type=str, default="baseline_ep",
                        help="suffix used in {id}_{idx}_{memory_tag}.json")
    parser.add_argument("--clip_seconds", type=int, default=10,
                        help="clip length in seconds")
    parser.add_argument("--processor_path", type=str,
                        default="Qwen/Qwen3-VL-30B-A3B-Instruct",
                        help="HF hub id or local path of the VL processor")
    parser.add_argument("--total_shards", type=int, default=1,
                        help="total number of parallel shards being produced")
    parser.add_argument("--shard_id", type=int, default=0,
                        help="zero-based index of this shard "
                             "(record i is kept iff i %% total_shards == shard_id)")
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    processor = AutoProcessor.from_pretrained(args.processor_path)

    with open(args.video_info) as f_in, open(args.output_path, "w") as f_out:
        for i, line in enumerate(f_in):
            if i % args.total_shards != args.shard_id:
                continue
            data = json.loads(line)
            for t in range(data["start_time"], data["end_time"], args.clip_seconds):
                clip_idx = str(t // args.clip_seconds)
                mem_file = os.path.join(
                    data["memory_path"],
                    clip_idx,
                    f"{data['id']}_{clip_idx}_{args.memory_tag}.json",
                )
                memory = json.load(open(mem_file))
                if "episodic" not in memory:
                    continue
                messages = [{"role": "user", "content": memory["episodic"]["input"]}]
                text = processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                _, _, video_kwargs = process_vision_info(
                    messages,
                    image_patch_size=processor.image_processor.patch_size,
                    return_video_kwargs=True,
                    return_video_metadata=True,
                )
                video_path = os.path.join(
                    data["memory_path"], clip_idx, f"{data['id']}_{clip_idx}.mp4"
                )
                res = {
                    "id": f"{data.get('folder', '')}_{data['id']}",
                    "type": "episodic",
                    "input": memory["episodic"]["input"],
                    "output": memory["episodic"]["output"],
                    "text": text,
                    "images": None,
                    "videos": video_path,
                    "video_kwargs": video_kwargs,
                    "episodic_video_path": os.path.join(
                        data["memory_path"],
                        clip_idx,
                        f"{data['id']}_{clip_idx}_new.mp4",
                    ),
                }
                f_out.write(json.dumps(res, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
