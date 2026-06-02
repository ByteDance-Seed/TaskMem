import argparse
import copy
import json
import os
import random

from memory.memory_process import construct_episodic_input


def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate on-policy episodic training trajectories. "
                    "Maintains a sliding batch of `batch_size` videos; each "
                    "step advances the clip pointer by one, occasionally "
                    "swaps an entry out for a fresh video using a depth-"
                    "dependent drop probability. This re-creates the "
                    "distribution of (step within video, video index) pairs "
                    "the policy will see during streaming inference."
    )
    parser.add_argument("--src_jsonl", type=str, required=True,
                        help="input jsonl of seed episodic records "
                             "(must contain id / start_time / memory_path)")
    parser.add_argument("--output_path", type=str, required=True,
                        help="destination jsonl for simulated trajectories")
    parser.add_argument("--episodic_folder", type=str, required=True,
                        help="root folder injected into each output record "
                             "as `episodic_folder` for downstream loaders")
    parser.add_argument("--total_step", type=int, default=50,
                        help="number of streaming steps to simulate")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="number of videos kept in the rolling batch")
    parser.add_argument("--soft_swap_after", type=int, default=11,
                        help="step depth at which each entry becomes eligible "
                             "for the probabilistic swap")
    parser.add_argument("--hard_swap_after", type=int, default=22,
                        help="step depth at which each entry is force-swapped")
    parser.add_argument("--soft_swap_prob", type=float, default=0.13,
                        help="per-step swap probability between soft and hard swap")
    parser.add_argument("--clip_seconds", type=int, default=10,
                        help="clip length in seconds (used to derive video_idx)")
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)) or ".", exist_ok=True)

    datas = []
    with open(args.src_jsonl) as f:
        for line in f:
            datas.append(json.loads(line))
    assert len(datas) >= args.batch_size, (
        f"need at least batch_size={args.batch_size} seed records, "
        f"got {len(datas)}"
    )

    p_data = args.batch_size
    data_steps = [[] for _ in range(args.total_step)]
    for i in range(args.batch_size):
        data_steps[0].append(copy.deepcopy(datas[i]))
        data_steps[0][-1]["step"] = 0

    for step in range(1, args.total_step):
        data_steps[step] = copy.deepcopy(data_steps[step - 1])
        for i in range(len(data_steps[step])):
            data_steps[step][i]["step"] += 1
            if data_steps[step][i]["step"] > args.soft_swap_after:
                drop = random.random() <= args.soft_swap_prob
                if data_steps[step][i]["step"] > args.hard_swap_after:
                    drop = True
                if drop:
                    if p_data >= len(datas):
                        continue
                    data_steps[step][i] = copy.deepcopy(datas[p_data])
                    p_data += 1
                    data_steps[step][i]["step"] = 0

    print(f"consumed {p_data} seed records")

    with open(args.output_path, "w") as f_out:
        for data in data_steps:
            for i in data:
                i["video_idx"] = i["start_time"] // args.clip_seconds + i["step"]
                i["type"] = "episodic_on_policy"
                i["input"] = construct_episodic_input(
                    {
                        "clip": f"{i['memory_path']}/{i['video_idx']}/{i['id']}_{i['video_idx']}.mp4",
                        "episodic": ["{}"],
                    },
                    "",
                )
                i["episodic_folder"] = args.episodic_folder
                assert os.path.exists(i["input"][1]["video"]), i["input"][1]["video"]
                i["id"] = f"{i['id']}*{i['video_idx']}"
                f_out.write(json.dumps(i) + "\n")


if __name__ == "__main__":
    main()
