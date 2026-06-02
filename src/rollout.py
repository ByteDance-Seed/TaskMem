import os
import json
import argparse
from typing import Any, List

from tools.chat_qwen_vl_vllm import ChatVL  

MAX_NEW_TOKENS = 8192

def _read_json(path: str):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def run_rollout(
    chat: ChatVL,
    model_path: str,
    task: str,
    inputs: List[Any],
    rollout_count: int,
):
    print(f"\n===== Model: {model_path} | task={task} | batch_size={len(inputs)} | rollout_count={rollout_count} =====")

    for inp in inputs:
        msg = chat.generate_messages(inp[1][task]["input"])
        outputs = chat.llm.generate([msg], sampling_params=chat.sampling_params)[0]
        inp[1][task]["rollout"] = []
        for rollout in range(rollout_count):
            text = outputs.outputs[rollout].text if outputs.outputs else ""
            inp[1][task]["rollout"].append(text)
        json.dump(inp[1], open(inp[0], "w"), ensure_ascii=False, indent=4)

def main(video_folder, model_path, read_memory_tag, write_memory_tag, start_time, end_time, task, rollout_count):
    datas = []
    video_id = os.path.basename(video_folder)
    for t in range(start_time, end_time):
        t = str(t // 10)
        datas.append([
            os.path.join(video_folder, t, f"{video_id}_{t}_{write_memory_tag}.json"),
            json.load(open(os.path.join(video_folder, t, f"{video_id}_{t}_{read_memory_tag}.json")))
        ])

    chat = ChatVL(model_path)
    chat.sampling_params.max_tokens = MAX_NEW_TOKENS
    chat.sampling_params.n = rollout_count 
    
    run_rollout(chat, model_path, task, datas, rollout_count)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=["episodic", "semantic"],
        help="semantic or episodic",
    )
    parser.add_argument(
        "--video_folder",
        type=str,
        required=True,
        help="json path of input",
    )
    parser.add_argument(
        "--read_memory_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--write_memory_tag",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--start_time",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--end_time",
        type=int,
        required=True,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="model path for inference",
    )
    parser.add_argument(
        "--rollout_count",
        type=int,
        required=True,
        help="count of rollout",
    )

    args = parser.parse_args()

    main(
        video_folder=args.video_folder,
        model_path=args.model_path,
        read_memory_tag=args.read_memory_tag,
        write_memory_tag=args.write_memory_tag,
        start_time=args.start_time,
        end_time=args.end_time,
        task=args.task,
        rollout_count=args.rollout_count,
    )
