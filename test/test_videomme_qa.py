import os
import time
import json
import pickle
import argparse
import openai
import random
from collections import defaultdict
from tools.chat_openai import generate_messages

prompt_template = """Based on the following video description, select one option as the answer to the question. Give your reasoning for your answer. And output the option letter A, B, C or D. If you cannot find the answer, output E.

Video Description:
{relevant_captions}

Question: 
{question}

Reasoning:
[Your reasoning here]

Answer:[A|B|C|D|E]"""

config = json.load(open(os.environ.get("TASKMEM_API_CONFIG", "configs/api_config.json")))
client = {}
for model_name in config.keys():
    if isinstance(config[model_name], list):
        client[model_name] = [openai.AzureOpenAI(
            azure_endpoint=conf["azure_endpoint"],
            api_version=conf["api_version"],
            api_key=conf["api_key"],
        ) for conf in config[model_name]]
    else:
        client[model_name] = openai.AzureOpenAI(
            azure_endpoint=config[model_name]["azure_endpoint"],
            api_version=config[model_name]["api_version"],
            api_key=config[model_name]["api_key"],
        )

def get_response(client_key, messages):
    selected_model = client[client_key]
    if isinstance(selected_model, list):
        selected_model = random.choice(selected_model)

    response = selected_model.chat.completions.create(
        model=client_key,
        messages=messages,
        temperature=1e-6,
        timeout=120,
        max_tokens=16384,
    )
    return response.choices[0].message.content

def to_string(self, memory_type="both", clip_ids=None):
    if clip_ids is None:
        if memory_type == "episodic":
            clip_ids = [str(i) for i in range(len(self.episodic.description))]
        else:
            clip_ids = [str(i) for i in range(len(self.semantic.knowledge))]
    des, kno = [], []
    for clip_id in clip_ids:
        if memory_type == "episodic" or memory_type == "both":
            des.append("{}-{}\n{}".format(self.format_second(int(clip_id) * self.interval), self.format_second((int(clip_id) + 1) * self.interval), self.episodic.description[clip_id]))
        if memory_type == "semantic" or memory_type == "both":
            kno.extend(self.semantic.knowledge[clip_id])
    try:
        des = "\n\n".join(des)
    except:
        des = str(des)
    try:
        kno = "\n".join(kno)
    except:
        kno = str(kno)
    if memory_type == "episodic":
        return "Description:\n\n{}".format(des)
    elif memory_type == "semantic":
        return "Knowledge:\n{}".format(kno)
    else:
        return "Description:\n\n{}\n\nKnowledge:\n{}".format(des, kno)

def answer_qa(task, args):
    if args.test_time:
        memory_path = os.path.join(args.memory_folder, f"{task['videoID']}/test_time/{task['videoID']}_{args.memory_name}.pkl")
    else:
        memory_path = os.path.join(args.memory_folder, f"{task['videoID']}/{task['videoID']}_{args.memory_name}.pkl")
    memory = pickle.load(open(memory_path, "rb"))
    memory.interval = 10
    # task["memory"] = memory.to_string(memory_type=memory_type)
    task["memory"] = to_string(memory, memory_type=args.memory_type)
    prompt = prompt_template.format(
        relevant_captions=task["memory"],
        question="\n".join([task["question"]] + task["options"])
    )
    message = generate_messages([{"type": "text", "text": prompt}])
    for _ in range(5):
        try:
            answer = get_response("gpt-4o-2024-11-20", message)
            break
        except:
            time.sleep(1)
            answer = "Answer: E"
    task["model_answer"] = answer
    return task

def check(task):
    return 1 if task["answer"] == task["model_answer"][0] else 0

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_type", type=str, default="short",
                        help="One of: short, medium, long.")
    parser.add_argument("--memory_name", type=str, default="baseline_ep",
                        help="Tag used to look up memory files (e.g. baseline_ep).")
    parser.add_argument("--memory_type", type=str, default="episodic",
                        help="One of: episodic, semantic, both.")
    parser.add_argument("--task_type", type=str, default="all",
                        help="VideoMME task category, or 'all'.")
    parser.add_argument("--memory_folder", type=str, required=True,
                        help="Folder containing per-video memory dumps.")
    parser.add_argument("--video_info_root", type=str, required=True,
                        help="Directory holding VideoMME video_info_{short|medium|long}.json files.")
    parser.add_argument("--times", type=str, default="1",
                        help="Run index suffix used for the output filename.")
    parser.add_argument("--start_index", type=int, default=0)
    parser.add_argument("--test_time", default=False, action="store_true")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    output_folder = os.path.join(args.memory_folder, "results")
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder,
                               f"{args.memory_name}_{args.memory_type}_{args.task_type}_{args.video_type}_{args.times}.jsonl")

    test_data = defaultdict(list)
    data = json.load(open(os.path.join(args.video_info_root, f"video_info_{args.video_type}.json")))
    for i in data[args.task_type][args.start_index:]:
        test_data[i["videoID"]].append(i)
    with open(output_path, "w") as f:
        for k, v in test_data.items():
            for task in v:
                task = answer_qa(task, args)
                task["eval"] = check(task)
                f.write(json.dumps(task) + '\n')
            time.sleep(1)
