import os
import re
import time
import json
import pickle
import argparse
import openai
import random
import ast
from collections import defaultdict
from tools.chat_openai import generate_messages

prompt_template = """These are descriptions of a video that I want to upload, please answer the question. You need to answer the question in any case and not demand additional context information. Note: All actions mentioned refer to the person recording the video.

Video Description:
{relevant_captions}

Question: 
{question}

If the provided description is insufficient to answer the question, output 'Insufficient Information'.
Answer:"""

def create_prompt(q, a, pred):
    return f"""role: "system",
content: "You are an intelligent chatbot designed for evaluating the correctness of AI assistant predictions for question-answer pairs.
Your task is to compare the predicted answer with the ground-truth answer and determine if the predicted answer is correct or not. Here's how you can accomplish the task:
-----##INSTRUCTIONS:
- Focus on the correctness and accuracy of the predicted answer with the ground-truth.
- Consider uncertain predictions, such as 'it is impossible to answer the question from the video', as incorrect, unless the ground truth answer also says that."
role: "user",
content: "Please evaluate the following video-based question-answer pair:
Question: {q}
Ground truth correct Answer: {a}
Predicted Answer: {pred}
Provide your evaluation as a correct/incorrect prediction along with the score where the score is an integer value between 0 (fully wrong) and 5 (fully correct). The middle score provides the percentage of correctness.
Please generate the response in the form of a Python dictionary string with keys 'pred', 'score' and 'reason', where value of 'pred' is a string of 'correct' or 'incorrect',
value of 'score' is in INTEGER, not STRING and value of 'reason' should provide the reason behind the decision."
"""

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

def get_response(client_key, messages, max_tokens):
    selected_model = client[client_key]
    if isinstance(selected_model, list):
        selected_model = random.choice(selected_model)

    response = selected_model.chat.completions.create(
        model=client_key,
        messages=messages,
        temperature=1e-6,
        timeout=120,
        max_tokens=max_tokens,
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
    if os.path.exists(memory_path):
        memory = pickle.load(open(memory_path, "rb"))
        memory.interval = 10
        task["memory"] = to_string(memory, memory_type=args.memory_type)
    else:
        memory_path = os.path.join(args.memory_folder, f"{task['videoID']}.txt")
        task["memory"] = open(memory_path).read().strip()

    prompt = prompt_template.format(
        relevant_captions=task["memory"],
        question=task["question"]
    )
    message = generate_messages([{"type": "text", "text": prompt}])
    for _ in range(5):
        try:
            answer = get_response("gpt-4o-2024-11-20", message, 16384)
            break
        except:
            time.sleep(1)
            answer = "None"
    task["model_answer"] = answer
    print(answer)

    if "Insufficient Information" in answer:
        answer = {
            "pred": "",
            "score": -1,
            "reason": "",
        }
    else:
        prompt = create_prompt(task["question"], task["answer"], task["model_answer"])
        message = generate_messages([{"type": "text", "text": prompt}])
        for _ in range(5):
            try:
                answer = get_response("gemini-1.5-pro-002", message, 4096)
                match = re.search(r'\{.*?\}', answer, re.DOTALL)
                if match:
                    eval_dict = ast.literal_eval(match.group(0))
                    answer = {
                        "pred": eval_dict.get("pred", ""),
                        "score": int(eval_dict.get("score", 0)),
                        "reason": eval_dict.get("reason", "")
                    }
                    break
                else:
                    assert True == False
            except Exception as e:
                time.sleep(1)
                answer = {
                    "pred": "",
                    "score": 0,
                    "reason": e,
                }
    
    task["eval"] = answer["score"]
    task["eval_result"] = answer
    
    return task

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--memory_name", type=str, default="baseline_ep",
                        help="Tag used to look up memory files (e.g. baseline_ep).")
    parser.add_argument("--memory_type", type=str, default="episodic",
                        help="One of: episodic, semantic, both.")
    parser.add_argument("--task_type", type=str, default="all",
                        help="EgoTempo task category, or 'all'.")
    parser.add_argument("--memory_folder", type=str, required=True,
                        help="Folder containing per-video memory dumps.")
    parser.add_argument("--video_info", type=str, required=True,
                        help="Path to EgoTempo video_info.json (the question pool).")
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
                               f"{args.memory_name}_{args.memory_type}_{args.task_type}_{args.times}.jsonl")

    test_data = defaultdict(list)
    data = json.load(open(args.video_info))
    for i in data[args.task_type][args.start_index:]:
        test_data[i["videoID"]].append(i)
    with open(output_path, "w") as f:
        for k, v in test_data.items():
            for task in v:
                task = answer_qa(task, args)
                f.write(json.dumps(task) + '\n')
            time.sleep(1)

