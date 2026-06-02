"""EgoLife QA evaluation driven by per-model memory pickles.

Directory convention: ``memory_root / {stem} / {stem}_{memory_tag}.pkl``

- ``stem`` is the clip folder name (e.g.
  ``current_123_DAY2_A1_JAKE_16190000__DAY2_A1_JAKE_16200000``).
- ``memory_tag`` is the model-specific suffix (e.g. ``baseline_ep``); a single
  run only evaluates one model.

For each QA item we resolve the ``current_groups`` / ``target_groups``
outputs to clip stems, load the matching pickle, materialise its text via
``LongMemory.to_string()`` and feed the concatenated text to the LLM judge.
The latest prompt is mirrored to a temp file for inspection.

Example:
    python test/test_egolife_qa.py \\
        --qa_file ./data/egolife_qa.json \\
        --memory_root ./out/egolife \\
        --memory_tag baseline_ep \\
        --output_file ./out/results/baseline_ep.jsonl
"""
import json
import os
import argparse
from pathlib import Path
from tqdm import tqdm
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
import pickle
import warnings
from tools.chat_openai import generate_messages
import openai
import random
import time

MODEL = "gpt-4o-2024-11-20"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

MEMORY_CACHE = {}

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

# Persist the current memory text to disk for offline inspection.
def _memory_temp_path():
    return Path(__file__).resolve().parent.parent / "memory_text_temp.txt"

def clean_unicode_surrogates(text):
    if isinstance(text, str):
        return text.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
    elif isinstance(text, dict):
        return {k: clean_unicode_surrogates(v) for k, v in text.items()}
    elif isinstance(text, list):
        return [clean_unicode_surrogates(item) for item in text]
    return text

def parse_model_json(text: str):
    s = str(text).strip().strip("`json").strip()
    try:
        obj = json.loads(s)
        cot, ans = obj.get("cot", ""), obj.get("answer", "").upper()
        if ans in {"A", "B", "C", "D", "E"}:
            return cot, ans
    except Exception:
        pass
    return s, "NULL"

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

def load_single_memory(mem_path: Path, memory_type, interval: int = 10):
    """Load memory from a pickle, set ``memory.interval`` and return the
    serialized text via ``to_string()``."""
    key = str(mem_path)
    if key in MEMORY_CACHE:
        return MEMORY_CACHE[key]
    if not mem_path.exists():
        return None
    with open(mem_path, "rb") as f:
        memory = pickle.load(f)

    memory.interval = interval
    memory_text = to_string(memory, memory_type=memory_type)
    MEMORY_CACHE[key] = memory_text
    return memory_text

def _format_ts_to_hhmmssmmm(ts: str) -> str:
    s = re.sub(r'\D+', '', str(ts))
    if not s:
        return "00:00:00:000"
    if len(s) == 8:
        hh, mm, ss, ms = int(s[0:2]), int(s[2:4]), int(s[4:6]), int(s[6:8])
        return f"{hh:02d}:{mm:02d}:{ss:02d}:{ms:03d}"
    if len(s) == 6:
        hh, mm, ss = int(s[0:2]), int(s[2:4]), int(s[4:6])
        return f"{hh:02d}:{mm:02d}:{ss:02d}:000"
    return "00:00:00:000"

def load_memories_by_model(groups, memory_root: Path, args: str):
    """Load memories for a single model from
    ``memory_root / stem / {stem}_{memory_tag}.pkl``.

    Each ``group`` provides an ``output`` field (path or stem) from which the
    stem and therefore the pickle path are derived.
    """
    results = []
    for group in groups:
        output = group.get("output")
        if not output:
            continue
        stem = Path(output).stem
        time_match = re.search(r"(\d{8})__.*?(\d{8})", stem)
        if not time_match:
            continue
        start_time = time_match.group(1)
        end_time = time_match.group(2)
        formatted_start = _format_ts_to_hhmmssmmm(start_time)
        formatted_end = _format_ts_to_hhmmssmmm(end_time)
        parts = stem.split("_")
        day = parts[2] if len(parts) > 2 else "DAY?"
        new_output = f"{day}: from {formatted_start} to {formatted_end}"
        mem_path = memory_root / stem / f"{stem}_{args.memory_tag}.pkl"
        mem_text = load_single_memory(mem_path, args.memory_type)
        if mem_text:
            results.append({"output": new_output, "memory_text": mem_text})
    return results

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

def call_model_once(prompt):
    inputs = [{"type": "text", "text": prompt}]
    messages = generate_messages(inputs)
    response = get_response(MODEL, messages)
    for _ in range(5):
        try:
            response = get_response(MODEL, messages)
            break
        except:
            time.sleep(1)
            response = ""
    if isinstance(response, tuple) and len(response) == 2:
        response = response[0]
    cot, ans = parse_model_json(response)
    return response, cot, ans

def process_item(item, args, memory_root: Path):
    try:
        qa = item.get("qa", item)
        assert qa.get("query_time") and qa["query_time"].get("time") and qa.get("target_time")
        current_groups = item.get("current_groups", [])
        target_groups = item.get("target_groups", [])

        item_id = item.get("ID") or item.get("id") or "UNKNOWN"
        current_mems = load_memories_by_model(
            current_groups, memory_root, args
        )
        target_mems = load_memories_by_model(
            target_groups, memory_root, args
        )
        assert current_mems and target_mems

        cur_memory_text = "=== CURRENT clip memories ===\n" + (
            f"{current_mems[0]['output']}\n{current_mems[0]['memory_text']}"
        )
        tar_memory_text = "=== TARGET clip memories ===\n" + "\n\n".join(
            [f"{m['output']}\n{m['memory_text']}" for m in target_mems]
        )
        memory_text = cur_memory_text + "\n\n" + tar_memory_text
        cleaned_memory_text = clean_unicode_surrogates(memory_text)
        # Mirror the latest prompt memory to a temp file for offline inspection.
        try:
            temp_path = _memory_temp_path()
            with open(temp_path, "w", encoding="utf-8", errors="replace") as f:
                f.write(cleaned_memory_text)
        except Exception as e:
            pass

        question = item["qa_raw"]["question"]
        options = (
            f"A. {item['qa_raw']['choice_a']}\n"
            f"B. {item['qa_raw']['choice_b']}\n"
            f"C. {item['qa_raw']['choice_c']}\n"
            f"D. {item['qa_raw']['choice_d']}"
        )
        prompt = f"""Based on the following video description, select one option as the answer to the question. The question is asked at the CURRENT time, but the relevant evidence is usually located in the TARGET clips. Only output the option letter A, B, C, D or E. If you cannot find the answer from the description, use E.

Description:
{cleaned_memory_text}

Question: {question}

Options:
{options}

Respond ONLY in strict JSON (no markdown, no code fences, no extra text).
The JSON schema is:
{{
  "cot": "Reasoning for the selected answer in English or Chinese",
  "answer": "A|B|C|D|E"
}}
"""
        num_samples = max(1, int(args.num_samples))
        votes = {"A": 0, "B": 0, "C": 0, "D": 0, "E": 0}
        sample_responses = []

        with ProcessPoolExecutor(max_workers=num_samples) as executor:
            futures = [executor.submit(call_model_once, prompt) for _ in range(num_samples)]
            for idx, fut in enumerate(as_completed(futures)):
                try:
                    raw_response, cot, ans = fut.result()
                    if ans in votes:
                        votes[ans] += 1
                    sample_responses.append({
                        "sample": idx + 1, "answer": ans,
                        "raw_response": raw_response, "cot": cot,
                    })
                except Exception as e:
                    sample_responses.append({
                        "sample": idx + 1, "answer": "NULL",
                        "raw_response": None, "cot": "", "error": str(e),
                    })

        final_answer = max(votes, key=votes.get) if votes and sum(votes.values()) > 0 else "NULL"
        mem_path_label = current_mems[0].get("output") if current_mems else None
        return {
            **item,
            "model_answer": final_answer,
            "model_answer_votes": votes,
            "sample_responses": sample_responses,
            "model_cot": "\n\n".join(
                [f"Sample #{s.get('sample', i+1)}:\n{s.get('cot', '')}" for i, s in enumerate(sample_responses)]
            ),
            "answer": qa.get("answer"),
            "memory": memory_text,
            "memory_extracted": True,
            "memory_path": mem_path_label,
            "memory_tag": args.memory_tag,
        }
    except Exception as e:
        item_id = item.get("ID") or item.get("id") or "UNKNOWN"
        return {
            **item,
            "model_answer": f"ERROR: {e}",
            "error": str(e),
            "memory_extracted": False,
            "memory_tag": getattr(args, "memory_tag", ""),
        }

def calculate_type_accuracy(results):
    stats = {}
    for r in results:
        t = r.get("qa_raw", {}).get("type") or r.get("qa", {}).get("type", "Unknown")
        if t not in stats:
            stats[t] = {"correct": 0, "total": 0}
        stats[t]["total"] += 1
        if r.get("model_answer") == r.get("answer"):
            stats[t]["correct"] += 1
    return {t: (v["correct"] / v["total"] * 100) for t, v in stats.items() if v["total"] > 0}

def main():
    parser = argparse.ArgumentParser(
        description="EgoLife QA: load memory pickles from "
                    "memory_root/{stem}/{stem}_{memory_tag}.pkl, build prompts via "
                    "to_string(), and run the LLM judge.")
    parser.add_argument("--qa_file", type=str, required=True,
                        help="Path to the EgoLife QA JSON file.")
    parser.add_argument("--memory_root", type=str, required=True,
                        help="Root directory holding per-clip memory subfolders.")
    parser.add_argument(
        "--memory_tag",
        type=str,
        default="baseline_ep",
        help="Memory file suffix; pickles are named {stem}_{memory_tag}.pkl.",
    )
    parser.add_argument("--num_samples", type=int, default=3)
    parser.add_argument("--output_file", type=str, default="egolife_qa_by_model.jsonl")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--memory_type", type=str, default="episodic",
                        help="One of: episodic, semantic, both.")
    parser.add_argument("--task_type", type=str, default="all")
    args = parser.parse_args()

    memory_root = Path(args.memory_root)
    with open(args.qa_file, "r", encoding="utf-8") as f:
        qa_data = []
        for data in json.load(f)["results"]:
            if args.task_type != "all" and args.task_type != data.get("qa_raw", {}).get("type", ""):
                continue
            qa_data.append(data)

    results = []
    skipped = []
    output_file = Path(args.output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    def flush_partial():
        with open(args.output_file, "w", encoding="utf-8", errors="replace") as fw:
            for r in results:
                cleaned_r = clean_unicode_surrogates(r)
                try:
                    fw.write(json.dumps(cleaned_r, ensure_ascii=False) + "\n")
                except (UnicodeEncodeError, UnicodeDecodeError):
                    fw.write(json.dumps(cleaned_r, ensure_ascii=True) + "\n")

    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        futures = {
            executor.submit(process_item, item, args, memory_root): item
            for item in qa_data
        }
        for i, fut in enumerate(tqdm(as_completed(futures), total=len(futures))):
            try:
                rec = fut.result()
                results.append(rec)
                if "SKIPPED" in rec.get("model_answer", ""):
                    skipped.append(rec.get("ID", "UNKNOWN"))
                done = i + 1
                if done % 50 == 0:
                    flush_partial()
                    correct_so_far = sum(1 for r in results if r.get("model_answer") == r.get("answer") and r.get("answer"))
            except Exception as e:
                item = futures.get(fut, {})
                failed_id = item.get("ID") or item.get("id") or "UNKNOWN"
                results.append({**item, "model_answer": f"ERROR: {e}", "error": str(e)})
                if (i + 1) % 50 == 0:
                    flush_partial()

    failed_ids = [
        r.get("ID") or r.get("id") or "UNKNOWN"
        for r in results
        if "ERROR" in str(r.get("model_answer", "")) or "SKIPPED" in str(r.get("model_answer", ""))
    ]
    type_acc = calculate_type_accuracy(results)
    total = len(results)
    valid_results = [
        r for r in results
        if "ERROR" not in str(r.get("model_answer", ""))
        and "SKIPPED" not in str(r.get("model_answer", ""))
        and r.get("answer")
    ]
    valid_count = len(valid_results)
    correct = sum(1 for r in valid_results if r.get("model_answer") == r.get("answer"))
    memory_extracted_count = sum(1 for r in results if r.get("memory_extracted", False))

    total_accuracy = (correct / total * 100) if total else 0.0
    valid_accuracy = (correct / valid_count * 100) if valid_count else 0.0

    flush_partial()
    print("\n" + "=" * 60)
    print("Final Statistics (by-model, no index)")
    print("=" * 60)
    print(f"memory_root: {args.memory_root}")
    print(f"memory_tag:  {args.memory_tag}")
    print(f"Total tasks: {total}")
    print(f"Valid (with answer): {valid_count}")
    print(f"Correct: {correct}")
    print(f"Error count: {len(failed_ids)}")
    print(f"Memory extracted: {memory_extracted_count} ({memory_extracted_count/total*100:.1f}%)" if total else "Memory extracted: 0")
    print(f"Total accuracy: {total_accuracy:.2f}%")
    print(f"Valid accuracy: {valid_accuracy:.2f}%")
    if failed_ids:
        print(f"Failed IDs (first 20): {failed_ids[:20]}")
        if len(failed_ids) > 20:
            print(f"  ... and {len(failed_ids) - 20} more")
    print("=" * 60)

if __name__ == "__main__":
    main()
