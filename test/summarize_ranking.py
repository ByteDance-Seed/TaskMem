import argparse
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean

USEFULNESS_METRIC_DEFAULT = "avg_item_usefulness_raw"

def _safe_int(x):
    try:
        return int(x)
    except Exception:
        return None

def _step_sort_key(step: str):
    si = _safe_int(step)
    return (0, si) if si is not None else (1, str(step))


def iter_json_files(dirs, pattern="*_eval.json"):
    for d in dirs:
        d = Path(d)
        for p in d.rglob(pattern):
            if p.is_file():
                yield p


def load_json(p: Path):
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def make_competition_rank(step_to_val, reverse=True):
    items = [(str(s), float(v)) for s, v in step_to_val.items()]
    if reverse:
        items.sort(key=lambda x: (-x[1], _step_sort_key(x[0])))
    else:
        items.sort(key=lambda x: (x[1], _step_sort_key(x[0])))

    ranks = {}
    prev_val = None
    prev_rank = 0
    for i, (s, v) in enumerate(items, start=1):
        if prev_val is None or v != prev_val:
            prev_val = v
            prev_rank = i
        ranks[s] = prev_rank
    return ranks

def get_rank_map(obj, usefulness_metric):
    rankings = obj.get("rankings")
    if isinstance(rankings, dict):
        rmap = rankings.get(usefulness_metric)
        if isinstance(rmap, dict) and rmap:
            out = {}
            for k, v in rmap.items():
                try:
                    out[str(k)] = int(v)
                except Exception:
                    pass
            if out:
                return out

    by_step = obj.get("by_step")
    if not isinstance(by_step, dict) or not by_step:
        return None

    step_to_val = {}
    for step, st in by_step.items():
        if isinstance(st, dict) and usefulness_metric in st:
            try:
                step_to_val[str(step)] = float(st[usefulness_metric])
            except Exception:
                pass

    if not step_to_val:
        return None

    return make_competition_rank(step_to_val, reverse=True)  

def summarize(dirs, pattern, usefulness_metric):   
    def token_len(text: str) -> int:
        return len(text)
    bucket_nc, bucket_gemini, bucket_redundancy = defaultdict(list), defaultdict(list), defaultdict(list)
    bucket_ne = defaultdict(list)    
    bucket_diff = defaultdict(list)  
    bucket_rank = defaultdict(list)  
    bucket_memlen = defaultdict(list)
 
    n_files_seen = 0
    n_files_used = 0

    for p in iter_json_files(dirs, pattern=pattern):
        n_files_seen += 1
        try:
            obj = load_json(p)
        except Exception:
            continue

        memlen_by_step = defaultdict(int)
        rollout_logs = obj.get("rollout_logs")
        if isinstance(rollout_logs, list):
            for lg in rollout_logs:
                if not isinstance(lg, dict):
                    continue
                step = str(lg.get("global_step", "unknown"))
                per_desc = lg.get("per_desc")
                if isinstance(per_desc, list):
                    for it in per_desc:
                        if isinstance(it, dict) and it.get("desc") is not None:
                            memlen_by_step[step] += token_len(str(it["desc"]))

        by_step = obj.get("by_step")
        if not isinstance(by_step, dict) or not by_step:
            continue

        rmap = get_rank_map(obj, usefulness_metric)  # may be None
        n_files_used += 1

        for step, st in by_step.items():
            if not isinstance(st, dict):
                continue
            step = str(step)
            
            try:
                nc = float(st["num_correct"])
                bucket_nc[step].append(nc)
            except Exception:
                nc = None

            try:
                ne = float(st["num_wrong"])
                bucket_ne[step].append(ne)
            except Exception:
                ne = None

            try:
                n_gemini = 1 - float(st["num_gemini_correct"]) / float(st["num_memory"]) if float(st["num_memory"]) > 0 else 0
                bucket_gemini[step].append(n_gemini)
            except Exception:
                n_gemini = None

            try:
                n_redundancy = float(st["num_redundant"]) / float(st["num_memory"]) if float(st["num_memory"]) > 0 else 0
                bucket_redundancy[step].append(n_redundancy)
            except Exception:
                n_redundancy = None

            if step in memlen_by_step:
                bucket_memlen[step].append(float(memlen_by_step[step]))

            if nc is not None and ne is not None:
                bucket_diff[step].append(nc - ne)

            if isinstance(rmap, dict) and step in rmap:
                bucket_rank[step].append(float(rmap[step]))

    all_steps = set(bucket_nc) | set(bucket_ne) | set(bucket_diff) | set(bucket_rank) | set(bucket_memlen)
    per_step = {}
    for step in sorted(all_steps, key=_step_sort_key):
        per_step[step] = {
            "num_correct_mean": mean(bucket_nc[step]) if bucket_nc[step] else None,
            "num_error_mean": mean(bucket_ne[step]) if bucket_ne[step] else None,
            "num_gemini_mean": mean(bucket_gemini[step]) if bucket_gemini[step] else None,
            "num_redundancy_mean": mean(bucket_redundancy[step]) if bucket_redundancy[step] else None,
            "num_correct_minus_num_error_mean": mean(bucket_diff[step]) if bucket_diff[step] else None,
            "usefulness_rank_mean": mean(bucket_rank[step]) if bucket_rank[step] else None,
            "memory_length_mean": mean(bucket_memlen[step]) if bucket_memlen[step] else None,
            "count_files": max(
                len(bucket_nc[step]),
                len(bucket_ne[step]),
                len(bucket_diff[step]),
                len(bucket_rank[step]),
                len(bucket_memlen[step]),
            ),
        }

    return {
        "meta": {
            "dirs": [str(Path(d)) for d in dirs],
            "pattern": pattern,
            "usefulness_metric": usefulness_metric,
            "num_files_seen": n_files_seen,
            "num_files_used": n_files_used,
        },
        "per_step": per_step,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", type=str, required=True,
                    help="Comma-separated list of directories containing *_eval.json files.")
    ap.add_argument("--pattern", type=str, default="*_eval.json")
    ap.add_argument("--usefulness_metric", type=str, default=USEFULNESS_METRIC_DEFAULT)
    ap.add_argument("--out", type=str, required=True,
                    help="Destination file for the aggregated summary JSON.")
    args = ap.parse_args()

    dirs = [d.strip() for d in args.dirs.split(",") if d.strip()]
    out = summarize(dirs, args.pattern, args.usefulness_metric)

    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote: {args.out}")
    print(json.dumps(out["meta"], ensure_ascii=False))


if __name__ == "__main__":
    main()