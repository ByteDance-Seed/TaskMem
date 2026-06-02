import os
import json
import shutil
import pickle
import argparse
from moviepy import VideoFileClip

from memory.long_term_memory import LongTermMemory
from memory.short_term_memory import ShortTermMemory
from memory.memory_process import generate_episodic, generate_semantic

def parse_args():
    """Entry-point for the TaskMem inference pipeline.

    Each invocation runs exactly one of: audio preprocessing, video
    preprocessing, episodic memory generation, or semantic memory generation.
    The full streaming-memory flow is three sequential invocations against
    the same ``--video_folder``:

        # 1. ASR + diarization. Writes <video_id>_voice.json.
        python src/main.py --video_path /path/to/clip.mp4 \\
            --video_folder ./outputs/demo \\
            --asr_model gemini-2.5-pro \\
            --end_time 180 --process_audio

        # 2. Face detection + speaker matching. Reads the voice JSON from
        #    step 1 and renders per-clip mp4s with face boxes and subtitles.
        python src/main.py --video_path /path/to/clip.mp4 \\
            --video_folder ./outputs/demo \\
            --voice_model gemini-2.5-pro \\
            --write_memory_tag raw \\
            --end_time 180 --process_video --process_voice

        # 3. Episodic memory generation.
        python src/main.py --video_path /path/to/clip.mp4 \\
            --video_folder ./outputs/demo \\
            --episodic_folder ./outputs/demo \\
            --read_memory_tag raw --write_memory_tag baseline_ep \\
            --end_time 180 --generate_episodic

    See ``examples/run_baseline.sh`` and ``examples/run_taskmem.sh`` for
    end-to-end driver scripts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True,
                        help="Path to the input .mp4 file.")
    parser.add_argument("--video_folder", type=str, default="",
                        help="Folder where per-clip artefacts (video / voice / map) are written.")
    parser.add_argument("--episodic_folder", type=str, default="",
                        help="Folder used when reading or writing episodic memory.")
    parser.add_argument("--semantic_folder", type=str, default="",
                        help="Folder used when reading or writing semantic memory.")
    # Distinguish memories produced by different models or experiment runs.
    parser.add_argument("--read_memory_tag", type=str, default="")
    parser.add_argument("--write_memory_tag", type=str, default="")
    parser.add_argument("--episodic_model", type=str, default="gemini-2.5-flash",
                        help="Backend used to generate episodic memory "
                             "(gemini-*, gpt-*, qwen3_vl_vllm).")
    parser.add_argument("--episodic_model_path", type=str, default="",
                        help="Local / HF path for the episodic VLM "
                             "(only required for qwen3_vl_vllm).")
    parser.add_argument("--semantic_model", type=str, default="gemini-2.5-flash")
    parser.add_argument("--semantic_model_path", type=str, default="")
    parser.add_argument("--asr_model", type=str, default="gemini-2.5-pro",
                        help="Backend used for ASR / voice transcription "
                             "(gemini-* or gpt-*; default gemini-2.5-pro).")
    parser.add_argument("--asr_model_path", type=str, default="",
                        help="Local / HF path for the ASR model (only used when "
                             "ASR backend is a local Qwen variant).")
    parser.add_argument("--voice_model", type=str, default="gemini-2.5-pro",
                        help="Backend used to match speakers to face IDs "
                             "(gemini-* or gpt-*; default gemini-2.5-pro).")
    parser.add_argument("--voice_model_path", type=str, default="")
    parser.add_argument("--supplement_prompt", type=str, default="")
    parser.add_argument("--start_time", type=int, default=0)
    # How long (in seconds) of the video to process; 0 = full duration.
    parser.add_argument("--end_time", type=int, default=0)
    parser.add_argument("--interval", type=int, default=10)
    parser.add_argument("--num_workers", type=int, default=64)
    # Minimum face cluster size for HDBSCAN.
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--fps", type=int, default=10)
    # Maximum number of seconds of preceding context that is fed back into the VLM.
    parser.add_argument("--context_duration", type=int, default=50)
    parser.add_argument("--extra_data", type=str, default="{}")
    parser.add_argument("--generate_episodic", action="store_true", default=False)
    parser.add_argument("--generate_semantic", action="store_true", default=False)
    parser.add_argument("--process_audio", action="store_true", default=False)
    parser.add_argument("--process_video", action="store_true", default=False)
    parser.add_argument("--process_voice", action="store_true", default=False)
    parser.add_argument("--asr_only", action="store_true", default=False)
    return parser.parse_args()

def fix_interval_schedule(video, args, file_id):
    if args.end_time <= 0:
        args.end_time = video.duration
    else:
        args.end_time = min(video.duration, args.end_time)
    if not 0 <= args.start_time <= args.end_time:
        args.start_time = 0
    start_time, end_time = args.start_time - args.context_duration, args.start_time
    
    while end_time < args.end_time - 1:
        start_time += args.interval
        truncate = False if start_time <= args.start_time else True
        new_end_time = min(end_time + args.interval, args.end_time)
        actrual_start_time = max(start_time, args.start_time)
        clip_id = str(end_time // args.interval)
        if args.process_audio or args.process_video:
            clip = video.subclipped(actrual_start_time, new_end_time)
        else:
            clip = VideoFileClip(os.path.join(args.video_folder, clip_id, "{}_{}.mp4".format(file_id, clip_id)))
            if not truncate:
                assert abs(new_end_time - actrual_start_time - clip.duration) < 0.5, \
                f"video length error, duration {clip.duration} != end_time {new_end_time} - start_time {actrual_start_time}, please redo --process_video"
        yield clip_id, clip, truncate, new_end_time - end_time
        end_time = new_end_time

def main():
    args = parse_args()
    file_id = args.video_path.split("/")[-1][:-4] # .mp4
    assert sum([args.process_video, args.process_audio, args.generate_episodic, args.generate_semantic]) == 1, "Only support one task at each time."
    assert args.start_time % args.interval == 0, f"start_time {args.start_time} must be divisible by interval {args.interval}."
    
    if args.process_audio or args.process_video:
        output_folder = args.video_folder
        memory = LongTermMemory(interval=args.interval)
    elif args.generate_episodic:
        with open(os.path.join(args.video_folder, f"{file_id}_{args.read_memory_tag}.pkl"), "rb") as f:
            memory = pickle.load(f)
        output_folder = args.episodic_folder
    elif args.generate_semantic:
        with open(os.path.join(args.episodic_folder, f"{file_id}_{args.read_memory_tag}.pkl"), "rb") as f:
            memory = pickle.load(f)
        output_folder = args.semantic_folder
    else:
        memory = LongTermMemory(interval=args.interval)
    os.makedirs(output_folder, exist_ok=True)
    video = VideoFileClip(args.video_path)
    context = ShortTermMemory(args=args)  
    
    if args.generate_semantic:
        episodic_out = json.load(open(os.path.join(args.episodic_folder, f"episodic_{args.read_memory_tag}.json")))
    else:
        episodic_out = {}

    # Process video in sequence
    schedule = fix_interval_schedule(video, args, file_id)
    for clip_id, clip, truncate, new_duration in schedule:
        os.makedirs(os.path.join(output_folder, clip_id), exist_ok=True)
        with open(os.path.join(output_folder, clip_id, "{}_{}.txt".format(file_id, args.write_memory_tag)), "w") as f:
            
            # process video and audio, prepare history memory
            context.add_clip(clip_id, clip, truncate, new_duration, args, memory, f, file_id)
            
            # generate memory
            if args.generate_episodic or args.generate_semantic:
                memory_io, memory_io_path = {}, os.path.join(output_folder, clip_id, f"{file_id}_{clip_id}_{args.write_memory_tag}.json")
                if os.path.exists(memory_io_path):
                    try:
                        memory_io = json.load(open(memory_io_path))
                    except:
                        pass
                
                # episodic
                if args.generate_episodic:
                    episodic_memory, episodic_io = generate_episodic(context, args, f)
                    memory_io["episodic"] = episodic_io
                    episodic_out[clip_id] = episodic_memory
                else:
                    episodic_memory = episodic_out[clip_id]
                f.write("Episodic memory local id:\n" + json.dumps(episodic_memory, indent=4, ensure_ascii=False) + "\n")
                episodic_memory = context.update_episodic(episodic_memory)
                f.write("Episodic memory global id:\n" + json.dumps(episodic_memory, indent=4, ensure_ascii=False) + "\n\n")
                
                # semantic
                semantic_memory = None
                if args.generate_semantic:
                    semantic_memory, semantic_io = generate_semantic(context, args, f)
                    f.write("Semantic memory local id:\n" + json.dumps(semantic_memory, indent=4, ensure_ascii=False) + "\n")
                    semantic_memory = context.update_semantic(semantic_memory)
                    f.write("Semantic memory global id:\n" + json.dumps(semantic_memory, indent=4, ensure_ascii=False) + "\n\n")
                    memory_io["semantic"] = semantic_io
                
                memory.update_memory(clip_id, semantic_memory, episodic_memory)
                json.dump(memory_io, open(memory_io_path, "w"), indent=4, ensure_ascii=False)
    video.close()
    if args.process_video or args.generate_episodic or args.generate_semantic:
        if args.generate_episodic:
            json.dump(episodic_out, open(os.path.join(output_folder, f"episodic_{args.write_memory_tag}.json"), "w"), indent=4, ensure_ascii=False)
        with open(f"/tmp/{file_id}.pkl", "wb") as f:
            pickle.dump(memory, f)
        memory_path = os.path.join(output_folder, f"{file_id}_{args.write_memory_tag}.pkl")
        shutil.copy2(f"/tmp/{file_id}.pkl", memory_path)

if __name__ == "__main__":
    main()