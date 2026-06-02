#!/usr/bin/env bash
#
# Streaming-memory inference with an off-the-shelf VLM.
#
# Runs the pipeline using a Gemini / GPT API backend (default) or a local
# vanilla Qwen3-VL via vLLM. Use this script to reproduce the baseline
# numbers reported in the paper or as a starting point for your own
# episodic-memory experiments.
#
# Pipeline (three sequential `src/main.py` invocations on the same OUTPUT_FOLDER):
#   1. --process_audio                : ASR + diarization, writes <id>_voice.json
#   2. --process_video --process_voice: face detection + speaker-id, renders
#                                        annotated per-clip mp4s
#   3. --generate_episodic            : streams the per-clip context through the
#                                        episodic VLM to write long-term memory
#
# Required environment variables:
#   VIDEO_PATH       Path to the input .mp4 file.
#   OUTPUT_FOLDER    Working directory for per-clip artefacts and outputs.
#
# Optional environment variables:
#   EPISODIC_MODEL   Backend used to write episodic memory.
#                      - gemini-* | gpt-*  (API; requires TASKMEM_API_CONFIG)
#                      - qwen3_vl_vllm     (local vLLM, vanilla Qwen3-VL)
#                      Default: gemini-2.5-flash
#   EPISODIC_MODEL_PATH  Required when EPISODIC_MODEL=qwen3_vl_vllm; the
#                         HF id or local path of the vanilla Qwen3-VL checkpoint.
#   ASR_MODEL        ASR backend (default: gemini-2.5-pro, API).
#   VOICE_MODEL      Backend used to match speakers to face ids
#                     (default: gemini-2.5-pro, API).
#   TOTAL_DURATION   Max number of seconds to process (default: 180).
#   READ_TAG / WRITE_TAG  Memory tags written by the video stage / episodic stage.
set -e

: "${VIDEO_PATH:?Please set VIDEO_PATH to an input .mp4 file}"
: "${OUTPUT_FOLDER:?Please set OUTPUT_FOLDER for the per-clip outputs}"
: "${EPISODIC_MODEL:=gemini-2.5-flash}"
: "${EPISODIC_MODEL_PATH:=}"
: "${ASR_MODEL:=gemini-2.5-pro}"
: "${VOICE_MODEL:=gemini-2.5-pro}"
: "${TOTAL_DURATION:=180}"
: "${READ_TAG:=baseline}"
: "${WRITE_TAG:=baseline_ep}"

# 1. ASR / diarization. Writes <video_id>_voice.json under OUTPUT_FOLDER.
python src/main.py \
    --video_path "$VIDEO_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --asr_model "$ASR_MODEL" \
    --write_memory_tag "$READ_TAG" \
    --end_time "$TOTAL_DURATION" \
    --process_audio

# 2. Face detection + speaker matching. Reads the voice JSON from step 1 and
#    renders per-clip mp4s with face boxes and subtitles.
python src/main.py \
    --video_path "$VIDEO_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --voice_model "$VOICE_MODEL" \
    --write_memory_tag "$READ_TAG" \
    --end_time "$TOTAL_DURATION" \
    --process_video --process_voice

# 3. Episodic memory generation with an off-the-shelf VLM.
python src/main.py \
    --video_path "$VIDEO_PATH" \
    --video_folder "$OUTPUT_FOLDER" \
    --episodic_folder "$OUTPUT_FOLDER" \
    --episodic_model "$EPISODIC_MODEL" \
    --episodic_model_path "$EPISODIC_MODEL_PATH" \
    --read_memory_tag "$READ_TAG" \
    --write_memory_tag "$WRITE_TAG" \
    --end_time "$TOTAL_DURATION" \
    --generate_episodic
