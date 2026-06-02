# Copyright (2025) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import json
import base64
import io
import copy
import re
import logging
import random
from concurrent.futures import ThreadPoolExecutor
from time import sleep

import openai

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)

MAX_IMAGES_PER_REQUEST = 100
LOCAL_IMAGE_BUDGET_BYTES = 49_000_000
LOCAL_IMAGE_BUDGET_UPPER = 60_000_000

IMAGE_SIZE_LIMIT_PATTERNS = [
    # Example:
    # "Total image size is 53.63MB, which exceeds the allowed limit of 50.0MB."
    r"total image size .* exceeds .* allowed limit",
    r"allowed limit of \d+(\.\d+)?\s*mb",
    r"image size .* exceeds",
    r"code['\"]?\s*:\s*['\"]?-4003",
]

# Resolve the API config path from TASKMEM_API_CONFIG, falling back to the
# example file under configs/.
_DEFAULT_CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "api_config.json",
)
_API_CONFIG_PATH = os.environ.get("TASKMEM_API_CONFIG", _DEFAULT_CONFIG_PATH)

with open(_API_CONFIG_PATH, "r", encoding="utf-8") as _f:
    config = json.load(_f)

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

def get_response(model, messages):
    """Get chat completion response from specified model.

    Args:
        model (str): Model identifier, e.g. "gpt-5-2025-08-07#medium#high", "gemini-1.5-pro-002#think"
    Returns:
        str: response content
    """
    # Parse `#`: the prefix is the model key. For GPT models the optional
    # suffixes encode reasoning_effort / verbosity; for non-GPT models the
    # optional suffix is the thinking budget.
    parts = [p.strip() for p in model.split("#")]
    client_key = parts[0]
    is_gpt = "gpt" in client_key.lower()

    if is_gpt:
        reasoning_effort = parts[1] if len(parts) >= 2 else "medium"
        verbosity = parts[2] if len(parts) >= 3 else "medium"
    else:
        budget_tokens = 8192 if len(parts) > 1 else 128

    if client_key not in client:
        raise KeyError(f"model key not in config: {client_key!r}, available: {list(client.keys())}")
    selected_model = client[client_key]
    if isinstance(selected_model, list):
        selected_model = random.choice(selected_model)

    extra_body = {}
    if not is_gpt and "1.5" not in client_key:
        extra_body["thinking"] = {"include_thoughts": True, "budget_tokens": budget_tokens}

    temperature = 1 if is_gpt else 1e-6

    logger.debug(
        "[get_response] is_gpt=%s client_key=%r temperature=%s extra_body.keys=%s",
        is_gpt, client_key, temperature, list(extra_body.keys())
    )
    if is_gpt:
        logger.debug("[get_response] GPT reasoning_effort=%s verbosity=%s", reasoning_effort, verbosity)

    if is_gpt:
        response = selected_model.chat.completions.create(
            model=client_key,
            messages=messages,
            temperature=temperature,
            timeout=120,
            max_tokens=8192,
            reasoning_effort=reasoning_effort,
            verbosity=verbosity,
        )
    else:
        response = selected_model.chat.completions.create(
            model=client_key,
            messages=messages,
            temperature=temperature,
            timeout=120,
            max_tokens=8192,
            extra_body=extra_body,
        )
    return response.choices[0].message.content


def _contains_image_size_limit_signal(value) -> bool:
    if value is None:
        return False
    text = str(value).lower()
    return any(re.search(pattern, text) for pattern in IMAGE_SIZE_LIMIT_PATTERNS)


def _downgrade_image_detail_to_low(messages):
    new_messages = copy.deepcopy(messages)
    for message in new_messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not (isinstance(item, dict) and item.get("type") == "image_url"):
                continue
            image_url = item.get("image_url")
            if isinstance(image_url, dict) and image_url.get("detail") != "low":
                image_url["detail"] = "low"
    return new_messages


def _raise_video_too_large_error(model, original_error):
    raise ValueError(
        f"Vision request still exceeds image-size limits after switching to detail=low. "
        f"Please split the video into shorter segments and retry. model={model}, error={original_error}"
    )


def get_response_with_vision_limit_retry(model, messages):
    """Vision-only fallback: one downgrade attempt to detail=low.

    - First call with original messages.
    - If visual limit is hit, downgrade image detail to low and retry once.
    - If the second attempt still fails due to limits, raise a clear split-video error.
    """
    try:
        response = get_response(model, messages)
        if not _contains_image_size_limit_signal(response):
            return response
        trigger_error = response
    except Exception as e:
        if not _contains_image_size_limit_signal(e):
            raise
        trigger_error = e

    downgraded_messages = _downgrade_image_detail_to_low(messages)

    logger.warning(
        "Detected vision limit signal. Downgrade image detail to low and retry once. model=%s",
        model,
    )
    try:
        response = get_response(model, downgraded_messages)
        if _contains_image_size_limit_signal(response):
            _raise_video_too_large_error(model, response)
        return response
    except Exception as e:
        if _contains_image_size_limit_signal(e):
            _raise_video_too_large_error(model, e)
        raise


def _estimate_high_detail_bytes(width: int, height: int) -> int:
    """Estimate per-image size for high detail by official resizing rules."""
    w, h = float(width), float(height)
    if w <= 0 or h <= 0:
        return 0

    # Step 1: fit within 2048x2048 if needed.
    max_side = max(w, h)
    if max_side > 2048.0:
        s = 2048.0 / max_side
        w *= s
        h *= s

    # Step 2: force shortest side to 768.
    min_side = min(w, h)
    s = 768.0 / min_side
    w *= s
    h *= s
    return int(w * h)


def _video_to_frames_b64(video_path: str, num_frames: int) -> list[tuple[str, int, int]]:
    """Sample one frame per second, encode to base64 JPEG; respects duration and MAX_IMAGES_PER_REQUEST. Keep num_frames for possible control"""
    if num_frames <= 0:
        return []
    try:
        from moviepy import VideoFileClip
    except Exception:
        from moviepy.editor import VideoFileClip  # type: ignore
    from PIL import Image

    frames_b64 = []
    with VideoFileClip(video_path) as clip:
        duration = float(getattr(clip, "duration", 0.0) or 0.0)
        duration = max(0.0, duration)
        # One frame per second: t = 0, 1, 2, ...; cap by duration and num_frames/MAX
        safe_end = max(0.0, duration - 1e-3)
        n_sec = max(1, 1 + int(safe_end)) if duration > 0 else 1
        n_sec = min(n_sec, num_frames, MAX_IMAGES_PER_REQUEST)
        ts_list = [min(i, safe_end) for i in range(n_sec)]
        for t in ts_list:
            img = Image.fromarray(clip.get_frame(t))
            w, h = img.size
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=85)
            frames_b64.append((base64.b64encode(buf.getvalue()).decode("utf-8"), int(w), int(h)))
    logger.debug("Sampled %d frames from %s (duration %.1fs)", len(frames_b64), video_path, duration)
    return frames_b64


def generate_messages(inputs, model_name: str | None = None):
    messages = [{"role": "system", "content": "You are an expert in video understanding."}]
    content = []
    # GPT models do not accept inline video, so sample frames instead.
    use_frames = model_name and "gpt" in model_name.lower()
    for inp in inputs:
        if inp["type"] == "text":
            content.append(inp)
        elif inp["type"] == "video":
            path = inp["video"]
            if use_frames:
                n = min(int(inp.get("num_frames", 50)), MAX_IMAGES_PER_REQUEST)
                frames = _video_to_frames_b64(path, n)
                est_total = sum(_estimate_high_detail_bytes(w, h) for _, w, h in frames)
                logger.debug(
                    "[LOCAL_CHECK] start path=%s frames=%d est_total=%.2fMB detail=high",
                    path, len(frames), est_total / 1_000_000,
                )
                # If the request still exceeds the hard upper bound, fall back
                # to detail=low (which downsamples to 512x512). Otherwise drop
                # the earliest frames until the estimate fits.
                if est_total > LOCAL_IMAGE_BUDGET_UPPER:
                    detail = "low"
                else:
                    detail = "high"
                    while len(frames) > 1 and est_total > LOCAL_IMAGE_BUDGET_BYTES:
                        frames = frames[2:]
                        est_total = sum(_estimate_high_detail_bytes(w, h) for _, w, h in frames)
                        logger.debug(
                            "[LOCAL_CHECK] reduced frames=%d est_total=%.2fMB",
                            len(frames), est_total / 1_000_000,
                        )
                for b64, _, _ in frames:
                    content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": detail}})
            else:
                with open(path, "rb") as f:
                    b64 = base64.b64encode(f.read()).decode("utf-8")
                content.append({"type": "image_url", "image_url": {"url": f"data:video/mp4;base64,{b64}", "detail": "high"}})
        else:
            raise ValueError(f"Invalid input type: {inp['type']}")
    messages.append({"role": "user", "content": content})
    return messages


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Smoke test for the chat-openai backend.")
    parser.add_argument("--video", type=str, required=True, help="Path to an input .mp4 file.")
    parser.add_argument("--model", type=str, required=True,
                        help="Model identifier defined in the API config "
                             "(e.g. 'gpt-4o-2024-11-20' or 'gemini-1.5-pro-002').")
    parser.add_argument("--prompt", type=str,
                        default="Briefly describe what happens in this video clip.",
                        help="Prompt sent together with the video.")
    args = parser.parse_args()

    input_data = [
        {"type": "text", "text": args.prompt},
        {"type": "video", "video": args.video},
    ]
    messages = generate_messages(input_data, model_name=args.model)
    print(get_response_with_vision_limit_retry(args.model, messages))
