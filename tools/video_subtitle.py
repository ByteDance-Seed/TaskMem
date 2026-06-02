#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Burn subtitles (from per-clip ASR JSON files) onto rendered video segments.

Each `<tag>` directory under ``--prepared-root`` is expected to contain a set
of ``{label}_720p.mp4`` clips and matching ``{label}_asr.json`` files
(produced upstream by the audio pipeline). The script renders the clips with
subtitles overlaid and writes them next to the sources as ``{label}_sub.mp4``.
"""

import os
import json
import copy
import shutil
import argparse
import logging
from pathlib import Path
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from moviepy import VideoFileClip, TextClip, CompositeVideoClip
except Exception:
    from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip


SEGMENT_LABELS = ("0_30", "30_60", "60_90")
DEFAULT_FONT_PATH = os.environ.get(
    "TASKMEM_SUBTITLE_FONT",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
)


def setup_logger(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s | %(levelname)s | %(message)s")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_paths(tag_dir: Path, label: str) -> Tuple[Path, Path, Path]:
    """Return (source video, source asr json, final subtitled video) paths."""
    src_video = tag_dir / f"{label}_720p.mp4"
    src_asr = tag_dir / f"{label}_asr.json"
    final_sub_video = tag_dir / f"{label}_sub.mp4"
    return src_video, src_asr, final_sub_video


def move_file(local_path: Path, final_path: Path, overwrite: bool = False) -> None:
    ensure_dir(final_path.parent)
    if final_path.exists():
        if overwrite:
            final_path.unlink()
        else:
            raise FileExistsError(f"Destination file already exists: {final_path}")
    shutil.move(str(local_path), str(final_path))


def draw_boxes(clip, faces, voices, fps, font_path: str, id_map=None):
    boxes = {}
    for face in faces:
        assert face["cluster_id"] != -1
        label = "[face_{}]".format(face["cluster_id"])
        if id_map is not None:
            label = id_map.get(label, label)
        if face["frame_id"] not in boxes:
            boxes[face["frame_id"]] = []
        boxes[face["frame_id"]].append((face["bounding_box"], label))

    def process_frame(frame, t):
        frame_copy = frame.copy()
        frame_id = int(t * fps + 1.0 / 2)
        if frame_id in boxes:
            for box in boxes[frame_id]:
                x1, y1, x2, y2 = box[0]
                color, thickness = (255, 0, 0), 2
                frame_copy[y1: y1 + thickness, x1: x2] = color
                frame_copy[y2 - thickness: y2, x1: x2] = color
                frame_copy[y1: y2, x1: x1 + thickness] = color
                frame_copy[y1: y2, x2 - thickness: x2] = color
        return frame_copy

    new_clips = [clip.transform(lambda gf, t: process_frame(gf(t), t))]

    # Face labels.
    for k, v in boxes.items():
        start = max((float(k) - 0.5) / fps, 0)
        end = min((float(k) + 0.5) / fps, clip.duration)
        for label in v:
            if label[1] != "":
                text_clip = TextClip(
                    font=font_path,
                    text=label[1],
                    font_size=16,
                    color="white",
                    bg_color="red",
                )
                text_clip = text_clip.with_position((label[0][0], label[0][1]))
                text_clip = text_clip.with_start(start).with_end(end)
                new_clips.append(text_clip)

    # Operate on a copy so we don't mutate the caller's voice list.
    voices = copy.deepcopy(voices)

    # Original subtitle clipping logic: if a voice overlaps with the voice two
    # positions later, truncate it so neighbouring captions don't pile up.
    for i in range(max(0, len(voices) - 2)):
        if voices[i]["end_time"] > voices[i + 2]["start_time"]:
            voices[i]["end_time"] = voices[i + 2]["start_time"]

    clip_w, clip_h = clip.size
    bottom_margin = 20  # distance from the bottom edge in pixels
    line_gap = 10       # vertical gap between stacked subtitle lines
    last_end = 0
    current_level = 0   # 0 = bottom row, 1 = row above

    for voice in voices:
        subtitle = voice["asr"]
        if id_map is not None and "speaker" in voice and voice["speaker"] != "unknown":
            subtitle = "{}: {}".format(id_map[voice["speaker"]], subtitle)

        # Avoid spanning the entire frame width; let the renderer wrap text
        # within ~86% of the clip and size the background to the wrapped text.
        text_clip = TextClip(
            font=font_path,
            text=subtitle,
            font_size=24,
            size=(int(clip_w * 0.86), None),
            color="black",
            bg_color=(255, 255, 255, 180),
            text_align="center",
            margin=(16, 10),
        )

        # If the new subtitle overlaps the previous one, stack it one row up.
        if voice["start_time"] < last_end:
            current_level = 1 - current_level
        else:
            current_level = 0

        y = clip_h - text_clip.h - bottom_margin - current_level * (text_clip.h + line_gap)

        text_clip = text_clip.with_position(("center", y))
        text_clip = text_clip.with_start(voice["start_time"]).with_end(voice["end_time"])

        new_clips.append(text_clip)
        last_end = voice["end_time"]

    new_clip = CompositeVideoClip(new_clips).with_duration(clip.duration)

    for c in new_clips[1:]:
        try:
            c.close()
        except Exception:
            pass

    return new_clip


def render_one_video(
    src_video: Path,
    src_asr_json: Path,
    local_out_video: Path,
    final_out_video: Path,
    font_path: str,
    font_size: int,
    overwrite: bool = False,
) -> None:
    if not src_video.exists():
        raise FileNotFoundError(f"video does not exist: {src_video}")
    if not src_asr_json.exists():
        raise FileNotFoundError(f"asr json does not exist: {src_asr_json}")

    if final_out_video.exists() and not overwrite:
        logging.info(f"[SKIP] already exists: {final_out_video}")
        return

    ensure_dir(local_out_video.parent)

    asr_data = load_json(src_asr_json)
    voices = asr_data["voices"]

    clip = None
    new_clip = None
    try:
        clip = VideoFileClip(str(src_video))
        fps = getattr(clip, "fps", None) or 25

        new_clip = draw_boxes(
            clip=clip,
            faces=[],
            voices=voices,
            fps=fps,
            font_path=font_path,
            id_map=None,
        )

        if local_out_video.exists():
            local_out_video.unlink()

        logging.info(f"[RENDER] {src_video} -> {local_out_video}")
        new_clip.write_videofile(
            str(local_out_video),
            codec="libx264",
            audio_codec="aac",
            fps=fps,
            preset="medium",
            threads=4,
            logger=None,
        )

        move_file(local_out_video, final_out_video, overwrite=overwrite)

    finally:
        if new_clip is not None:
            try:
                new_clip.close()
            except Exception:
                pass
        if clip is not None:
            try:
                clip.close()
            except Exception:
                pass


def process_one_tag(
    tag_dir: Path,
    local_tmp_root: Path,
    font_path: str,
    font_size: int,
    overwrite: bool = False,
) -> Tuple[str, str]:
    tag = tag_dir.name
    try:
        stage_dir = local_tmp_root / tag
        if stage_dir.exists():
            shutil.rmtree(stage_dir)
        ensure_dir(stage_dir)

        for label in SEGMENT_LABELS:
            src_video, src_asr, final_sub_video = build_paths(tag_dir, label)
            local_sub_video = stage_dir / f"{label}_sub.mp4"

            if not src_video.exists():
                logging.warning(f"[SKIP] missing video: {src_video}")
                continue
            if not src_asr.exists():
                logging.warning(f"[SKIP] missing asr json: {src_asr}")
                continue

            render_one_video(
                src_video=src_video,
                src_asr_json=src_asr,
                local_out_video=local_sub_video,
                final_out_video=final_sub_video,
                font_path=font_path,
                font_size=font_size,
                overwrite=overwrite,
            )

        if stage_dir.exists():
            shutil.rmtree(stage_dir)

        logging.info(f"[OK] {tag}")
        return tag, "ok"

    except Exception as e:
        logging.exception(f"[FAIL] {tag}: {e}")
        return tag, "fail"


def main():
    parser = argparse.ArgumentParser(description="Burn subtitles into prepared 720p videos.")
    parser.add_argument("--prepared-root", type=str, required=True,
                        help="Root directory containing one sub-directory per tag.")
    parser.add_argument("--local-tmp-root", type=str, default="./taskmem_render_tmp",
                        help="Local scratch directory for intermediate renders.")
    parser.add_argument("--tag", default=None, help="Process only a single tag sub-directory.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing *_sub.mp4.")
    parser.add_argument("--workers", type=int, default=1, help="Number of tags processed in parallel.")
    parser.add_argument("--font-path", default=DEFAULT_FONT_PATH,
                        help="TTF/TTC font used for subtitles.")
    parser.add_argument("--font-size", type=int, default=24, help="Subtitle font size.")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging.")
    args = parser.parse_args()

    setup_logger(args.verbose)

    prepared_root = Path(args.prepared_root)
    local_tmp_root = Path(args.local_tmp_root)
    ensure_dir(local_tmp_root)

    if args.tag is not None:
        tag_dirs = [prepared_root / args.tag]
    else:
        tag_dirs = [p for p in prepared_root.iterdir() if p.is_dir()]

    ok_count = 0
    fail_count = 0

    if args.workers <= 1:
        for tag_dir in tag_dirs:
            _, status = process_one_tag(
                tag_dir=tag_dir,
                local_tmp_root=local_tmp_root,
                font_path=args.font_path,
                font_size=args.font_size,
                overwrite=args.overwrite,
            )
            if status == "ok":
                ok_count += 1
            else:
                fail_count += 1
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = []
            for tag_dir in tag_dirs:
                futures.append(
                    executor.submit(
                        process_one_tag,
                        tag_dir,
                        local_tmp_root,
                        args.font_path,
                        args.font_size,
                        args.overwrite,
                    )
                )
            for future in as_completed(futures):
                _, status = future.result()
                if status == "ok":
                    ok_count += 1
                else:
                    fail_count += 1

    logging.info("=" * 60)
    logging.info(f"Done: ok={ok_count}, fail={fail_count}")


if __name__ == "__main__":
    main()
