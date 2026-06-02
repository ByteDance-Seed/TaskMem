import os
import cv2
import base64
import logging
import numpy as np
from PIL import ImageFont, ImageDraw, Image
from moviepy import CompositeVideoClip, ImageClip

logging.getLogger('moviepy').setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# Font paths used when overlaying face labels and subtitles on top of clips.
# Override with TASKMEM_LABEL_FONT (ASCII font used for face labels) and
# TASKMEM_SUBTITLE_FONT (CJK-capable font used for subtitles).
LABEL_FONT_PATH = os.environ.get(
    "TASKMEM_LABEL_FONT",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
)
SUBTITLE_FONT_PATH = os.environ.get(
    "TASKMEM_SUBTITLE_FONT",
    LABEL_FONT_PATH,
)

def extract_frames(video, sample_fps=10):

    start = 0
    interval = video.duration

    frames = []
    frame_interval = 1.0 / sample_fps

    for t in np.arange(
        start, min(start + interval, video.duration), frame_interval
    ):
        frame = video.get_frame(t)
        _, buffer = cv2.imencode(".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        frames.append(base64.b64encode(buffer).decode("utf-8"))
        
    return frames

class TextClip(ImageClip):
    def __init__(self, text, font, font_size, size, position=None, color="black", bg_color=(255, 255, 255, 128), interval=5, margin=(50, 80)):
        img = Image.new("RGBA", size, color=(0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(font, font_size)
        bbox, lines, line_height = self.get_wrapped_text_bbox(text, font, size[0] - margin[0] * 2, interval=interval)
        if position: # left-bottom
            x, y = position
            y -= bbox[3] - bbox[1] + 2 * interval
        else: # subtitle
            x = (size[0] - (bbox[2] - bbox[0]) - interval * 2) // 2
            y = size[1] - margin[1] - (bbox[3] - bbox[1]) - interval * 2
        draw.rectangle([(x, y), (x + bbox[2] - bbox[0] + interval * 2, y + bbox[3] - bbox[1] + interval * 2)], fill=bg_color)
        x += interval
        if position: # for NotoSansCJK-Regular.ttc
            y += interval
        for line in lines:
            draw.text((x, y), line, font=font, fill=color)
            y += line_height
        img_numpy = np.array(img)
        ImageClip.__init__(self, img=img_numpy, transparent=True)
    
    def get_wrapped_text_bbox(self, text, font, max_width, interval=4):
        draw = ImageDraw.Draw(Image.new('RGB', (1, 1)))
        lines, current_line, words = [], "", text.split(' ')
        all_bboxes, y_offset, line_height = [], 0, 0
        for word in words:
            test_line = current_line + (word if not current_line else ' ' + word)
            bbox = draw.textbbox((0, 0), test_line, font=font)
            line_width = bbox[2] - bbox[0]

            if line_width > max_width:
                if current_line:
                    lines.append(current_line)
                    current_line = word
                else:
                    lines.append(word)
                    current_line = ""
                bbox = draw.textbbox((0, y_offset), lines[-1], font=font)
                all_bboxes.append(bbox)
                if len(lines) == 1:
                    line_height = bbox[3] - bbox[1] + interval
                y_offset += line_height
            else:
                current_line = test_line
        
        if current_line:
            lines.append(current_line)
            bbox = draw.textbbox((0, y_offset), lines[-1], font=font)
            all_bboxes.append(bbox)
            if len(lines) == 1:
                line_height = bbox[3] - bbox[1] + interval
        
        if not all_bboxes:
            return (0, 0, 0, 0), [], 0
        xmin = min(bbox[0] for bbox in all_bboxes)
        ymin = min(bbox[1] for bbox in all_bboxes)
        xmax = max(bbox[2] for bbox in all_bboxes)
        ymax = max(bbox[3] for bbox in all_bboxes)
        return (xmin, ymin, xmax, ymax), lines, line_height

def draw_boxes(clip, faces, voices, fps, id_map=None):
    boxes = {}
    for face in faces:
        assert face["cluster_id"] != -1
        label = "[face_{}]".format(face["cluster_id"])
        if id_map is not None:
            label = id_map[label]
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
    for k, v in boxes.items():
        start = max((float(k) - 0.5) / fps, 0)
        end = min((float(k) + 0.5) / fps, clip.duration)
        for label in v:
            if label[1] != "":
                text_clip = TextClip(
                    font=LABEL_FONT_PATH,
                    text=label[1],
                    font_size=16,
                    size=new_clips[0].size,
                    position=(label[0][0], label[0][1]),
                    color=(255, 255, 255),
                    bg_color=(255, 0, 0, 255)
                )
                text_clip = text_clip.with_start(start).with_end(end)
                new_clips.append(text_clip)
    
    # Ensure that two voices separated by one voice do not intersect
    for i in range(max(0, len(voices) - 2)):
        if voices[i]["end_time"] > voices[i + 2]["start_time"]:
            voices[i]["end_time"] = voices[i + 2]["start_time"]

    margin, end = (50, 80), 0
    for voice in voices:
        if voice["start_time"] < end:
            if margin[1] == 80:
                margin = (50, 50)
            else:
                margin = (50, 80)
        else:
            margin = (50, 80)
        end = voice["end_time"]
        subtitle = voice["asr"]
        if id_map is not None and "speaker" in voice and voice["speaker"] != "unknown":
            subtitle = "{}: {}".format(id_map[voice["speaker"]], subtitle)
        text_clip = TextClip(
            font=SUBTITLE_FONT_PATH,
            text=subtitle,
            font_size=24,
            size=new_clips[0].size,
            color=(0, 0, 0),
            bg_color=(255, 255, 255, 128),
            margin=margin
        )
        text_clip = text_clip.with_start(voice["start_time"]).with_end(end)
        new_clips.append(text_clip)

    new_clip = CompositeVideoClip(new_clips)
    for clip in new_clips[1:]:
        clip.close()
    return new_clip