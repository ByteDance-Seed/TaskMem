import os
import base64
import importlib
from tqdm import tqdm
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict
from memory.long_term_memory import Character
from tools.video_process import extract_frames

MAX_RETRIES = 5

# Pluggable face backend.
#
# By default we run face detection / embedding / clustering locally using
# `tools.face_extraction` (insightface RetinaFace + ArcFace) and
# `tools.face_clustering` (HDBSCAN).
#
# To plug in a remote / custom backend, set the env var
# `TASKMEM_FACE_BACKEND=your_pkg.your_module` and that module must expose
# two callables with the following signatures:
#
#     def extract_faces(frames: list[str]) -> list[dict]:
#         """Each frame is base64-encoded JPEG. Each returned face dict has
#         keys: frame_id, bounding_box, face_emb, cluster_id (=-1), extra_data
#         (with face_base64 / face_detection_score / face_quality_score)."""
#
#     def cluster_faces(faces: list[dict], min_cluster_size: int) -> list[dict]:
#         """Assign cluster_id (>=0 or -1 for noise) to each face dict in place
#         (or return new list of dicts with cluster_id filled)."""
_face_backend = os.environ.get("TASKMEM_FACE_BACKEND", "tools.face_extraction")
_face_cluster_backend = os.environ.get("TASKMEM_FACE_CLUSTER_BACKEND", "tools.face_clustering")
extract_faces = importlib.import_module(_face_backend).extract_faces
cluster_faces = importlib.import_module(_face_cluster_backend).cluster_faces

def process_batch(params):
    frames = params[0]
    offset = params[1]
    for _ in range(MAX_RETRIES):
        try:
            faces = extract_faces(frames)
            break
        except Exception as e:
            faces = []
    for face in faces:
        face["frame_id"] += offset
    return faces

def process_faces(base64_frames, min_cluster_size, batch_size):
    num_batches = (len(base64_frames) + batch_size - 1) // batch_size
    batched_frames = [
        (base64_frames[i * batch_size : (i + 1) * batch_size], i * batch_size)
        for i in range(num_batches)
    ]

    faces = []
    # parallel process the batches
    with ThreadPoolExecutor(max_workers=num_batches) as executor:
        for batch_faces in tqdm(
            executor.map(process_batch, batched_frames), total=num_batches
        ):
            faces.extend(batch_faces)

    for _ in range(MAX_RETRIES):
        try:
            clustered_faces = cluster_faces(faces, min_cluster_size)
            break
        except Exception as e:
            clustered_faces = []
    return clustered_faces

def process_and_stitch_images(faces, size=(108, 108), images_per_row=10):
    from io import BytesIO
    from PIL import Image, ImageDraw, ImageFont
    import time
    processed_images = []
    for face in faces:
        b64_str = face["extra_data"]["face_base64"]
        img_data = base64.b64decode(b64_str)
        img = Image.open(BytesIO(img_data))
        resized_img = img.resize(size, Image.Resampling.LANCZOS)
        draw = ImageDraw.Draw(resized_img)
        font = ImageFont.load_default()
        draw.text((0, 0), str(face["extra_data"]["face_detection_score"]), font=font)
        draw.text((0, 20), str(face["extra_data"]["face_quality_score"]), font=font)
        draw.text((0, 40), str(face["cluster_id"]), font=font)
        draw.text((0, 60), str(face["bounding_box"]), font=font)
        processed_images.append(resized_img)
    
    num_images = len(processed_images)
    rows = (num_images + images_per_row - 1) // images_per_row  # round up
    width = size[0] * min(images_per_row, num_images)
    height = size[1] * rows
    stitched_image = Image.new('RGB', (width, height))
    for index, img in enumerate(processed_images):
        row = index // images_per_row
        col = index % images_per_row
        x = col * size[0]
        y = row * size[1]
        stitched_image.paste(img, (x, y))
    stitched_image.save('face_{}.jpg'.format(int(time.time())))

def get_face_id(faces, memory):
    tempid2face = {}
    for face in faces:
        if face["cluster_id"] == -1:
            continue
        if face["cluster_id"] not in tempid2face:
            tempid2face[face["cluster_id"]] = []
        tempid2face[face["cluster_id"]].append(face)
    face_ids, face2frame = set(), defaultdict(list)
    for _, face_list in tempid2face.items():
        q_embs = [face["face_emb"] for face in face_list]
        sim_face = memory.semantic.search_node(q_embs, "face")
        if len(sim_face) > 0:
            face_id, merge = sim_face[0][0], True
            if face_id in face2frame:
                for face in face_list:
                    if face["frame_id"] in face2frame[face_id]:
                        merge = False
                        break
            if merge:
                face_ids.add(face_id)
                for face in face_list:
                    face2frame[face_id].append(face["frame_id"])
                memory.semantic.add_node(face_id, [{
                    "embedding": face["face_emb"],
                    "base64": face["extra_data"]["face_base64"]
                } for face in face_list], "face")
                for face in face_list:
                    face["cluster_id"] = face_id
                continue
        character = Character(memory.semantic.get_node_id(), [{
            "embedding": face["face_emb"],
            "base64": face["extra_data"]["face_base64"]
        } for face in face_list])
        face_id = character.id
        for face in face_list:
            face2frame[face_id].append(face["frame_id"])
        memory.semantic.characters[face_id] = character
        for face in face_list:
            face["cluster_id"] = face_id
    return faces, list(face_ids)

def calculate_iou(box1, box2):
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)    
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area != 0 else 0.0

def serialize_faces(faces, minimum_occurrence):
    face_dict, face_lists = {}, []
    for face in faces:
        if not all(x >= 0 for x in face["bounding_box"]):
            continue
        if face["frame_id"] not in face_dict:
            face_dict[face["frame_id"]] = []
        face_dict[face["frame_id"]].append(face)
    face_dict = OrderedDict(sorted(face_dict.items(), key=lambda x: x[0]))
    total_frames = len(face_dict)
    while face_dict:
        del_id = []
        for frame_id in face_dict:
            face_list = [face_dict[frame_id][0]]
            face_dict[frame_id] = face_dict[frame_id][1:]
            if len(face_dict[frame_id]) == 0:
                del_id.append(frame_id)
            for idx in range(frame_id + 1, total_frames):
                if idx not in face_dict:
                    break
                for i, face in enumerate(face_dict[idx]):
                    iou = calculate_iou(face_list[-1]["bounding_box"], face["bounding_box"])
                    if iou > 0.6:
                        face_list.append(face)
                        del face_dict[idx][i]
                        if len(face_dict[idx]) == 0:
                            del_id.append(idx)
                        break
            face_lists.append(face_list)
            break
        for id in del_id:
            del face_dict[id]
        
    result_face_dic = {}
    for face_list in face_lists:
        cluster_ids = []
        for face in face_list:
            if face["cluster_id"] != -1:
                cluster_ids.append(face["cluster_id"])
        if len(cluster_ids) == 0:
            continue

        count_dic = {}
        for cluster_id in cluster_ids:
            if cluster_id not in count_dic:
                count_dic[cluster_id] = 0
            count_dic[cluster_id] += 1

        max_cluster_id = max(count_dic, key=count_dic.get)
        for face in face_list:
            face["cluster_id"] = max_cluster_id
        if max_cluster_id not in result_face_dic:
            result_face_dic[max_cluster_id] = face_list
        else:
            result_face_dic[max_cluster_id].extend(face_list)
    
    result_face = []
    for _, faces in result_face_dic.items():
        if len(faces) >= minimum_occurrence:
            result_face.extend(faces)

    return result_face

def face_tools(clip, old_faces, truncate, new_duration, args, memory, results=None):
    # extract and cluster faces, assign global id
    face_start, face_end = clip.duration - new_duration, clip.duration
    new_clip = clip.subclipped(face_start, face_end)
    frames = extract_frames(new_clip, args.fps)
    faces = process_faces(frames, args.min_cluster_size, args.num_workers)
    faces = serialize_faces(faces, 0.6 * args.fps)
    faces, face_ids = get_face_id(faces, memory) # Assign each face a unique ID
    new_faces, truncate_frames = [], args.interval * args.fps if truncate else 0
    for face in old_faces:
        face["frame_id"] -= truncate_frames
        if face["frame_id"] >= 0:
            new_faces.append(face)
    for face in faces:
        face["frame_id"] += face_start * args.fps
        new_faces.append(face)
    if results is not None:
        results.put(("face", [new_faces, face_ids])) # parallel processing
    else:
        return new_faces, face_ids # serial processing

if __name__ == "__main__":
    import argparse
    from moviepy import VideoFileClip

    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=str, required=True, help="path to a local mp4 file")
    parser.add_argument("--start", type=float, default=40.0)
    parser.add_argument("--end", type=float, default=50.0)
    parser.add_argument("--fps", type=int, default=10)
    parser.add_argument("--min_cluster_size", type=int, default=3)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--minimum_occurrence", type=int, default=6)
    args = parser.parse_args()

    video = VideoFileClip(args.video)
    new_clip = video.subclipped(args.start, args.end)
    frames = extract_frames(new_clip, args.fps)
    faces = process_faces(frames, args.min_cluster_size, args.num_workers)
    faces = serialize_faces(faces, args.minimum_occurrence)
    process_and_stitch_images(faces)