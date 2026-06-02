import cv2
import numpy as np
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
import base64
from insightface.app import FaceAnalysis
face_app = FaceAnalysis(name="buffalo_l")  # RetinaFace + ArcFace
face_app.prepare(ctx_id=-1)

def test(frames, results):
    print("request recieved. now responding.")
    results.put('1')

def extract_faces(image_list, num_workers=4):
    lock = Lock()
    faces = []

    def process_image(args):
        frame_idx, img_base64 = args
        try:
            img_bytes = base64.b64decode(img_base64)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

            if img is None:
                print(f"Unable to decode image {frame_idx}")
                return []

            detected_faces = face_app.get(img)
            frame_faces = []

            for face in detected_faces:
                bbox = [int(x) for x in face.bbox.astype(int).tolist()]
                dscore = face.det_score
                embedding = [float(x) for x in face.normed_embedding.tolist()]

                embedding_np = np.array(face.embedding)
                qscore = np.linalg.norm(embedding_np, ord=2)

                # Calculate the aspect ratio of the face frame
                height = bbox[3] - bbox[1]
                width = bbox[2] - bbox[0]
                aspect_ratio = height / width

                # Determine whether it is ortho face or side face
                face_type = "ortho" if 1 < aspect_ratio < 1.5 else "side"

                face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                _, buffer = cv2.imencode('.jpg', face_img)
                face_base64 = base64.b64encode(buffer).decode('utf-8')

                face_info = {
                    "frame_id": frame_idx,
                    "bounding_box": bbox,
                    "face_emb": embedding,
                    "cluster_id": -1,  # default cluster_id is -1
                    "extra_data": {
                        "face_type": face_type,
                        "face_base64": face_base64,
                        "face_detection_score": str(dscore),
                        "face_quality_score": str(qscore)
                    },
                }
                
                frame_faces.append(face_info)

            return frame_faces

        except Exception as e:
            print(f"Error accurs while processing {frame_idx}: {str(e)}")
            return []

    indexed_inputs = list(enumerate(image_list))

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        for frame_faces in tqdm(
            executor.map(process_image, indexed_inputs), total=len(image_list)
        ):
            faces.extend(frame_faces)

    return faces