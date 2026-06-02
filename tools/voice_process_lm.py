import io
import os
import re
import json
import time
import base64
import struct
import random
import tempfile
import importlib
import traceback
import numpy as np
from pydub import AudioSegment
from collections import defaultdict
from json_repair import repair_json, loads

MAX_RETRIES = 5
MIN_DURATION_FOR_AUDIO = 1.5

# Pluggable speaker-embedding backend used for cross-clip speaker re-identification.
#
# By default speaker embedding is DISABLED. In that case `voice_match` still
# performs face<->voice matching for the current clip via the VL model, but
# unknown speakers in the current clip will NOT be re-identified against
# previously-seen speakers from the long-term memory.
#
# To enable, set `TASKMEM_AUDIO_EMBED_BACKEND=your_pkg.your_module` and that
# module must expose:
#
#     def get_audio_embeddings(wav_b64_list: list[str]) -> list[list[float] | None]:
#         """Return a list of L2-normalised speaker embeddings, one per input
#         (or None for failures). Each input is a base64-encoded wav blob."""
_AUDIO_EMBED_BACKEND = os.environ.get("TASKMEM_AUDIO_EMBED_BACKEND", "")
if _AUDIO_EMBED_BACKEND:
    _get_audio_embeddings = importlib.import_module(_AUDIO_EMBED_BACKEND).get_audio_embeddings
else:
    _get_audio_embeddings = None

prompt_audio_segmentation = """You are given a video with a total duration of {} seconds. Your task is to perform Automatic Speech Recognition (ASR) and audio diarization on the provided video. Extract all speech segments with accurate timestamps and segment them by speaker turns (i.e., different speakers should have separate segments), but without assigning speaker identifiers.

Return a JSON list where each entry represents a speech segment with the following fields:
- start_time: Start time in seconds, represented as a floating-point number, accurate to 0.1s.
- end_time: End time in seconds, represented as a floating-point number, accurate to 0.1s.
- asr: The transcribed text for that segment.

Example Output:
```json
[
    {{"start_time": 5.3, "end_time": 6.9, "asr": "Hello, everyone."}},
    {{"start_time": 9.2, "end_time": 11.6, "asr": "Welcome to the meeting."}}
]
```

Strict Requirements:

- Ensure precise speech segmentation with accurate timestamps.
- Adjacent sentences need to be separated. Each list item can only have one sentence.
- Preserve punctuation and capitalization in the ASR output.
- Skip the speeches that can hardly be clearly recognized or extremely SHORT in time.
- Return only the valid JSON list (which starts with "[" and ends with "]") without additional explanations.
- If the video contains no speech, return an empty list ("[]").
	
Now generate the JSON list based on the given video:"""

def normalize_embedding(embedding):
    """Normalize embedding to unit length.

    Accepts either a raw bytes blob (legacy backend) or a python sequence of floats.
    """
    if isinstance(embedding, (bytes, bytearray)):
        format_string = 'f' * (len(embedding) // struct.calcsize('f'))
        emb = np.array(struct.unpack(format_string, embedding))
    else:
        emb = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(emb)
    return (emb / norm).tolist() if norm > 0 else emb.tolist()

def diarize_audio(clip_path, duration, f, context):
    inputs = [
        {
            "type": "video",
            "video": clip_path,
        },
        {
            "type": "text",
            "text": prompt_audio_segmentation.format(duration),
        },
    ]
    messages = context.asr_generate_messages(inputs)
    asrs = None
    for _ in range(MAX_RETRIES):
        res = None
        try:
            res = context.asr_get_response(messages)
            res = res.split("</think>")[-1].strip().strip("```json").strip("```python").strip("```").strip()
            asrs = loads(repair_json(res))
            new_asrs = []
            for asr in asrs:
                assert "asr" in asr
                asr["start_time"] = max(asr["start_time"], 0)
                asr["end_time"] = min(asr["end_time"], duration)
                asr["duration"] = asr["end_time"] - asr["start_time"]
                new_asrs.append(asr)
            f.write(json.dumps(asrs, ensure_ascii=False, indent=4) + "\n")
            break
        except Exception:
            time.sleep(5)
            if res is not None:
                f.write(res + '\n')
            traceback.print_exc(file=f)
    if asrs is None:
        raise Exception("Failed to diarize audio")
    return new_asrs

def get_audio_segments(audio, asrs):
    for asr in asrs:
        start_msec = asr["start_time"] * 1000
        end_msec = asr["end_time"] * 1000
        segment = audio[start_msec:end_msec]
        with io.BytesIO() as segment_buffer:
            segment.export(segment_buffer, format='wav')
            segment_buffer.seek(0)
            asr["audio_segment"] = base64.b64encode(segment_buffer.getvalue()).decode("utf-8")
    return asrs

def get_normed_audio_embeddings(audios):
    """Fill in `embedding` on each audio dict using the configured backend.

    If no backend is configured (`TASKMEM_AUDIO_EMBED_BACKEND` unset),
    every embedding is set to None and downstream cross-clip speaker
    re-identification is skipped.
    """
    if _get_audio_embeddings is None:
        for audio in audios:
            audio["embedding"] = None
        return
    audio_segments = [audio["audio_segment"] for audio in audios]
    normed_embeddings = [None] * len(audios)
    for _ in range(MAX_RETRIES):
        try:
            raw_embeddings = _get_audio_embeddings(audio_segments)
            normed_embeddings = [
                normalize_embedding(e) if e is not None else None
                for e in raw_embeddings
            ]
            break
        except Exception:
            normed_embeddings = [None] * len(audios)
    for audio, embedding in zip(audios, normed_embeddings):
        audio["embedding"] = embedding

def process_voices(clip, args, f, context):
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f".mp4") as temp_video:
        clip.write_videofile(temp_video.name, logger=None, threads=4)
        asrs = diarize_audio(temp_video.name, clip.duration, f, context)
    return asrs

    # voice_ids = set()
    # for voice in voices:
    #     q_embs = [voice["embedding"]]
    #     sim_voice = memory.semantic.search_node(q_embs, typ="voice")
    #     if len(sim_voice) > 0:
    #         # only choose the most similar voice, update voice information
    #         voice_id = sim_voice[0][0]
    #         voice["cluster_id"] = voice_id
    #         if voice["duration"] > memory.semantic.voices[voice_id].voice_info["duration"]:
    #             memory.semantic.voices[voice_id].voice_info = {
    #                 "asr": voice["asr"],
    #                 "duration": voice["duration"],
    #                 "voice_emb": voice["embedding"],
    #                 "voice_base64": voice["audio_segment"]
    #             }
    #         voice_ids.add(voice_id)
    #         memory.semantic.voices[voice_id].embs.extend(q_embs)
    #         if len(memory.semantic.voices[voice_id].embs) > memory.semantic.voices[voice_id].max_embs:
    #             random.shuffle(memory.semantic.voices[voice_id].embs)
    #             memory.semantic.voices[voice_id].embs = memory.semantic.voices[voice_id].embs[:memory.semantic.voices[voice_id].max_embs]
    #     else:
    #         voice_info = {
    #             "asr": voice["asr"],
    #             "duration": voice["duration"],
    #             "voice_emb": voice["embedding"],
    #             "voice_base64": voice["audio_segment"]
    #         }
    #         voice_node = VoiceNode(memory.semantic.get_node_id(), clip_id, q_embs, voice_info)
    #         memory.semantic.voices[voice_node.id] = voice_node
    #         voice["cluster_id"] = voice_node.id
    # return short_asrs + voices, list(voice_ids)

def voice_tools(clip, voice_start, args, f, context):
    new_clip = clip.subclipped(voice_start, clip.duration)
    voices = process_voices(new_clip, args, f, context) # Assign each voice a unique ID
    return voices

prompt_audio_match = """You are given a video. Your task is to match the subtitle with the <face_id of> its speaker.
The subtitle to be matched is given in the following JSON list:
```json
{}
```

The returned list must have the same length as the input JSON list. 
Each item in the list shall include an additional string field named "speaker", with the value determined as follows:
- If the corresponding subtitle is definitively associated with a <face_id>, set "speaker" to that <face_id>;
- Otherwise, set "speaker" to the string literal "unknown".

Example Output:
```json
[
    {{"start_time": 5.3, "end_time": 6.9, "asr": "Hello, everyone.", "speaker": "<face_1>"}},
    {{"start_time": 9.2, "end_time": 11.6, "asr": "Welcome to the meeting.", "speaker": "unknown"}}
]
```

Now generate the JSON list based on the given video:"""

def voice_match(clip, voices, global2local, context, f, memory, args):
    have_matched, tobe_matched = [], []
    for voice in voices:
        if "speaker" in voice:
            have_matched.append(voice)
        else:
            del voice["duration"]
            tobe_matched.append(voice)
    
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f".mp4") as temp_video:
        clip.write_videofile(temp_video.name, logger=None, threads=4)
        inputs = [
            {
                "type": "video",
                "video": temp_video.name,
            },
            {
                "type": "text",
                "text": prompt_audio_match.format(json.dumps(tobe_matched, indent=4, ensure_ascii=False)),
            },
        ]
        messages = context.voice_generate_messages(inputs)
        face_ids = {v: k for k, v in global2local.items()}
        face_ids["unknown"] = "unknown"

        for _ in range(MAX_RETRIES):
            res = None
            try:
                res = context.voice_get_response(messages)
                res = res.split("</think>")[-1].strip().strip("`jsonpyth").strip()
                tobe_matched = loads(repair_json(res))
                finish = True
                for voice in tobe_matched:
                    assert "start_time" in voice and "end_time" in voice and "asr" in voice
                    if "speaker" not in voice:
                        finish = False
                        voice["speaker"] = "unknown"
                    if voice["speaker"] not in face_ids:
                        finish = False
                        voice["speaker"] = "unknown"
                    voice["speaker"] = face_ids[voice["speaker"]] 
                if finish:
                    break
            except Exception:
                time.sleep(5)
                f.write("face_ids" + json.dumps(face_ids) + '\n')
                if res is not None:
                    f.write(res + '\n')
                traceback.print_exc(file=f)

    if args.process_voice:
        # save the embedding of the voice of a known speaker, assign unknown speaker's voice based on embedding
        with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f".wav") as temp_audio:
            assert clip.audio is not None, "clip.audio can't be None!"
            clip.audio.write_audiofile(temp_audio.name, fps=16000, logger=None)
            audio = AudioSegment.from_wav(temp_audio.name)
            audios = get_audio_segments(audio, tobe_matched)
        get_normed_audio_embeddings(audios)
        voice_dic = defaultdict(list)
        for audio in audios:
            if "speaker" not in audio:
                audio["speaker"] = "unknown"
            voice_dic[audio["speaker"]].append(audio)
        for speaker, voices in voice_dic.items():
            if speaker == "unknown":
                for voice in voices:
                    if voice["embedding"] is None:
                        continue
                    q_embs = [voice["embedding"]]
                    sim_voice = memory.semantic.search_node(q_embs, "voice")
                    if len(sim_voice) > 0:
                        voice["speaker"] = f"[face_{sim_voice[0][0]}]"
                        if voice["speaker"] not in global2local:
                            local_id = "<face_{}>".format(len(global2local) + 1)
                            global2local[voice["speaker"]] = local_id
                        f.write("[assign] {} to {}\n".format(voice["speaker"], voice["asr"]))
            else:
                face_id = int(re.search(r"\d+", speaker).group())
                memory.semantic.add_node(face_id, [{
                    "embedding": voice["embedding"],
                    "base64": voice["audio_segment"]
                } for voice in voices if voice["embedding"] is not None], "voice")
    
    new_tobe_matched = []
    for voice in tobe_matched:
        try:
            x = {
                "start_time": voice["start_time"],
                "end_time": voice["end_time"],
                "asr": voice["asr"],
                "duration": voice["end_time"] - voice["start_time"],
                "speaker": voice["speaker"],
            }
            new_tobe_matched.append(x)
        except:
            pass
    
    return have_matched + new_tobe_matched