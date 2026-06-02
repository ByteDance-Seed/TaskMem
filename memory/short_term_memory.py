import os
import re
import json
import shutil
from functools import partial

def unify_face_id(text):
    pattern = r'([<\(]?)(face)([\s_]*)(\d+)([>\)]?)'
    replaced_text = re.sub(
        pattern,
        lambda match: f"<{match.group(2).lower()}_{match.group(4)}>",
        text,
        flags=re.IGNORECASE,
    )
    return replaced_text

class ClipInfo:
    def __init__(self, faces=[], voices=[], description=[], knowledge=[], global_id=[]):
        self.faces = faces
        self.voices = voices
        self.description = description
        self.knowledge = knowledge
        self.global_id = global_id # global id used in memory generation
        # long-term semantic
        self.face_info = None
        self.voice_info = None

class ShortTermMemory:
    def __init__(self, args):
        self.global2local = {}
        self.clip_info = ClipInfo() # global id
        self.memories = {} # local id `<>`
        if args.generate_episodic:
            if args.episodic_model == "qwen3_omni_vllm":
                from tools.chat_qwen_omni_vllm import ChatOmni
                chatOmni = ChatOmni(args.episodic_model_path)
                self.get_response_episodic = chatOmni.get_response
                self.generate_messages_episodic = chatOmni.generate_messages
            elif args.episodic_model == "qwen3_vl_vllm":
                from tools.chat_qwen_vl_vllm import ChatVL
                chatVL = ChatVL(args.episodic_model_path)
                self.get_response_episodic = chatVL.get_response
                self.generate_messages_episodic = chatVL.generate_messages
            elif args.episodic_model.startswith("gemini") or args.episodic_model.startswith("gpt"):
                from tools.chat_openai import get_response, generate_messages, get_response_with_vision_limit_retry
                self.get_response_episodic = partial(get_response_with_vision_limit_retry, args.episodic_model)
                self.generate_messages_episodic = partial(generate_messages, model_name=args.episodic_model)
            else:
                raise ValueError("Invalid model name: {}".format(args.episodic_model))

        if args.generate_semantic:
            if args.semantic_model == "qwen3_omni_vllm":
                from tools.chat_qwen_omni_vllm import ChatOmni
                chatOmni = ChatOmni(args.semantic_model_path)
                self.get_response_semantic = chatOmni.get_response
                self.generate_messages_semantic = chatOmni.generate_messages
            elif args.semantic_model == "qwen3_vl_vllm":
                from tools.chat_qwen_vl_vllm import ChatVL
                chatVL = ChatVL(args.semantic_model_path)
                self.get_response_semantic = chatVL.get_response
                self.generate_messages_semantic = chatVL.generate_messages
            elif args.semantic_model.startswith("gemini") or args.semantic_model.startswith("gpt"):
                from tools.chat_openai import get_response, generate_messages, get_response_with_vision_limit_retry
                self.get_response_semantic = partial(get_response_with_vision_limit_retry, args.semantic_model)
                self.generate_messages_semantic = partial(generate_messages, model_name=args.semantic_model)
            else:
                raise ValueError("Invalid model name: {}".format(args.semantic_model))
        
        if args.process_audio:
            if args.asr_model == "qwen3_omni_vllm":
                from tools.chat_qwen_omni_vllm import ChatOmni
                chatOmni = ChatOmni(args.asr_model_path)
                self.asr_get_response = chatOmni.get_response
                self.asr_generate_messages = chatOmni.generate_messages
            elif args.asr_model.startswith("gemini") or args.asr_model.startswith("gpt"):
                from tools.chat_openai import get_response, generate_messages
                self.asr_get_response = partial(get_response, args.asr_model)
                self.asr_generate_messages = partial(generate_messages, model_name=args.asr_model)
            else:
                raise ValueError("Invalid model name: {}".format(args.asr_model))
        
        if args.process_video:
            if args.voice_model == "qwen3_omni_vllm":
                from tools.chat_qwen_omni_vllm import ChatOmni
                chatOmni = ChatOmni(args.voice_model_path)
                self.voice_get_response = chatOmni.get_response
                self.voice_generate_messages = chatOmni.generate_messages
            elif args.voice_model.startswith("gemini") or args.voice_model.startswith("gpt"):
                from tools.chat_openai import get_response, generate_messages
                self.voice_get_response = partial(get_response, args.voice_model)
                self.voice_generate_messages = partial(generate_messages, model_name=args.voice_model)
            else:
                raise ValueError("Invalid model name: {}".format(args.voice_model))

    def add_clip(self, clip_id, clip, truncate, new_duration, args, memory, f, file_id, add_clip_id=False):
        
        def truncate_voice(voices, voice_start):
            new_voices = []
            for voice in voices:
                if voice["end_time"] <= voice_start:
                    new_voices.append(voice)
            return new_voices

        # prepare audio
        new_voices = []
        truncate_duration = args.interval if truncate else 0
        for voice in self.clip_info.voices:
            if voice["end_time"] > truncate_duration:
                voice["end_time"] -= truncate_duration
                voice["start_time"] = max(0, voice["start_time"] - truncate_duration)
                new_voices.append(voice)
        new_voices.sort(key=lambda x: x["start_time"])
        if args.process_audio:
            voice_file = os.path.join(args.video_folder, "{}_voice.json".format(file_id))
            save_voices = json.load(open(voice_file)) if os.path.exists(voice_file) else {}
            voice_start = clip.duration - new_duration
            if len(new_voices) != 0:
                vad = [[new_voices[0]["start_time"], new_voices[0]["end_time"]]]
                for voice in new_voices[1:]:
                    if voice["start_time"] < vad[-1][1]: # Not merge the adjacent voice
                        vad[-1][1] = max(voice["end_time"], vad[-1][1])
                    else:
                        vad.append([voice["start_time"], voice["end_time"]])
                vad.sort(key=lambda x: x[1], reverse=True)
                # At most borrow one
                if len(vad) > 0 and voice_start - vad[0][1] < 1:
                    if len(vad) > 1:
                        voice_start = vad[1][1]
                    else:
                        voice_start = vad[0][0]
            from tools.voice_process_lm import voice_tools
            voices = voice_tools(clip, voice_start, args, f, self)
            new_voices = truncate_voice(new_voices, voice_start)
            save_voices[clip_id] = {
                "voice_start": voice_start,
                "voices": voices
            }
            json.dump(save_voices, open(voice_file, "w"), ensure_ascii=False, indent=4)
        elif not args.process_audio and args.process_video:
            voice_file = os.path.join(args.video_folder, "{}_voice.json".format(file_id))
            save_voices = json.load(open(voice_file)) if os.path.exists(voice_file) else {}
            if clip_id in save_voices:
                voice_start, voices = save_voices[clip_id]["voice_start"], save_voices[clip_id]["voices"]
            else:
                voice_start, voices = 0, []
            new_voices = truncate_voice(new_voices, voice_start)
        else:
            voice_start, voices = 0, [] # unnecessary
        for voice in voices:
            voice["start_time"] += voice_start
            voice["end_time"] += voice_start
            new_voices.append(voice)

        # prepare video
        if args.process_video:
            map_file = os.path.join(args.video_folder, "{}_map.json".format(file_id))
            save_map = json.load(open(map_file)) if os.path.exists(map_file) else {}
            from tools.face_process import face_tools
            faces, _ = face_tools(clip, self.clip_info.faces, truncate, new_duration, args, memory)
            # generate global id <--> local id, process faces for drawing box
            cluster_ids = set()
            for face in faces:
                if face["cluster_id"] != -1:
                    cluster_ids.add(("face", face["cluster_id"]))
            self.global2local = {"[{}_{}]".format(cluster_id[0], cluster_id[1]): "<{}_{}>".format(cluster_id[0], i + 1) for i, cluster_id in enumerate(cluster_ids)}
            for voice in new_voices:
                if "speaker" in voice and voice["speaker"] != "unknown":
                    if voice["speaker"] not in self.global2local:
                        local_id = "<face_{}>".format(len(self.global2local) + 1)
                        self.global2local[voice["speaker"]] = local_id
            f.write("Clip id: {}\nMap global2local: {}\n\n".format(clip_id, self.global2local))
            save_map[clip_id] = self.global2local
            # draw faces and voices on the video with local id, close after generate base64
            if args.asr_only:
                for voice in new_voices:
                    voice["speaker"] = "unknown"
            tobe_matched = [voice for voice in new_voices if "speaker" not in voice]
            from tools.video_process import draw_boxes
            if len(tobe_matched) > 0:
                from tools.voice_process_lm import voice_match
                new_clips = draw_boxes(clip, faces, new_voices, args.fps, self.global2local)
                new_voices = voice_match(new_clips, new_voices, self.global2local, self, f, memory, args)
                new_clips.close()
            new_clips = draw_boxes(clip, faces, new_voices, args.fps, self.global2local)
            video_file = "/tmp/{}_{}.mp4".format(file_id, clip_id)
            new_clips.write_videofile(video_file, logger=None, threads=4)
            save_path = os.path.join(args.video_folder, clip_id, "{}_{}.mp4".format(file_id, clip_id))
            shutil.copy2(video_file, save_path)
            self.memories["clip"] = save_path
            json.dump(save_map, open(map_file, "w"), ensure_ascii=False, indent=4)
            save_path = os.path.join(args.video_folder, clip_id, "{}_{}_new.mp4".format(file_id, clip_id))
            extra_data = json.loads(args.extra_data)
            if "train" in extra_data and extra_data["train"]:
                with new_clips.subclipped(new_clips.duration - new_duration, new_clips.duration) as clip_for_eval_episodic:
                    video_file = "/tmp/{}_{}_new.mp4".format(file_id, clip_id)
                    clip_for_eval_episodic.write_videofile(video_file, logger=None, threads=4)
                    shutil.copy2(video_file, save_path)
            new_clips.close()
            if not args.generate_episodic and not args.generate_semantic:
                global_id = self.clip_info.global_id[1:] if truncate else self.clip_info.global_id
                self.clip_info = ClipInfo(faces, new_voices)
                return
        else:
            map_file = os.path.join(args.video_folder, "{}_map.json".format(file_id))
            save_map = json.load(open(map_file)) if os.path.exists(map_file) else {}
            if not args.generate_episodic and not args.generate_semantic:
                self.clip_info = ClipInfo([], new_voices)
                return
            else:
                faces = []
                self.memories["clip"] = os.path.join(args.video_folder, clip_id, "{}_{}.mp4".format(file_id, clip_id))
                self.global2local = save_map.get(clip_id, [])

        # prepare history memory
        description = self.clip_info.description[1:] if truncate else self.clip_info.description
        if args.generate_semantic:
            knowledge = self.clip_info.knowledge[1:] if truncate else self.clip_info.knowledge
        else:
            knowledge = None
        global_id = self.clip_info.global_id[1:] if truncate else self.clip_info.global_id
        self.clip_info = ClipInfo(faces, new_voices, description, knowledge, global_id)

        for global_ids in self.clip_info.global_id:
            for global_id in global_ids:
                if global_id not in self.global2local:
                    match = re.search(r"\[([a-zA-Z]+)_(\d+)\]", global_id)
                    self.global2local[global_id] = "<{}_{}>".format(match.group(1), len(self.global2local) + 1)

        # map clip_info.knowledge and clip_info.description into local id
        episodic_list = []
        for des in self.clip_info.description:
            for glo, loc in self.global2local.items():
                des = des.replace(glo, loc)
            episodic_list.append(des)
        self.memories["episodic"] = episodic_list
        
        if args.generate_semantic:
            semantic_list = sum(self.clip_info.knowledge, [])
            for i in range(len(semantic_list)):
                for glo, loc in self.global2local.items():
                    semantic_list[i] = str(semantic_list[i]).replace(glo, loc)
            self.memories["semantic"] = semantic_list

        # TODO: search long-term memory and map their ids into local id, considering equivalence
    
    def update_episodic(self, episodic_memory):
        # add to memories; map to origin id and add to clip_info
        try:
            episodic_memory = unify_face_id(episodic_memory)
        except:
            pass
        self.memories["episodic"].append(episodic_memory)
        global_id = set()
        for glo, loc in self.global2local.items():
            if loc in episodic_memory:
                episodic_memory = episodic_memory.replace(loc, glo)
                global_id.add(glo)
        self.clip_info.description.append(episodic_memory)
        self.clip_info.global_id.append(global_id)
        return self.clip_info.description[-1]

    def update_semantic(self, semantic_memory):
        # delete equivalence; map to origin id and add to clip_info
        semantic, global_id = [], set()
        for memory in semantic_memory:
            try:
                memory = unify_face_id(memory)
            except:
                pass
            for glo, loc in self.global2local.items():
                if loc in memory:
                    memory = memory.replace(loc, glo)
                    global_id.add(glo)
            semantic.append(memory)
        self.clip_info.knowledge.append(semantic)
        self.clip_info.global_id.append(global_id)
        return self.clip_info.knowledge[-1]