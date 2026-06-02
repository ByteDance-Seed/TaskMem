import re
import json
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

memory_config = {
    "face": {
        "max_embedding_num": 15,
        "prune_embedding_num": 10,
        "threshold": 0.3,
    },
    "voice": {
        "max_embedding_num": 25,
        "prune_embedding_num": 20,
        "threshold": 0.7
    }
}

class Character:
    def __init__(self, node_id, face=[], voice=[]):
        self.id = node_id
        self.features = {
            "face": face,
            "voice": voice
        }
    
    def get_similarity(self, q_embs, typ):
        embs = [feature["embedding"] for feature in self.features[typ]]
        q_embs = np.array(q_embs)
        embs = np.array(embs)
        assert embs.shape[-1] == q_embs.shape[-1], f"q_embs: {q_embs.shape}, embs: {embs.shape}"
        sims = cosine_similarity(q_embs, embs)
        mean_sim = np.mean(sims)
        return mean_sim

    def __str__(self):
        return "[face_{}]".format(self.id)

def get_redundancy(embs):
    norm_embs = np.linalg.norm(embs, axis=-1, keepdims=True)
    norm_embs = np.where(norm_embs == 0, 1e-10, norm_embs)
    similarity = np.matmul(embs, embs.T) / (norm_embs * norm_embs.T)
    similarity = np.clip(similarity, -1.0, 1.0)
    similarity = np.mean(similarity, axis=-1)
    sort_similarity = []
    for i, sim in enumerate(similarity):
        sort_similarity.append([i, sim])
    sort_similarity = sorted(sort_similarity, key=lambda x: x[1])
    return sort_similarity

class Semantic:
    def __init__(self):
        self.node_id = 0
        self.characters = {} # id -> Character
        self.knowledge = {}
    
    def __str__(self):
        return json.dumps(self.knowledge, indent=4, ensure_ascii=False)
    
    def get_node_id(self):
        self.node_id += 1
        return self.node_id
    
    def search_node(self, embs, typ):
        results = []
        for id, character in self.characters.items():
            if len(character.features[typ]) == 0:
                continue
            try:
                sim = character.get_similarity(embs, typ)
            except:
                print("q_embs", embs)
                print("embs", character.features[typ])
                sim = 0
            if sim > memory_config[typ]["threshold"]:
                results.append([id, sim])
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def add_node(self, node_id, nodes, typ):
        self.characters[node_id].features[typ].extend(nodes)
        if len(self.characters[node_id].features[typ]) > memory_config[typ]["max_embedding_num"]:
            self.prune_node(node_id, typ)
    
    def prune_node(self, node_id, typ):
        q_embs = []
        for feature in self.characters[node_id].features[typ]:
            q_embs.append(feature["embedding"])
        q_embs = np.array(q_embs)
        redundancy = get_redundancy(q_embs)
        redundancy_id = [x[0] for x in redundancy[:memory_config[typ]["prune_embedding_num"]]]
        features = [feature for i, feature in enumerate(self.characters[node_id].features[typ]) if i in redundancy_id]
        self.characters[node_id].features[typ] = features

class Episodic:
    def __init__(self):
        self.description = {}
    
    def __str__(self):
        return json.dumps(self.description, indent=4, ensure_ascii=False)

class LongTermMemory:
    def __init__(self, interval):
        self.interval = interval
        self.semantic = Semantic()
        self.episodic = Episodic()

    def __str__(self):
        return "Episodic Memory:\n{}\nSemantic Memory:\n{}".format(str(self.episodic), str(self.semantic))

    def format_second(self, seconds):
        hours = seconds // 3600
        seconds %= 3600
        minutes = seconds // 60
        seconds %= 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def to_string(self, memory_type="both", clip_ids=None):
        if clip_ids is None:
            clip_ids = [str(i) for i in range(len(self.semantic.knowledge))]
        des, kno = [], []
        for clip_id in clip_ids:
            des.append("{}-{}\n{}".format(self.format_second(int(clip_id) * self.interval), self.format_second((int(clip_id) + 1) * self.interval), self.episodic.description[clip_id]))
            kno.extend(self.semantic.knowledge[clip_id])
        if memory_type == "episodic":
            return "Description:\n\n{}".format("\n".join(kno))
        elif memory_type == "semantic":
            return "Knowledge:\n{}".format("\n".join(kno))
        else:
            return "Description:\n\n{}\n\nKnowledge:\n{}".format("\n\n".join(des), "\n".join(kno))

    def update_memory(self, clip_id, semantic_memory, episodic_memory):
        if episodic_memory is not None:
            self.episodic.description[clip_id] = episodic_memory
        if semantic_memory is not None:
            self.semantic.knowledge[clip_id] = semantic_memory
