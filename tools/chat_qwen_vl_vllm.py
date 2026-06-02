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
"""vLLM backend for Qwen3-VL based models (including TaskMem checkpoints)."""
import os
import logging

import torch
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)


class ChatVL:
    def __init__(self, model_path):
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.llm = LLM(
            model=model_path,
            trust_remote_code=True,
            gpu_memory_utilization=0.95,
            tensor_parallel_size=torch.cuda.device_count(),
            limit_mm_per_prompt={'image': 3, 'video': 3, 'audio': 3},
            max_num_seqs=8,
            max_model_len=32768,
        )
        self.sampling_params = SamplingParams(
            temperature=0.6,
            top_p=0.95,
            top_k=20,
            max_tokens=32768,
        )

    def get_response(self, messages):
        """Get chat completion response from the loaded Qwen3-VL model."""
        outputs = self.llm.generate([messages], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_messages(self, inputs):
        messages = [{"role": "user", "content": inputs}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        # qwen_vl_utils 0.0.14+ required
        image_inputs, video_inputs, video_kwargs = process_vision_info(
            messages,
            image_patch_size=self.processor.image_processor.patch_size,
            return_video_kwargs=True,
            return_video_metadata=True,
        )

        mm_data = {}
        if image_inputs is not None:
            mm_data['image'] = image_inputs
        if video_inputs is not None:
            mm_data['video'] = video_inputs
        raw_prompt_ids = self.processor.tokenizer.encode(text, add_special_tokens=False)

        return {
            'prompt_token_ids': raw_prompt_ids,
            'multi_modal_data': mm_data,
            'mm_processor_kwargs': video_kwargs,
        }


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Run a Qwen3-VL TaskMem checkpoint over a validation JSONL.")
    parser.add_argument("--ckpt", type=str, required=True,
                        help="HuggingFace-style checkpoint folder loadable by AutoProcessor / vLLM.")
    parser.add_argument("--validation_jsonl", type=str, required=True,
                        help="Validation JSONL produced by the data preparation scripts.")
    parser.add_argument("--memory_tag_in", type=str, default="gemini_val",
                        help="Memory tag used for the input rollouts (read).")
    parser.add_argument("--memory_tag_out", type=str, default="qwen3_vl_val",
                        help="Memory tag used for the model output (write).")
    parser.add_argument("--output_root", type=str, required=True,
                        help="Root folder under which `validation_ori/<id>/<i>/` will be written.")
    parser.add_argument("--num_passes", type=int, default=10,
                        help="Number of rollout passes per video to evaluate.")
    parser.add_argument("--shard_index", type=int, default=0,
                        help="Shard index when sharding validation across multiple processes.")
    parser.add_argument("--num_shards", type=int, default=1,
                        help="Total number of shards.")
    parser.add_argument("--retry_times", type=int, default=5,
                        help="Number of retries per rollout when JSON parsing fails.")
    args = parser.parse_args()

    chatVL = ChatVL(args.ckpt)
    with open(args.validation_jsonl, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % args.num_shards != args.shard_index:
                continue
            data = json.loads(line)
            for i in range(args.num_passes):
                memory_in = os.path.join(
                    data["memory_path"], str(i), f"{data['id']}_{i}_{args.memory_tag_in}.json"
                )
                if not os.path.isfile(memory_in):
                    continue
                memory = json.load(open(memory_in))
                messages = chatVL.generate_messages(memory["semantic"]["input"])

                generate_result = []
                for _ in range(args.retry_times):
                    try:
                        res = chatVL.get_response(messages)
                        generate_result = json.loads(
                            res.split("</think>")[-1].strip().strip("`json").strip()
                        )
                        break
                    except Exception:
                        continue

                memory["output_qwen3"] = generate_result
                out_dir = os.path.join(args.output_root, data["id"], str(i))
                os.makedirs(out_dir, exist_ok=True)
                json.dump(
                    memory,
                    open(os.path.join(out_dir, f"{data['id']}_{i}_{args.memory_tag_out}.json"), "w"),
                    ensure_ascii=False,
                    indent=4,
                )
