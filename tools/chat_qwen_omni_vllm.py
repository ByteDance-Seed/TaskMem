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
"""vLLM backend for Qwen3-Omni Thinker.

Note: requires a vLLM build with Qwen3-Omni support. Install via the
``setup.sh`` script or follow the upstream Qwen3-Omni vLLM instructions.
"""
import os
import logging

import torch
from transformers import Qwen3OmniMoeProcessor
from qwen_omni_utils import process_mm_info
from vllm import LLM, SamplingParams

os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_USE_V1', '0')

logger = logging.getLogger(__name__)

logging.getLogger("httpx").setLevel(logging.CRITICAL)
logging.getLogger("urllib3").setLevel(logging.CRITICAL)
logging.getLogger("httpcore").setLevel(logging.CRITICAL)


class ChatOmni:
    def __init__(self, model_path):
        self.processor = Qwen3OmniMoeProcessor.from_pretrained(model_path)
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
        """Get chat completion response from the loaded Qwen3-Omni model."""
        outputs = self.llm.generate([messages], sampling_params=self.sampling_params)
        return outputs[0].outputs[0].text

    def generate_messages(self, inputs):
        messages = [{"role": "user", "content": inputs}]
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        audios, images, videos = process_mm_info(messages, use_audio_in_video=True)
        inputs = {
            'prompt': text,
            'multi_modal_data': {},
            "mm_processor_kwargs": {
                "use_audio_in_video": True,
            },
        }
        if images is not None:
            inputs['multi_modal_data']['image'] = images
        if videos is not None:
            inputs['multi_modal_data']['video'] = videos
        if audios is not None:
            inputs['multi_modal_data']['audio'] = audios
        return inputs
