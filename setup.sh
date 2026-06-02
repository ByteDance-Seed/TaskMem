#!/usr/bin/env bash
#
# Install the dependencies needed by the inference pipeline:
# face / audio processing, prompt construction, and the OpenAI client used
# by the Gemini / GPT API backends.
set -e

install_common_python_deps() {
    pip install moviepy
    pip install pydub
    pip install hdbscan
    pip install insightface
    pip install json-repair
    pip install openai
}

install_ffmpeg() {
    if command -v apt-get >/dev/null 2>&1; then
        sudo apt-get update
        sudo apt-get install -y ffmpeg
    fi
}

# Opt-in: build the Qwen3-Omni fork of vLLM. Only useful if you plan to
# wire `tools/chat_qwen_omni_vllm.py` into one of the *_model slots in
# `src/main.py`. Not exposed in the README or the example scripts.
install_qwen3_omni_vllm() {
    git clone -b qwen3_omni https://github.com/wangxiongts/vllm.git
    pushd vllm
    while true; do
        sudo pip uninstall -y vllm || true
        pip install -r requirements/build.txt
        pip install -r requirements/cuda.txt
        pip install git+https://github.com/huggingface/transformers@decde58eda17aec894f17b15e7a5cdf4bf82d46a
        export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f/vllm-0.9.2-cp38-abi3-manylinux1_x86_64.whl
        pip install accelerate
        pip install qwen-omni-utils -U
        pip install nvidia-nccl-cu12==2.27.3
        pip install -U flash-attn
        if pip install -e . -v; then
            pip uninstall -y pandas
            pip install pandas
            break
        else
            pip uninstall -y pandas
            pip install pandas
        fi
    done
    popd
}

install_common_python_deps
install_ffmpeg

if [ "${TASKMEM_INSTALL_QWEN3_OMNI:-0}" = "1" ]; then
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        install_qwen3_omni_vllm
    else
        echo "TASKMEM_INSTALL_QWEN3_OMNI=1 requested but no CUDA toolchain found; skipping vLLM build." >&2
    fi
fi
