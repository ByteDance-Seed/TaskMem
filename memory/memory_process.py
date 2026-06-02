import json
import time
import traceback
from json_repair import repair_json, loads
RETRY_TIMES = 10

system_prompt = """You are given the following content:
- A video with corresponding faces (presented via bounding boxes) and subtitles.
- [Description of the preceding part] of the video. This field can be empty.
"""

episodic_prompt = """Using the provided face IDs, write a detailed and cohesive description of the given video. The description should capture the complete set of observable and inferable events in the video. Your output should incorporate the following categories (but is not limited to these):

1. Characters' Appearance: Describe the characters' appearance, such as their clothing, facial features, or any distinguishing characteristics.
2. Characters' Actions & Movements: Describe specific gestures, movements, or interactions performed by the characters.
3. Characters' Spoken Dialogue: Transcribe or summarize what is spoken by the characters.
4. Characters' Contextual Behavior: Describe the characters' roles in the scene or their interaction with other characters, focusing on their behavior, emotional state, or relationships.

Strict Requirements:
- If a character has an associated face ID in the video, refer to them ONLY using that face ID.
- If characters DO NOT have associated face IDs in the whole video, it's ok not to describe them.
- A character may have multiple face IDs, and the ID currently displayed on the screen should be used for description.
- Ensure the continuity and uniformity of content between adjacent descriptions.
- Directly describe the video content, DO NOT start with 'The video ...'.
- If the video has an incomplete ending plot, the last line is truncated or asr is not a complete sentence, DO NOT describe it.
- The final output must be a dictionary, with the key being "description".{}

Output format:
```json
{{
    "description": "<face_1> is standing outside under a blue sky with clouds. <face_1> gets out of the car and says: \\"Hello everyone, welcome to my channel\\"."
}}
```

{}

- Generate subsequent descriptions not covered in [Description of the preceding part], maintain coherence with it, and avoid any repetition of similar information.
- If [Description of the preceding part] is empty, describe the video from scratch.
- Generate the description briefly in one or two sentences.
Please output the description."""

def construct_episodic_input(memories, supplement_prompt):
    return [
        {
            "type": "text",
            "text": system_prompt,
        },
        {
            "type": "video",
            "video": memories["clip"],
        },
        {
            "type": "text",
            "text": "\n\n".join([
                episodic_prompt.format(supplement_prompt, "[Description of the preceding part]:\n" + " ".join(memories["episodic"]))
            ])
        }
    ]

def generate_episodic(context, args, output_file):
    inputs = construct_episodic_input(context.memories, args.supplement_prompt)
    messages = context.generate_messages_episodic(inputs)
    output_file.write("-----Context:\n" + inputs[0]["text"] + '\n' + inputs[2]["text"] + '\n-----End of context.\n\n')
    generate_result = ""
    episodic_io = {"response": None}
    for _ in range(RETRY_TIMES):
        res = None
        try:
            res = context.get_response_episodic(messages)
            generate_result = res.split("</think>")[-1].strip().strip("`json").strip()
            generate_result = loads(repair_json(generate_result))
            generate_result = generate_result["description"]
            episodic_io["response"] = res
            break
        except Exception:
            time.sleep(15)
            if res is not None:
                output_file.write(res + '\n')
            traceback.print_exc(file=output_file)
    episodic_io["input"] = inputs
    episodic_io["output"] = generate_result
    return generate_result, episodic_io

semantic_prompt = """Using the provided face IDs, generate a list of high-level reasoned knowledge about the video. Focus less on trivial events, surface-level observations and specific details. The knowledge should be detailed and self-contained.

Strict Requirements:
- If a character has an associated face ID in the video, refer to them ONLY using that face ID.
- If characters DO NOT have associated face IDs in the whole video, it's ok not to describe them.
- A character may have multiple face IDs, and the ID currently displayed on the screen should be used for description.
- Provide only the final high-level thinking knowledge, without detailing the reasoning process or restating simple observations from the video.
- If there is already SIMILAR knowledge in [Supplementary features], DO NOT generate it again.
- The final output must be a list of strings, with each string representing exactly one ATOMIC KNOWLEDGE.{}

Generate knowledge across, but not limited to, the following categories.
1. character-level attributes such as: name, role, personality, and interests ...
2. interpersonal relationships and interactions between characters.
3. general knowledge such as: norms, real-world knowledge, and common-sense ...

Output format:
```json
[
    "<face_1>'s name is Alice.",
    "<face_2> is a teacher."
]
```

{}

- DO NOT simply repeat the subtitle.
- DO NOT regenerate the similar information that has appeared in [Supplementary features].
Please output the knowledge."""

def construct_semantic_input(memories, supplement_prompt):
    return [
        {
            "type": "text",
            "text": system_prompt + "- [Supplementary features] of the faces retrieved from previous long-term memory. For a newcomer, this field can be empty.\n",
        },
        {
            "type": "video",
            "video": memories["clip"],
        },
        {
            "type": "text",
            "text": "\n\n".join([
                "[Description of the preceding part]:\n" + " ".join(memories["episodic"]),
                semantic_prompt.format(supplement_prompt, "[Supplementary features]:\n" + json.dumps(memories["semantic"], indent=4, ensure_ascii=False))
            ])
        }
    ]

def generate_semantic(context, args, output_file):
    inputs = construct_semantic_input(context.memories, args.supplement_prompt)
    messages = context.generate_messages_semantic(inputs)
    output_file.write("-----Context:\n" + inputs[0]["text"] + '\n' + inputs[2]["text"] + '\n-----End of context.\n\n')
    generate_result = []
    semantic_io = {"response": None}
    for _ in range(RETRY_TIMES):
        res = None
        try:
            res = context.get_response_semantic(messages)
            generate_result = res.split("</think>")[-1].strip().strip("`json").strip()
            generate_result = loads(repair_json(generate_result))
            semantic_io["response"] = res
            break
        except Exception:
            time.sleep(15)
            if res is not None:
                output_file.write(res + '\n')
            traceback.print_exc(file=output_file)
    semantic_io["input"] = inputs
    semantic_io["output"] = generate_result
    return generate_result, semantic_io
