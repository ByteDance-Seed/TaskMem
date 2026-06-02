import tempfile

RETRY_TIMES = 5

def generate_memory(video, model, context):
    with tempfile.NamedTemporaryFile(dir="/tmp", suffix=f".mp4") as temp_video:
        video.write_videofile(temp_video.name, logger=None, threads=4)
        inputs = [
            {
                "type": "text",
                "text": "Please describe the given video.",
            },
            {
                "type": "video",
                "video": temp_video.name,
            }
        ]
        messages = context.generate_messages(inputs, "description")
        generate_result = ""
        for _ in range(RETRY_TIMES):
            try:
                generate_result = context.get_response(model, messages)
                break
            except:
                continue
        return generate_result