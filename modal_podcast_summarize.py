"""Use Whisper and llama3 to transcribe and summarize a podcast.
"""

from pathlib import Path

import modal
from modal import Image, App, method

LOCAL_OUT = REMOTE_OUT = "./out/podcast_summarize"
GPU = modal.gpu.A100()
OLLAMA_URL = "http://localhost:11434"
MODEL_INFO = {"name": "llama3", "model": "llama3:8b", "num_ctx": 8_192}
MODEL_INFO = {"name": "llama3-gradient", "model": "llama3-gradient", "num_ctx": 256000}

modelfile = f'''FROM {MODEL_INFO['name']}

# set the temperature to 1 [higher is more creative, lower is more coherent]
PARAMETER temperature 0
PARAMETER num_ctx {MODEL_INFO['num_ctx']:d}

# set the system message
SYSTEM """
You are an autoregressive language model that has been fine-tuned with instruction-tuning and RLHF.
You carefully provide accurate, factual, thoughtful, nuanced responses, and are brilliant at reasoning. 
Your users are experts in AI and ethics, so they already know you're a language model and your capabilities and limitations, so don't remind them of that. 
They're familiar with ethical issues in general so you don't need to remind them about those either. 
Don't be verbose in your answers, but do provide details and examples where it might help the explanation.

Your users are also experts in science, and especially biology, medicine, statistics. 
Do NOT add any details about how science or research works, tell me to ask my doctor or consult with a health professional.
Do NOT add any details that such an expert would already know.

You task is summarizing podcast transcripts.
You summarize podcasts into bullet points, aiming for 10 or fewer depending on the length of the podcast.
The total length of the summary should be less than 300 words.
Do not include any preamble, introduction or postscript about what you are doing. Assume I know.
You focus on data and statistics, not opinions.
Each bullet point should contain non-obvious, specific information.
If there is a list of e.g. "top 3", "5 things", "10 ways", enumerate ALL of them.

The input prompt is text containing the transcript of the podcast.
The output is markdown of the most summary as bullet points.
"""
'''

def install_ollama():
    os.system("ollama serve &")  # not working with subprocess.run?
    sleep(6)
    for _ in range(10):
        res = run(f"curl {OLLAMA_URL}", shell=True, check=True, capture_output=True)
        if res.returncode == 0:
            break
        sleep(10)
    else:
        raise Exception("ollama not running")

    res = run(f"ollama run {MODEL_INFO['model']}", shell=True, check=True, capture_output=True)

    with open("/Modelfile", "w") as out:
        out.write(modelfile)

    res = run(
        "ollama create podcast_summarizer -f /Modelfile",
        shell=True,
        check=True,
        capture_output=True,
    )


app = App("podcast_summarize")

image = (
    Image.debian_slim()
    .apt_install("ffmpeg", "curl")
    .pip_install("ffmpeg-python", "yt-dlp", "pipx")
    .run_commands("curl -fsSL https://ollama.com/install.sh | sh")
    .run_commands(
        "PIPX_HOME=/ PIPX_BIN_DIR=/bin pipx install insanely-fast-whisper && pipx ensurepath && pipx --global ensurepath"
    )
    .run_function(install_ollama, gpu=GPU)
)

with image.imports():
    import glob
    import json
    import os
    import re
    import requests
    from subprocess import run
    from time import sleep


@app.cls(image=image, gpu=GPU, timeout=60 * 15)
class Llama:
    @method()
    def podcast_summarize(self, youtube_url: str) -> list[str, str]:
        """Summarize a podcast from a youtube URL"""
        Path(REMOTE_OUT).mkdir(parents=True, exist_ok=True)

        res = run(["yt-dlp", youtube_url, "-x", "--audio-format", "mp3", "--audio-quality", "5",
                   "-o", f'{REMOTE_OUT}/%(title)s_%(uploader)s.%(ext)s'],
                  check=True, capture_output=True)  # fmt: skip

        mp3_file = re.findall("Destination: (.+?)\n", res.stdout.decode())[-1]
        mp3_stem = Path(mp3_file).with_suffix("")

        res = run(["insanely-fast-whisper", "--file-name", f'{mp3_file}',
                   "--language", "english",
                   "--transcript-path", f'{mp3_stem}.transcript.json'],
                  check=True, capture_output=True)  # fmt: skip

        transcript = str(json.load(open(f"{mp3_stem}.transcript.json"))["text"])
        with open(f"{mp3_stem}.transcript.txt", "w") as out:
            out.write(transcript)

        os.system("ollama serve &")
        sleep(6)
        for _ in range(10):
            res = run(f"curl {OLLAMA_URL}", shell=True, check=True, capture_output=True)
            if res.returncode == 0:
                break
            sleep(10)
        else:
            raise Exception("ollama not running")

        url = f"{OLLAMA_URL}/api/generate"
        data = {"model": "podcast_summarizer", "prompt": f"{transcript}", "stream": False}
        response = requests.post(url, json=data)
        response.raise_for_status()

        with open(f"{mp3_stem}.info.json", "w") as out:
            out.write(str(response.json()))

        with open(f"{mp3_stem}.summary.txt", "w") as out:
            out.write(str(response.json()["response"]))

        return [
            (out_file, open(out_file, "rb").read())
            for out_file in glob.glob(f"{LOCAL_OUT}/**/*.*", recursive=True)
        ]


@app.local_entrypoint()
def main(youtube_url):
    outputs = Llama().podcast_summarize.remote(youtube_url)

    for out_file, out_content in outputs:
        Path(out_file).parent.mkdir(parents=True, exist_ok=True)
        if out_content:
            with open(out_file, "wb") as out:
                out.write(out_content)
