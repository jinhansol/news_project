import os
import tempfile
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 환경변수 로드
load_dotenv()
API_KEY = os.getenv('OPENAI_API_KEY')
if not API_KEY:
    raise ValueError('OPENAI_API_KEY is not defined')

client = OpenAI(api_key=API_KEY)

def generate_audio_file(text: str) -> str:
    if not text.strip():
        return None
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix='.mp3')
    path = Path(tmp.name)
    tmp.close()
    resp = client.audio.speech.create(model='tts-1', voice='alloy', input=text)
    resp.stream_to_file(path)
    return str(path)

def transcribe_audio_bytes(audio_bytes: bytes, model: str='whisper-1') -> str:
    resp = client.audio.transcriptions.create(file=audio_bytes, model=model)
    return resp.text