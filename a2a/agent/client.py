import asyncio
import os
import sys
import traceback
from uuid import uuid4
from typing import Any
from urllib.parse import urlparse, parse_qs, urljoin
import httpx
from dotenv import load_dotenv
from authlib.integrations.httpx_client import AsyncOAuth2Client
from tempfile import NamedTemporaryFile

from a2a.client import A2AClient
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    SendMessageSuccessResponse,
    TaskState,
)

import sounddevice as sd
from scipy.io.wavfile import write
from pydub import AudioSegment

# === Load environment variables ===
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://localhost/auth")
AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "https://localhost")
AGENT_CARD_PATH = "/.well-known/agent.json"

# === Trigger Audio2Face ===
async def trigger_audio2face(wav_path: str, config: str = "a2f_client/scripts/audio2face_3d_microservices_interaction_app/config/config_mark.yml"):
    script_path = "a2f_client/scripts/audio2face_3d_microservices_interaction_app/a2f_3d.py"
    cmd = [
        sys.executable,
        script_path,
        "run_inference",
        wav_path,
        config,
        "--url", os.getenv("A2F_GRPC_URL", "localhost:50051"),
        "--skip-print-to-files"
]
    print("🎭 Starting A2F animation with command:", " ".join(cmd))
    proc = await asyncio.create_subprocess_exec(*cmd)
    await proc.communicate()
    print("✅ Audio2Face animation complete.")

# === OAuth Flow ===
async def get_google_oauth_token() -> str:
    oauth_client = AsyncOAuth2Client(
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope='openid email profile',
    )
    authorization_url, _ = oauth_client.create_authorization_url(
        'https://accounts.google.com/o/oauth2/v2/auth'
    )
    print(f"🔗 Open this URL in your browser to authorize:\n{authorization_url}")
    redirect_response = input("🔑 Paste the full redirect URL here: ").strip()

    parsed_redirect_query = parse_qs(urlparse(redirect_response).query)
    code = parsed_redirect_query.get("code", [None])[0]
    if not code:
        raise ValueError("❌ OAuth code not found in response.")

    token = await oauth_client.fetch_token(
        url='https://oauth2.googleapis.com/token',
        grant_type='authorization_code',
        code=code,
        redirect_uri=REDIRECT_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
    )
    return token["id_token"]

# === Audio playback + A2F animation ===
async def play_audio_from_url(tts_url: str):
    try:
        output_dir = "C:/Temp/Agent_Output"
        os.makedirs(output_dir, exist_ok=True)

        async with httpx.AsyncClient() as client:
            response = await client.get(tts_url)
            response.raise_for_status()
            audio_data = response.content

        with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            f.write(audio_data)
            mp3_path = f.name

        audio_segment = AudioSegment.from_file(mp3_path, format="mp3")
        wav_path = os.path.join(output_dir, "latest.wav")
        audio_segment.set_frame_rate(16000).set_channels(1).set_sample_width(2).export(wav_path, format="wav")

        print(f"💾 Saved TTS .wav output to: {wav_path}")
        print(f"🧠 Load this file manually in Audio2Face for animation.")

        os.unlink(mp3_path)  # Optionally delete intermediate .mp3

    except Exception as e:
        print(f"❌ Error during saving audio: {e}")

# === Audio capture and send ===
async def record_and_send_audio(http_client: httpx.AsyncClient, agent_endpoint: str, duration_sec: int = 20):
    fs = 16000
    print(f"🎙️ Recording {duration_sec}s of audio...")
    recording = sd.rec(
        int(duration_sec * fs),
        samplerate=fs,
        channels=1,
        dtype='int16',
        device=2  # 👈 This tells it to use the Yeti GX
    )
    sd.wait()

    with NamedTemporaryFile(suffix=".wav", delete=False) as temp_wav:
        write(temp_wav.name, fs, recording)
        wav_path = temp_wav.name

    audio = AudioSegment.from_wav(wav_path)
    with NamedTemporaryFile(suffix=".mp3", delete=False) as temp_mp3:
        audio.export(temp_mp3.name, format="mp3", bitrate="64k")
        mp3_path = temp_mp3.name

    try:
        base_url = agent_endpoint.rstrip("/").removesuffix("/a2a")
        audio_url = f"{base_url}/audio"
        with open(mp3_path, "rb") as f:
            files = {"file": ("voice.mp3", f, "audio/mpeg")}
            response = await http_client.post(audio_url, files=files, timeout=60)
            response.raise_for_status()
            data = response.json()
            print(f"📝 Agent response: {data.get('response_text')}")
            if "tts_url" in data:
                await play_audio_from_url(data["tts_url"])
                await asyncio.sleep(1)
    finally:
        os.unlink(wav_path)
        os.unlink(mp3_path)

# === Agent Messaging ===
def create_send_message_payload(text: str, task_id: str = None, context_id: str = None) -> dict[str, Any]:
    payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex
        }
    }
    if task_id:
        payload["message"]["taskId"] = task_id
    if context_id:
        payload["message"]["contextId"] = context_id
    return payload

def extract_clean_text(task_data: dict[str, Any]) -> str:
    if task_data.get("response_text"):
        return task_data["response_text"]
    message = task_data.get("result", {}).get("message", {})
    parts = message.get("parts", [])
    return next((p.get("text") for p in parts if "text" in p), "[No text found]")

# === Main Chat Loop ===
async def chat_loop(client: A2AClient, http_client: httpx.AsyncClient, agent_endpoint: str):
    context_id = task_id = None
    while True:
        user_input = input("\n❓ Ask something (or type 'mic'): ").strip()
        if user_input.lower() == "mic":
            await record_and_send_audio(http_client, agent_endpoint)
            continue
        if not user_input:
            break
        payload = create_send_message_payload(user_input, task_id, context_id)
        request = SendMessageRequest(params=MessageSendParams(**payload))
        response = await client.send_message(request)
        if isinstance(response, SendMessageSuccessResponse):
            task_id = response.result.id
            context_id = getattr(response.result, "contextId", context_id)
            for _ in range(60):
                await asyncio.sleep(1)
                task = await client.get_task(GetTaskRequest(params=TaskQueryParams(id=task_id)))
                if task.result.status.state == TaskState.completed:
                    answer = extract_clean_text(task.result.model_dump())
                    print(f"✅ Answer: {answer}")
                    break
                elif task.result.status.state in [TaskState.failed, TaskState.cancelled]:
                    print(f"❌ Task failed: {task.result.status.state}")
                    break
            else:
                print("⚠️ Task polling timeout.")
        else:
            print("❌ Failed to send message.")

# === Main Entry Point ===
async def main():
    try:
        id_token = await get_google_oauth_token()
        headers = {"Authorization": f"Bearer {id_token}"}
        async with httpx.AsyncClient(headers=headers) as http_client:
            agent_card_url = urljoin(AGENT_BASE_URL, AGENT_CARD_PATH)
            card_resp = await http_client.get(agent_card_url)
            card_resp.raise_for_status()
            agent_card = card_resp.json()
            endpoint = agent_card.get("endpoint")
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client=http_client,
                base_url=AGENT_BASE_URL,
                agent_card_path=AGENT_CARD_PATH
            )
            print("🗣️ Chat session started")
            await chat_loop(client, http_client, endpoint)
    except Exception:
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main())
