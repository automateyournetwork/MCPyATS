import os
import uuid
import httpx
import uvicorn
import asyncio
from io import BytesIO
from asyncio import Queue
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File, Response
from fastapi.middleware import Middleware
from fastapi.responses import RedirectResponse, JSONResponse, FileResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

import speech_recognition as sr
from pydub import AudioSegment

from authlib.integrations.starlette_client import OAuth
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

import tempfile
from pathlib import Path
from openai import OpenAI

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, SendMessageRequest, Message, TaskState
from a2a.server.agent_execution import RequestContext

from .agent_executor import LangGraphAgentExecutor

# === Load config ===
load_dotenv()
HOST = os.getenv("A2A_HOST", "0.0.0.0")
PORT = int(os.getenv("A2A_PORT", 10000))
PUBLIC_URL = os.getenv("PUBLIC_BASE_URL", f"http://localhost:{PORT}")
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
SESSION_SECRET = os.getenv("SESSION_SECRET", "supersecret")
TRUSTED_AGENT_EMAILS = os.getenv("TRUSTED_AGENT_EMAILS", "").split(",")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
TTS_DIR = Path(tempfile.gettempdir())

# === Audio2Face Trigger ===
def convert_mp3_to_pcm_wav(mp3_path: Path) -> Path:
    audio = AudioSegment.from_mp3(mp3_path)
    audio = audio.set_frame_rate(16000).set_channels(1).set_sample_width(2)
    wav_path = mp3_path.with_suffix(".wav")
    audio.export(wav_path, format="wav")
    return wav_path

# === OpenAI TTS Function ===
def generate_openai_tts(text: str, voice: str = "ash", model: str = "gpt-4o-mini-tts") -> str:
    filename = f"tts_{uuid.uuid4().hex}.mp3"
    full_path = TTS_DIR / filename
    with client.audio.speech.with_streaming_response.create(
        model=model,
        voice=voice,
        input=text,
        instructions="Speak clearly and naturally."
    ) as response:
        response.stream_to_file(full_path)

    # Trigger Audio2Face (optional)
    wav_path = convert_mp3_to_pcm_wav(full_path)
    return filename

# === OAuth Setup ===
oauth = OAuth()
oauth.register(
    name='google',
    client_id=GOOGLE_CLIENT_ID,
    client_secret=GOOGLE_CLIENT_SECRET,
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'},
)

# === Middleware and App ===
app = FastAPI(
    middleware=[Middleware(SessionMiddleware, secret_key=SESSION_SECRET, same_site="none", https_only=True)]
)

@app.get("/login")
async def login(request: Request):
    redirect_uri = f"{PUBLIC_URL}/auth"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = await oauth.google.parse_id_token(request, token)
    request.session["user"] = dict(user)
    return RedirectResponse(url="/")

@app.get("/.well-known/agent.json")
async def agent_card():
    card = build_agent_card()
    card_dict = card.model_dump(exclude_none=False)
    card_dict["endpoint"] = PUBLIC_URL
    return JSONResponse(content=card_dict)

@app.post("/audio")
async def handle_audio_input(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        audio = AudioSegment.from_file(BytesIO(audio_bytes))
        wav_io = BytesIO()
        audio.export(wav_io, format="wav", parameters=["-acodec", "pcm_s16le"])
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
            print(f"🎤 User asked: {transcribed_text}")

        message_id = str(uuid.uuid4())
        request_payload = {
            "jsonrpc": "2.0",
            "method": "message/send",
            "id": str(uuid.uuid4()),
            "params": {
                "message": {
                    "role": "user",
                    "messageId": message_id,
                    "parts": [{"kind": "text", "text": transcribed_text}]
                }
            }
        }

        async with httpx.AsyncClient(timeout=120.0) as client:
            post_response = await client.post(f"{PUBLIC_URL}/", json=request_payload)
            post_response.raise_for_status()
            task_response = post_response.json()
            task_id = task_response.get("result", {}).get("id")

            if not task_id:
                raise ValueError(f"Missing task ID in response: {task_response}")

            for _ in range(60):
                await asyncio.sleep(1)
                poll_payload = {
                    "jsonrpc": "2.0",
                    "method": "tasks/get",
                    "id": str(uuid.uuid4()),
                    "params": {"id": task_id}
                }
                poll_response = await client.post(f"{PUBLIC_URL}/", json=poll_payload)
                task_status = poll_response.json()
                status = task_status.get("result", {}).get("status", {}).get("state")
                print(f"🕒 Polling task {task_id} status: {status}")

                if status == "completed":
                    result_message = task_status.get("result", {}).get("status", {}).get("message", {})
                    parts = result_message.get("parts", [])
                    text_reply = next((p.get("text") for p in parts if p.get("kind") == "text"), "[No text found in parts]")
                    tts_filename = generate_openai_tts(f"You asked: {transcribed_text}. {text_reply}")
                    tts_url = f"{PUBLIC_URL}/tts/{tts_filename}"
                    return JSONResponse({
                        "transcription": transcribed_text,
                        "response_text": text_reply,
                        "task_id": task_id,
                        "tts_url": tts_url
                    })

                elif status in ["failed", "cancelled"]:
                    return JSONResponse({
                        "transcription": transcribed_text,
                        "error": f"Task failed with status: {status}",
                        "task_id": task_id
                    })

            return JSONResponse({
                "transcription": transcribed_text,
                "error": "Timeout waiting for task result",
                "task_id": task_id
            })

    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

@app.get("/tts/{filename}")
async def get_tts_file(filename: str):
    file_path = TTS_DIR / filename
    if not file_path.exists():
        return JSONResponse({"error": "File not found"}, status_code=404)
    return FileResponse(file_path, media_type="audio/mpeg")

class InjectBearerUserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        auth_header = request.headers.get("authorization")
        if auth_header and auth_header.lower().startswith("bearer "):
            token = auth_header.split(" ", 1)[1]
            try:
                id_info = id_token.verify_oauth2_token(
                    token,
                    google_requests.Request(),
                    GOOGLE_CLIENT_ID
                )
                email = id_info.get("email")
                if email not in TRUSTED_AGENT_EMAILS:
                    return JSONResponse({"error": f"Unauthorized agent: {email}"}, status_code=403)
                request.state.user = id_info
            except Exception as e:
                return JSONResponse({"error": f"Invalid Bearer token: {str(e)}"}, status_code=401)
        return await call_next(request)

def build_agent_card() -> AgentCard:
    return AgentCard(
        name="MCpyATS",
        description="Cisco pyATS LangGraph agent with A2A interface",
        version="1.0.0",
        url=PUBLIC_URL,
        endpoint=PUBLIC_URL,
        defaultInputModes=["text", "audio"],
        defaultOutputModes=["text", "audio"],
        capabilities=AgentCapabilities(
            a2a=True,
            toolUse=True,
            chat=True,
            streaming=True,
            push=True,
        ),
        skills=[
            AgentSkill(
                id="pyats",
                name="Cisco pyATS",
                description="Run show commands or perform configuration management on Cisco network devices",
                tags=["cisco", "network", "automation", "show commands", "configure"]
            )
        ]
    )

executor = LangGraphAgentExecutor()
request_handler = DefaultRequestHandler(
    agent_executor=executor,
    task_store=InMemoryTaskStore(),
    push_notifier=InMemoryPushNotifier(httpx.AsyncClient()),
)
a2a_app = A2AStarletteApplication(
    agent_card=build_agent_card(),
    http_handler=request_handler,
)

app.add_middleware(InjectBearerUserMiddleware)
app.mount("/", a2a_app.build())

if __name__ == "__main__":
    uvicorn.run("agent.__main__:app", host=HOST, port=PORT, reload=True)
    print(f"🚀 A2A Agent running at {PUBLIC_URL}")