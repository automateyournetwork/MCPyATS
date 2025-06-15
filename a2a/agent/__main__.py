import os
import uuid
import base64
import httpx
import uvicorn
from io import BytesIO
from asyncio import Queue
from typing import Optional
from dotenv import load_dotenv
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.middleware import Middleware
from fastapi.responses import RedirectResponse, JSONResponse
from starlette.middleware.sessions import SessionMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

import speech_recognition as sr
from pydub import AudioSegment

from authlib.integrations.starlette_client import OAuth
from google.oauth2 import id_token
from google.auth.transport import requests as google_requests

from a2a.server.apps import A2AStarletteApplication
from a2a.server.tasks import InMemoryTaskStore, InMemoryPushNotifier
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.types import AgentCard, AgentCapabilities, AgentSkill, SendMessageRequest, Message
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

# === OAuth Setup for Human Login ===
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

@app.get("/")
async def home(request: Request):
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login")
    return JSONResponse({"message": "Welcome to the MCpyATS Agent. POST to / with A2A requests."})

@app.get("/login")
async def login(request: Request):
    redirect_uri = f"{PUBLIC_URL}/auth"
    return await oauth.google.authorize_redirect(request, redirect_uri)

@app.get("/auth")
async def auth(request: Request):
    token = await oauth.google.authorize_access_token(request)
    user = await oauth.google.parse_id_token(request, token)
    request.session["user"] = dict(user)
    request.session["user"]["id_token"] = token["id_token"]  # ✅ Save id_token
    return RedirectResponse(url="/")

@app.get("/.well-known/agent.json")
async def agent_card():
    card = build_agent_card()
    card_dict = card.model_dump(exclude_none=False)
    card_dict["endpoint"] = PUBLIC_URL
    return JSONResponse(content=card_dict)

# === Audio Input Endpoint ===
@app.post("/audio")
async def handle_audio_input(request: Request, file: UploadFile = File(...)):
    print(f"📥 Received audio upload: filename={file.filename}, content_type={file.content_type}")
    
    if not file or not file.filename:
        return JSONResponse({"error": "No file provided"}, status_code=400)

    try:
        audio_bytes = await file.read()
        if len(audio_bytes) == 0:
            return JSONResponse({"error": "Empty file received"}, status_code=400)

        try:
            audio = AudioSegment.from_file(BytesIO(audio_bytes))
        except Exception as decode_err:
            return JSONResponse({"error": f"Could not decode audio input: {str(decode_err)}"}, status_code=400)

        wav_io = BytesIO()
        audio.export(wav_io, format="wav", parameters=["-acodec", "pcm_s16le"])
        wav_io.seek(0)

        recognizer = sr.Recognizer()
        with sr.AudioFile(wav_io) as source:
            audio_data = recognizer.record(source)
            transcribed_text = recognizer.recognize_google(audio_data)
            print(f"📝 Transcription: {transcribed_text}")

        # Format as SendMessageRequest JSON
        text_payload = {
            "type": "SendMessageRequest",
            "params": {
                "message": {
                    "role": "user",
                    "messageId": str(uuid.uuid4()),
                    "parts": [{"kind": "text", "text": transcribed_text}]
                }
            }
        }

        headers = {}
        user = request.session.get("user")
        if user and "id_token" in user:
            headers["Authorization"] = f"Bearer {user['id_token']}"

        # ✅ Post to A2A endpoint
        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{PUBLIC_URL}/a2a/messages",
                json=text_payload,
                headers=headers,
                timeout=30.0
            )

        return JSONResponse({
            "transcription": transcribed_text,
            "agent_response": response.json()
        })

    except sr.UnknownValueError:
        return JSONResponse({"error": "Could not understand audio"}, status_code=400)
    except sr.RequestError as e:
        return JSONResponse({"error": f"Speech recognition error: {e}"}, status_code=500)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return JSONResponse({"error": f"Audio processing error: {str(e)}"}, status_code=500)

# === Middleware to Inject Bearer User ===
class InjectBearerUserMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in ["/audio", "/.well-known/agent.json"]:
            return await call_next(request)

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

# === Agent Metadata ===
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

# === A2A App Setup ===
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

# === Mount A2A Agent ===
app.add_middleware(InjectBearerUserMiddleware)
app.mount("/a2a", a2a_app.build())

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)