import os
import json
import traceback
import streamlit as st
import httpx
from uuid import uuid4
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from pydub import AudioSegment
from urllib.parse import urlparse, parse_qs
from authlib.integrations.httpx_client import AsyncOAuth2Client
import asyncio
import socket
from streamlit_advanced_audio import audix, WaveSurferOptions
import streamlit.components.v1 as components

# === ENV ===
load_dotenv()
base_url = os.getenv("AGENT_BASE_URL", "https://localhost").rstrip("/")
audio_url = f"{base_url}/audio"
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://localhost/auth")

# === Page Config ===
st.set_page_config(page_title="🎙️ A2A Voice Agent", layout="wide")

# === Session State ===
if "id_token" not in st.session_state:
    st.session_state["id_token"] = None
if "auth_success" not in st.session_state:
    st.session_state["auth_success"] = False

# === OAuth Login ===
async def get_google_oauth_token():
    oauth_client = AsyncOAuth2Client(
        client_id=CLIENT_ID,
        client_secret=CLIENT_SECRET,
        redirect_uri=REDIRECT_URI,
        scope='openid email profile',
    )
    authorization_url, _ = oauth_client.create_authorization_url(
        'https://accounts.google.com/o/oauth2/v2/auth'
    )
    st.markdown(f"🔗 [Click here to authorize with Google]({authorization_url})")
    redirect_response = st.text_input("🔑 Paste the full redirect URL here:")

    if redirect_response:
        parsed_redirect_query = parse_qs(urlparse(redirect_response).query)
        code = parsed_redirect_query.get("code", [None])[0]
        if not code:
            st.error("❌ OAuth code not found in response.")
            return

        try:
            token = await oauth_client.fetch_token(
                url='https://oauth2.googleapis.com/token',
                grant_type='authorization_code',
                code=code,
                redirect_uri=REDIRECT_URI,
                client_id=CLIENT_ID,
                client_secret=CLIENT_SECRET,
            )
            st.session_state["id_token"] = token["id_token"]
            st.session_state["auth_success"] = True
            st.success("✅ Logged in! Click Next to continue.")
            st.rerun()
        except Exception as e:
            st.error(f"OAuth Error: {e}")
            st.error(traceback.format_exc())

if not st.session_state["id_token"] and not st.session_state["auth_success"]:
    asyncio.run(get_google_oauth_token())
    st.stop()

if st.session_state["auth_success"] and not st.session_state["id_token"]:
    if st.button("➡️ Next"):
        st.rerun()
    st.stop()

# === Authenticated Page ===
st.title("🎤 Talk To Your Network")

# === 3D GLB AI Face ===
st.subheader("🧠 AI Agent Avatar")
# === Text Input ===
st.subheader("💬 Or type your message:")
text_prompt = st.text_input("Type your prompt and press Enter")
if text_prompt:
    try:
        headers = {"Authorization": f"Bearer {st.session_state['id_token']}", "Content-Type": "application/json"}
        payload = {"text": text_prompt}
        st.info(f"📤 POST to `{audio_url}` with payload: {payload}")
        response = httpx.post(audio_url, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        tts_url = data.get("tts_url")
        if tts_url:
            tts_audio = httpx.get(tts_url).content
            with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(tts_audio)
                mp3_path = f.name
            wav_path = mp3_path + ".wav"
            AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())

def trigger_avatar_animation(event_type, audio_url=None):
    payload = {"type": event_type}
    if audio_url:
        payload["audioUrl"] = audio_url
    st.components.v1.html(f"""
    <script>
        const iframe = Array.from(parent.document.querySelectorAll('iframe')).find(f =>
            f.contentWindow?.document?.getElementById('three-container'));
        if (iframe) {{
            iframe.contentWindow.postMessage({json.dumps(payload)}, "*");
        }}
    </script>
    """, height=0)

# === Mic Input ===
st.subheader("🎤 Or record your question:")
audio_value = st.audio_input("Record a voice message")
model_url = "https://www.automateyournetwork.ca/avatar/capo.glb"
html_content = open("flapping_avatar.html").read().replace("{{MODEL_URL}}", model_url)
components.html(html_content, height=600)
if audio_value and not st.session_state.get("ai_response_ready", False):
    st.audio(audio_value)
    with NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_value.getvalue())
        wav_path = f.name
    try:
        with open(wav_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            headers = {"Authorization": f"Bearer {st.session_state['id_token']}"}
            response = httpx.post(audio_url, files=files, headers=headers, timeout=60)
            response.raise_for_status()
            data = response.json()
            tts_url = data.get("tts_url")
            if tts_url:
                tts_audio = httpx.get(tts_url).content
                with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                    f.write(tts_audio)
                    mp3_path = f.name
                wav_path = mp3_path + ".wav"
                AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
                os.makedirs("static/audio", exist_ok=True)
                audio_filename = "audio.wav"
                static_path = os.path.join("static/audio", audio_filename)
                AudioSegment.from_mp3(mp3_path).export(static_path, format="wav")

                # Save it for later injection
                st.session_state["tts_audio_url"] = f"http://localhost:8501/static/audio/{audio_filename}"               
                st.session_state.update({
                    "ai_response_ready": True,
                    "tts_wav_path": wav_path,
                    "agent_text_response": data.get("response_text"),
                    "agent_transcription": data.get("transcription", "You asked (via voice)")
                })
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())

# === Agent Response Display ===
if st.session_state.get("ai_response_ready"):
    st.subheader("✅ Agent's Answer")   
    wav_path = st.session_state["tts_wav_path"]
    tts_audio_url = st.session_state["tts_audio_url"]  # http://localhost:8501/static/audio/audio.wav

    options = WaveSurferOptions(wave_color="#2B88D9", progress_color="#b91d47", height=100)

    # ✅ Only call once, with consistent key
    audix(wav_path, wavesurfer_options=options, key="agent-audio")

    components.html(f"""
    <script>
      const findAudio = () => document.querySelector('#agent-audio audio, [data-testid="stAudio"] audio, audio');
      const findIframe = () => Array.from(parent.document.querySelectorAll('iframe')).find(f =>
        f.contentWindow?.document?.getElementById('three-container'));

      const audio = findAudio();
      const iframe = findIframe();

      if (audio && iframe && !audio.dataset.hooked) {{
        audio.dataset.hooked = "true";

        console.log("✅ Hooked into audio element");

        audio.addEventListener('play', () => {{
          console.log("▶️ Detected audio play");
          iframe.contentWindow.postMessage({{
            type: "AUDIO_PLAYBACK_STARTED",
            audioUrl: "{tts_audio_url}"
          }}, "*");
        }});

        console.log("✅ {tts_audio_url} is ready to play");
        
        audio.addEventListener('ended', () => {{
          console.log("🛑 Detected audio ended");
          iframe.contentWindow.postMessage({{
            type: "STOP_FLAPPING"
          }}, "*");
        }});
      }} else {{
        console.warn("❌ Audio or iframe not found, or already hooked");
      }}
    </script>
    """, height=0)

    st.warning(f"User: {st.session_state.get('agent_transcription')}")
    st.success(st.session_state.get("agent_text_response", ""))