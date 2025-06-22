import os
import json
import traceback
import streamlit as st
import httpx
from uuid import uuid4
from tempfile import NamedTemporaryFile
from dotenv import load_dotenv
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
from pydub import AudioSegment
from urllib.parse import urlparse, parse_qs
from authlib.integrations.httpx_client import AsyncOAuth2Client
import asyncio
import socket

# === ENV ===
load_dotenv()
base_url = os.getenv("AGENT_BASE_URL", "https://localhost").rstrip("/")
audio_url = f"{base_url}/audio"
CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://localhost/auth")

# === Page Config ===
st.set_page_config(page_title="🎙️ A2A Voice Agent", layout="wide")
st.title("🔐 Login to MCpyATS")

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
st.title("🎤 Ask MCpyATS with Your Voice")

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
        st.success(f"✅ Agent response: {data.get('response_text')}")
        tts_url = data.get("tts_url")
        if tts_url:
            tts_audio = httpx.get(tts_url).content
            st.audio(tts_audio, format="audio/mp3")
            with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                f.write(tts_audio)
                mp3_path = f.name
            wav_path = mp3_path + ".wav"
            AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
            rate, samples = wavfile.read(wav_path)
            if samples.ndim > 1:
                samples = samples[:, 0]
            times = np.arange(len(samples)) / rate
            fig, ax = plt.subplots(figsize=(10, 2))
            ax.plot(times, samples)
            ax.set_title("🧠 Agent Voice Waveform")
            st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())

# === Mic Input ===
st.subheader("🎤 Or record your question:")
audio_value = st.audio_input("Record a voice message")
if audio_value:
    st.audio(audio_value)
    with NamedTemporaryFile(delete=False, suffix=".wav") as f:
        f.write(audio_value.getvalue())
        wav_path = f.name

    try:
        with open(wav_path, "rb") as f:
            files = {"file": ("voice.wav", f, "audio/wav")}
            headers = {"Authorization": f"Bearer {st.session_state['id_token']}"}

            hostname = audio_url.replace("https://", "").split("/")[0]
            try:
                resolved_ip = socket.gethostbyname(hostname)
                st.write(f"📡 DNS Resolved: `{hostname}` → `{resolved_ip}`")
            except Exception as dns_e:
                st.error(f"❌ DNS Error: {dns_e}")

            if "localhost" in audio_url or "127.0.0.1" in audio_url:
                st.error("❌ AUDIO_ENDPOINT cannot be localhost inside Docker. Use a reachable hostname or public URL.")
            else:
                st.info(f"📤 POSTing to: `{audio_url}`")
                response = httpx.post(audio_url, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                data = response.json()
                st.success(f"✅ Agent response: {data.get('response_text')}")
                tts_url = data.get("tts_url")
                if tts_url:
                    tts_audio = httpx.get(tts_url).content
                    st.audio(tts_audio, format="audio/mp3")
                    with NamedTemporaryFile(delete=False, suffix=".mp3") as f:
                        f.write(tts_audio)
                        mp3_path = f.name
                    wav_path = mp3_path + ".wav"
                    AudioSegment.from_mp3(mp3_path).export(wav_path, format="wav")
                    rate, samples = wavfile.read(wav_path)
                    if samples.ndim > 1:
                        samples = samples[:, 0]
                    times = np.arange(len(samples)) / rate
                    fig, ax = plt.subplots(figsize=(10, 2))
                    ax.plot(times, samples)
                    ax.set_title("🧠 Agent Voice Waveform")
                    st.pyplot(fig)
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())
