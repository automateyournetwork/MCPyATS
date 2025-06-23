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
model_url = "https://www.automateyournetwork.ca/avatar/capo.glb"
html_content = f'''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Three.js GLTF Model</title>
    <style>
        body {{ margin: 0; }}
        canvas {{ display: block; }}
        #three-container {{ width: 100vw; height: 100vh; }}
    </style>
    <script type="importmap">
    {{
        "imports": {{
        "three": "https://cdn.jsdelivr.net/npm/three@0.149.0/build/three.module.js",
        "three/examples/jsm/": "https://cdn.jsdelivr.net/npm/three@0.149.0/examples/jsm/"
        }}
    }}
    </script>
</head>
<body id="avatar-body">
    <div id="three-container"></div>
    <script type="module">
        import * as THREE from 'three';
        import {{ OrbitControls }} from 'three/examples/jsm/controls/OrbitControls.js';
        import {{ GLTFLoader }} from 'three/examples/jsm/loaders/GLTFLoader.js';

        let camera, scene, renderer;
        let model;

        function startModelSpin() {{
            if (!model) return;
            let t = 0;
            const spinDuration = 2000;
            const start = performance.now();
            function spinStep(now) {{
                const elapsed = now - start;
                if (elapsed < spinDuration) {{
                    model.rotation.y += 0.05;
                    render(); // <-- Ensure we re-render after rotation
                    requestAnimationFrame(spinStep);
                }}
            }}
            requestAnimationFrame(spinStep);
        }}

        window.addEventListener("message", (event) => {{
            if (event.data && event.data.type === "AUDIO_PLAYBACK_STARTED") {{
                console.log("🔄 Received AUDIO_PLAYBACK_STARTED → spinning head!");
                startModelSpin();
            }}
            if (event.data && event.data.type === "MIC_RECORDING_STARTED") {{
                console.log("🎙️ Received MIC_RECORDING_STARTED → spinning head!");
                startModelSpin();
            }}
        }});

        function init() {{
            const container = document.getElementById('three-container');

            camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.25, 20);
            camera.position.set(1, 0.9, 1);

            scene = new THREE.Scene();
            scene.background = new THREE.Color(0x222233);

            const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
            scene.add(ambientLight);
            const directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
            directionalLight.position.set(1, 1, 1);
            scene.add(directionalLight);

            renderer = new THREE.WebGLRenderer({{ antialias: true }});
            renderer.setPixelRatio(window.devicePixelRatio);
            renderer.setSize(container.clientWidth, container.clientHeight);
            container.appendChild(renderer.domElement);

            const controls = new OrbitControls(camera, renderer.domElement);
            controls.addEventListener('change', render);
            controls.maxDistance = 10;
            controls.target.set(0, 0, -0.2);
            controls.update();

            const loader = new GLTFLoader();
            loader.load('{model_url}', function (gltf) {{
                model = gltf.scene;
                const box = new THREE.Box3().setFromObject(model);
                const size = new THREE.Vector3();
                box.getSize(size);
                const center = new THREE.Vector3();
                box.getCenter(center);
                model.position.sub(center);

                const maxDim = Math.max(size.x, size.y, size.z);
                const fov = camera.fov * (Math.PI / 180);
                let cameraZ = Math.abs(maxDim / 2 / Math.tan(fov / 2));
                cameraZ *= 1.5;
                camera.position.set(0, 0, cameraZ);
                camera.lookAt(0, 0, 0);

                scene.add(model);
                render();
            }}, undefined, function (error) {{
                console.error('An error happened:', error);
            }});

            window.addEventListener('resize', onWindowResize);
            render();
        }}     

        function onWindowResize() {{
            const container = document.getElementById('three-container');
            camera.aspect = container.clientWidth / container.clientHeight;
            camera.updateProjectionMatrix();
            renderer.setSize(container.clientWidth, container.clientHeight);
            render();
        }}

        function render() {{
            renderer.render(scene, camera);
        }}

        init();
    </script>
</body>
</html>
'''
components.html(html_content, height=600)

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
            options = WaveSurferOptions(wave_color="#2B88D9", progress_color="#b91d47", height=100)
            result = audix(wav_path, wavesurfer_options=options)
            if result:
                st.write(f"▶️ Current Time: {result['currentTime']}s")
                if result['selectedRegion']:
                    st.write(f"🔍 Selected: {result['selectedRegion']['start']} - {result['selectedRegion']['end']}s")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())

def trigger_avatar_animation(event_type):
    st.components.v1.html(f"""
    <script>
        function findAvatarIframe() {{
            const iframes = Array.from(parent.document.querySelectorAll('iframe'));
            return iframes.find(iframe => (
                iframe.contentWindow &&
                iframe.contentWindow.document &&
                iframe.contentWindow.document.getElementById('three-container')
            ));
        }}
        function sendAnimationEvent() {{
            const avatarIframe = findAvatarIframe();
            if (avatarIframe) {{
                avatarIframe.contentWindow.postMessage({{ type: "{event_type}" }}, "*");
                console.log("Sent {event_type} to avatar iframe");
            }} else {{
                setTimeout(sendAnimationEvent, 500);
            }}
        }}
        sendAnimationEvent();
    </script>
    """, height=0)

# === Mic Input ===
st.subheader("🎤 Or record your question:")
audio_value = st.audio_input("Record a voice message")
if audio_value and not st.session_state.get("ai_response_ready", False):
    trigger_avatar_animation("MIC_RECORDING_STARTED")
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
            except Exception as dns_e:
                st.error(f"❌ DNS Error: {dns_e}")

            if "localhost" in audio_url or "127.0.0.1" in audio_url:
                st.error("❌ AUDIO_ENDPOINT cannot be localhost inside Docker. Use a reachable hostname or public URL.")
            else:
                with st.spinner(f"📤 POSTing to: `{audio_url}` please be patient"):
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

                        # Save to session to avoid reruns
                        st.session_state["ai_response_ready"] = True
                        st.session_state["tts_wav_path"] = wav_path
                        st.session_state["agent_text_response"] = data.get("response_text")
                        st.session_state["agent_transcription"] = data.get("transcription", "You asked (via voice)")
    except Exception as e:
        st.error(f"❌ Error: {e}")
        st.error(traceback.format_exc())

# === Agent Response Display ===
if st.session_state.get("ai_response_ready"):
    st.subheader("✅ Agent's Answer")   
    wav_path = st.session_state["tts_wav_path"]
    options = WaveSurferOptions(wave_color="#2B88D9", progress_color="#b91d47", height=100)
    trigger_avatar_animation("AUDIO_PLAYBACK_STARTED")
    result = audix(wav_path, wavesurfer_options=options)

    components.html(f"""
    <script>
        setTimeout(() => {{
            const waveform = document.querySelector('canvas');  // audix renders a canvas
    
            let realAudio = document.getElementById('hidden-native-audio');
            if (!realAudio) {{
                realAudio = document.createElement('audio');
                realAudio.id = 'hidden-native-audio';
                realAudio.src = "{st.session_state['tts_wav_path']}";
                realAudio.style.display = 'none';
                document.body.appendChild(realAudio);
            }}
    
            realAudio.addEventListener("play", () => {{
                // Find the iframe that hosts the avatar
                const iframes = parent.document.querySelectorAll("iframe");
                for (const iframe of iframes) {{
                    if (iframe.src.includes("Three.js") || iframe.contentWindow?.postMessage) {{
                        iframe.contentWindow.postMessage({{ type: "AUDIO_PLAYBACK_STARTED" }}, "*");
                        console.log("📨 Sent AUDIO_PLAYBACK_STARTED to avatar iframe");
                    }}
                }}
            }});
    
            if (waveform) {{
                waveform.addEventListener("click", () => {{
                    realAudio.play().catch(err => console.warn("Audio play error", err));
                }});
            }}
        }}, 1000);
    </script>
    """, height=0)

    st.warning(f"User: {st.session_state.get('agent_transcription')}")
    st.success(st.session_state.get("agent_text_response", ""))
    
    if result:
        st.write(f"▶️ Current Time: {result['currentTime']}s")
        if result['selectedRegion']:
            st.write(f"🔍 Selected: {result['selectedRegion']['start']} - {result['selectedRegion']['end']}s")

