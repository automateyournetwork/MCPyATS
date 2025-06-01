import asyncio
import os
import traceback
from uuid import uuid4
from typing import Any
from urllib.parse import urlparse, parse_qs, urljoin

import httpx
from dotenv import load_dotenv
from authlib.integrations.httpx_client import AsyncOAuth2Client

from a2a.client import A2AClient
from a2a.types import (
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    SendMessageSuccessResponse,
    TaskState,
)

# === Load environment variables ===
load_dotenv()
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET")
REDIRECT_URI = os.getenv("REDIRECT_URI", "https://localhost/auth")

AGENT_BASE_URL = os.getenv("AGENT_BASE_URL", "https://localhost")
AGENT_CARD_PATH = "/.well-known/agent.json"

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

    if not redirect_response.startswith("http"):
        raise ValueError("Invalid redirect URL pasted. It should start with http or https.")

    parsed_redirect_query = parse_qs(urlparse(redirect_response).query)
    code = parsed_redirect_query.get("code", [None])[0]
    if not code:
        error = parsed_redirect_query.get("error", [None])[0]
        error_description = parsed_redirect_query.get("error_description", ["No description"])[0]
        raise ValueError(f"❌ OAuth error: {error} - {error_description}")

    token = await oauth_client.fetch_token(
        url='https://oauth2.googleapis.com/token',
        grant_type='authorization_code',
        code=code,
        redirect_uri=REDIRECT_URI,
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
    )
    return token["id_token"]

# === Message helpers ===
def create_send_message_payload(text: str, task_id: str | None = None, context_id: str | None = None) -> dict[str, Any]:
    payload = {
        "message": {
            "role": "user",
            "parts": [{"kind": "text", "text": text}],
            "messageId": uuid4().hex,
        }
    }
    if task_id:
        payload["message"]["taskId"] = task_id
    if context_id:
        payload["message"]["contextId"] = context_id
    return payload

def extract_clean_text(task_data: dict[str, Any]) -> str:
    try:
        message_source = task_data.get("status", {}).get("message", {})
        if task_data.get("status", {}).get("state") == "completed" and task_data.get("result", {}).get("message"):
            message_source = task_data.get("result", {}).get("message", {})
        parts = message_source.get("parts", [])
        for part in parts:
            if "text" in part:
                return part["text"]
        return "[⚠️ No valid text found in message parts]"
    except Exception as e:
        return f"[⚠️ Error extracting answer: {e} from data: {task_data}]"

# === Chat loop ===
async def chat_loop(client: A2AClient):
    context_id = None
    task_id = None
    while True:
        try:
            user_input = input("\n❓ Question: ").strip()
            if not user_input:
                print("👋 Exiting.")
                break
            payload = create_send_message_payload(user_input, task_id, context_id)
            request = SendMessageRequest(params=MessageSendParams(**payload))
            print(f"🚀 Sending message to: {getattr(client, 'agent_endpoint', 'N/A')}")
            response_model = await client.send_message(request)
            if not isinstance(response_model, SendMessageSuccessResponse):
                print(f"❌ Failed to send message. Response: {response_model}")
                continue
            task = response_model.result
            task_id = task.id
            context_id = getattr(task, 'contextId', context_id)
            print(f"⏳ Task created: ID={task_id}, ContextID={context_id}. Polling for completion...")
            for i in range(60):
                await asyncio.sleep(1)
                task_response_model = await client.get_task(GetTaskRequest(params=TaskQueryParams(id=task_id)))
                task_status_data = task_response_model.result
                current_state = task_status_data.status.state
                print(f"📡 Status (Attempt {i+1}): {current_state}")
                if current_state == TaskState.completed:
                    answer = extract_clean_text(task_status_data.model_dump())
                    print(f"✅ Answer: {answer}")
                    break
                elif current_state in [TaskState.failed, TaskState.cancelled]:
                    error_message = extract_clean_text(task_status_data.model_dump())
                    print(f"❌ Task {current_state}: {error_message}")
                    break
            else:
                print(f"⚠️ Timeout waiting for response for task {task_id}.")
        except KeyboardInterrupt:
            print("\n👋 Conversation ended.")
            break
        except Exception as e:
            traceback.print_exc()
            print(f"❌ Unexpected error in chat loop: {e}")

# === Main logic ===
async def main():
    print("🔌 Starting OAuth flow...")
    try:
        id_token_str = await get_google_oauth_token()
        print("✅ ID token obtained.")

        headers = {"Authorization": f"Bearer {id_token_str}"}
        async with httpx.AsyncClient(headers=headers, timeout=600.0) as http_client_session:
            agent_card_full_url = urljoin(AGENT_BASE_URL, AGENT_CARD_PATH)
            print(f"ℹ️ Fetching agent card from: {agent_card_full_url}")
            agent_card_resp = await http_client_session.get(agent_card_full_url)
            agent_card_resp.raise_for_status()
            agent_card = agent_card_resp.json()
            print(f"🃏 Agent card content: {agent_card}")
            discovered_endpoint = agent_card.get("endpoint")
            print(f"🗺️ Endpoint from agent card: {discovered_endpoint}")

            print(f"🛠️ Initializing A2AClient using agent card endpoint.")
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client=http_client_session,
                base_url=AGENT_BASE_URL,
                agent_card_path=AGENT_CARD_PATH
            )
            print(f"✅ Client initialized. Effective POST endpoint: {getattr(client, 'agent_endpoint', 'unknown')}")
            print(f"🗣️ Starting chat loop...\n")
            await chat_loop(client)

    except httpx.HTTPStatusError as e:
        traceback.print_exc()
        print(f"❌ HTTP error: {e.response.status_code} - {e.response.text}")
        print(f"Request was to: {e.request.url}")
    except ValueError as e:
        traceback.print_exc()
        print(f"❌ Configuration or Value Error: {e}")
    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed during client setup or communication: {e}")

if __name__ == "__main__":
    asyncio.run(main())
