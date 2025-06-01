import json
import os
from uuid import uuid4
import httpx

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events.event_queue import EventQueue
from a2a.types import (
    Task,
    TaskStatusUpdateEvent,
    TaskStatus,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    AgentSkill,
    AgentCard
)
from a2a.utils import new_agent_text_message
from a2a.client import A2AClient

LANGGRAPH_URL = os.getenv("LANGGRAPH_URL", "http://host.docker.internal:2024")
ASSISTANT_ID = os.getenv("ASSISTANT_ID", "MCpyATS")
PEER_AGENT_URLS = os.getenv("PEER_AGENT_URLS", "").split(",") if os.getenv("PEER_AGENT_URLS") else []

class LangGraphAgentExecutor(AgentExecutor):
    """A2A AgentExecutor wrapper for LangGraph with intelligent peer delegation."""

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        query = context.get_user_input()
        task = context.current_task or Task(
            id=context.task_id or str(uuid4()),
            contextId=context.context_id or str(uuid4()),
            kind="task",
            status=TaskStatus(state=TaskState.submitted),
            history=[]
        )

        task_id = task.id
        context_id = task.contextId

        print("🔍 Selecting best agent for query...")
        best_agent = await self.select_best_agent_for_query(query)

        if best_agent == "self":
            print("💡 Handling task locally.")
            await self.execute_locally(query, context_id, task_id, event_queue)
        elif isinstance(best_agent, str) and best_agent.startswith("http"):
            print(f"🤝 Delegating task to peer: {best_agent}")
            await self.delegate_to_peer(best_agent, query, context_id, task_id, event_queue)
        else:
            print("❌ No matching agent found.")
            event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message("No agent found to handle the request.", context_id, task_id),
                ),
                contextId=context_id,
                taskId=task_id,
                final=True,
            ))

    async def execute_locally(self, query: str, context_id: str, task_id: str, event_queue: EventQueue) -> None:
        try:
            print(f"🔁 Calling LangGraph locally at {LANGGRAPH_URL}")
            async with httpx.AsyncClient(base_url=LANGGRAPH_URL, timeout=600) as client:
                thread_resp = await client.post("/threads", json={"assistant_id": ASSISTANT_ID})
                thread_resp.raise_for_status()
                thread_id = thread_resp.json().get("thread_id")
                print(f"✅ Thread created: {thread_id}")

                if not thread_id:
                    raise RuntimeError("❌ No thread_id returned")

                content_chunks = []
                async with client.stream("POST", f"/threads/{thread_id}/runs/stream", json={
                    "input": {
                        "messages": [{"role": "user", "type": "human", "content": query}],
                        "metadata": {}
                    },
                    "assistant_id": ASSISTANT_ID,
                }) as response:
                    async for line in response.aiter_lines():
                        if not line.startswith("data:"):
                            continue
                        try:
                            payload = json.loads(line[5:].strip())
                            if isinstance(payload.get("content"), str):
                                content_chunks.append(payload["content"])
                            elif isinstance(payload.get("messages"), list):
                                for msg in reversed(payload["messages"]):
                                    if msg.get("type") == "ai":
                                        content_chunks.append(msg["content"])
                                        break
                        except Exception as e:
                            print(f"⚠️ JSON decode failed: {e}")
                            continue

                final_content = "\n".join(content_chunks).strip()
                if not final_content:
                    raise RuntimeError("❌ No usable content returned")

                event_queue.enqueue_event(TaskStatusUpdateEvent(
                    status=TaskStatus(
                        state=TaskState.completed,
                        message=new_agent_text_message(final_content, context_id, task_id)
                    ),
                    contextId=context_id,
                    taskId=task_id,
                    final=True,
                ))

        except Exception as e:
            error_msg = f"🔥 Exception: {e}"
            print(error_msg)
            event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(error_msg, context_id, task_id),
                ),
                contextId=context_id,
                taskId=task_id,
                final=True,
            ))

    async def delegate_to_peer(self, url: str, query: str, context_id: str, task_id: str, event_queue: EventQueue) -> None:
        try:
            payload = SendMessageRequest(params=MessageSendParams(
                message={
                    "role": "user",
                    "parts": [{"kind": "text", "text": query}],
                    "messageId": str(uuid4()),
                    "contextId": context_id,
                    "taskId": task_id,
                }
            ))

            final_chunks = []

            async with httpx.AsyncClient(timeout=600) as httpx_client:
                peer = await A2AClient.get_client_from_agent_card_url(httpx_client, url)

                # 🧠 Fetch agent card manually
                response = await httpx_client.get(f"{url.rstrip('/')}/.well-known/agent.json")
                response.raise_for_status()
                agent_card = AgentCard.model_validate(response.json())
                peer.agent_card = agent_card  # manually patch

                # 🧪 Check if streaming is supported
                if "stream" in (agent_card.defaultOutputModes or []):
                    print("📡 Using streaming mode with peer")
                    async for msg in peer.send_message_streaming(payload):
                        print("📥 Peer stream part:", msg)
                        if msg.parts and msg.parts[0].kind == "text":
                            final_chunks.append(msg.parts[0].text)
                else:
                    print("📨 Using non-streaming mode with peer")
                    result = await peer.send_message(payload)
                    task = result.root.result

                    final_chunks = []
                    for part in task.status.message.parts:
                        if hasattr(part, "text"):  # safest universal fallback
                            final_chunks.append(part.text)
                        else:
                            final_chunks.append(str(part))  # just in case

            final_msg = "\n".join(final_chunks).strip() or "✅ Delegated, but no final message received."

            event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.completed,
                    message=new_agent_text_message(final_msg, context_id, task_id),
                ),
                contextId=context_id,
                taskId=task_id,
                final=True,
            ))

        except Exception as e:
            error_msg = f"❌ Failed to delegate to peer: {e}"
            print(error_msg)
            event_queue.enqueue_event(TaskStatusUpdateEvent(
                status=TaskStatus(
                    state=TaskState.failed,
                    message=new_agent_text_message(error_msg, context_id, task_id),
                ),
                contextId=context_id,
                taskId=task_id,
                final=True,
            ))

    async def select_best_agent_for_query(self, query: str) -> str | tuple[A2AClient, str] | None:
        query_words = set(word.lower() for word in query.split())

        local_skills = await self.get_local_skills()
        best_score = await self._score_agent_skills(query_words, local_skills)
        best_agent = "self"

        print(f"🔍 Self score: {best_score}")

        async with httpx.AsyncClient() as client:
            for url in PEER_AGENT_URLS:
                try:
                    peer = await A2AClient.get_client_from_agent_card_url(client, url)
                    response = await client.get(f"{url.rstrip('/')}/.well-known/agent.json")
                    response.raise_for_status()
                    peer_card = AgentCard.model_validate(response.json())
                    peer.agent_card = peer_card  # Patch the client

                    peer_score = await self._score_agent_skills(query_words, peer_card.skills)
                    print(f"🔗 Peer {url} score: {peer_score}")
                    if peer_score > best_score:
                        best_score = peer_score
                        best_agent = url
                except Exception as e:
                    print(f"⚠️ Could not contact peer {url}: {e}")

        return best_agent

    async def _score_agent_skills(self, query_words: set, skills: list[AgentSkill]) -> int:
        score = 0
        for skill in skills:
            text_blob = " ".join([
                skill.name or "",
                skill.description or "",
                " ".join(skill.tags or [])
            ]).lower()
            skill_words = set(text_blob.split())
            matches = query_words & skill_words
            score += len(matches)
        return score

    async def get_local_skills(self) -> list[AgentSkill]:
        return [
            AgentSkill(
                id="default",
                name="LangGraph handler",
                description="Handles requests with LangGraph",
                tags=["langgraph", "network", "selector", "pyats"]
            )
        ]

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Cancel is not implemented.")
