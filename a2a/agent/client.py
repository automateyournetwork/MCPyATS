from a2a.client import A2AClient
from typing import Any
from uuid import uuid4
from a2a.types import (
    SendMessageResponse,
    GetTaskResponse,
    SendMessageSuccessResponse,
    Task,
    TaskState,
    SendMessageRequest,
    MessageSendParams,
    GetTaskRequest,
    TaskQueryParams,
    SendStreamingMessageRequest,
)
import httpx
import traceback

# ✅ Set to your running public A2A Agent
AGENT_URL = "https://c111-76-68-137-181.ngrok-free.app"


def create_send_message_payload(
    text: str, task_id: str | None = None, context_id: str | None = None
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': text}],
            'messageId': uuid4().hex,
        },
    }
    if task_id:
        payload['message']['taskId'] = task_id
    if context_id:
        payload['message']['contextId'] = context_id
    return payload


def print_json_response(response: Any, description: str) -> None:
    print(f'--- {description} ---')
    if hasattr(response, 'root'):
        print(f'{response.root.model_dump_json(exclude_none=True)}\n')
    else:
        print(f'{response.model_dump(mode="json", exclude_none=True)}\n')


async def run_single_turn_test(client: A2AClient) -> None:
    send_payload = create_send_message_payload(text='Check interface status on R1')
    request = SendMessageRequest(params=MessageSendParams(**send_payload))

    print('--- Single Turn Request ---')
    send_response: SendMessageResponse = await client.send_message(request)
    print_json_response(send_response, 'Single Turn Request Response')

    if not isinstance(send_response.root, SendMessageSuccessResponse):
        print('Received non-success response. Aborting get task.')
        return

    if not isinstance(send_response.root.result, Task):
        print('Received non-task response. Aborting get task.')
        return

    task_id: str = send_response.root.result.id
    print('--- Query Task ---')
    get_request = GetTaskRequest(params=TaskQueryParams(id=task_id))
    get_response: GetTaskResponse = await client.get_task(get_request)
    print_json_response(get_response, 'Query Task Response')


async def main() -> None:
    print(f'Connecting to agent at {AGENT_URL}...')
    try:
        async with httpx.AsyncClient(timeout=600) as httpx_client:
            client = await A2AClient.get_client_from_agent_card_url(
                httpx_client, AGENT_URL
            )
            print('✅ Connection successful.')
            await run_single_turn_test(client)
    except Exception as e:
        traceback.print_exc()
        print(f'❌ An error occurred: {e}')
        print('Make sure the agent is running and publicly reachable.')


if __name__ == '__main__':
    import asyncio
    asyncio.run(main())
