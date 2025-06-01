import os
import re
import json
import httpx
import uuid
import asyncio
import inspect
import logging
import importlib
import subprocess
from functools import wraps
from dotenv import load_dotenv
from langsmith import traceable
from pydantic import BaseModel, Field, ValidationError, validator
from typing_extensions import TypedDict
from langchain_core.documents import Document
from langchain_core.messages import ToolMessage, BaseMessage
from langchain.tools import Tool, StructuredTool
from langgraph.graph.message import add_messages
from langchain_core.vectorstores import InMemoryVectorStore
from typing import Dict, Any, List, Optional, Union, Annotated
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt.tool_node import tools_condition, ToolNode
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class GraphState(TypedDict):
    """Improved state tracking for LangGraph."""
    messages: Annotated[list[BaseMessage], add_messages]
    selected_tools: Optional[list[str]]  # Tools selected by LLM
    used_tools: list[str]  # Tools already called in this session
    context: dict  # Any additional context
    file_path: Optional[str]
    run_mode: Optional[str]  # "start" or "continue"

def load_local_tools_from_folder(folder_path: str) -> List[Tool]:
    """Loads tools from a local folder."""
    local_tools = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".py") and not filename.startswith("__"):
            module_name = filename[:-3]
            try:
                module = importlib.import_module(f"{folder_path}.{module_name}")
                for name, obj in inspect.getmembers(module):
                    if isinstance(obj, Tool):
                        wrapped = wrap_dict_input_tool(obj)
                        local_tools.append(wrapped)
                        print(f"✅ Loaded local tool: {wrapped.name}")
                    elif isinstance(obj, StructuredTool):
                        local_tools.append(obj)
                        print(f"✅ Loaded structured tool: {obj.name}")
            except Exception as e:
                print(f"❌ Failed to import {module_name}: {e}")
    return local_tools

def wrap_dict_input_tool(tool_obj: Tool) -> Tool:
    """Wraps a tool function to handle string or dict input."""
    original_func = tool_obj.func

    @wraps(original_func)
    def wrapper(input_value):
        if isinstance(input_value, str):
            input_value = {"ip": input_value}
        elif isinstance(input_value, dict) and "ip" not in input_value:
            logger.warning(f"⚠️ Missing 'ip' key in dict: {input_value}")
        return original_func(input_value)

    return Tool(
        name=tool_obj.name,
        description=tool_obj.description,
        func=wrapper,
    )

def schema_to_pydantic_model(name: str, schema: dict):
    """Dynamically creates a Pydantic model class from a JSON Schema."""
    from typing import Any, List, Dict, Optional
    namespace = {"__annotations__": {}}

    if schema.get("type") != "object":
        raise ValueError("Only object schemas are supported.")

    properties = schema.get("properties", {})
    required_fields = set(schema.get("required", []))

    for field_name, field_schema in properties.items():
        json_type = field_schema.get("type", "string")
        is_optional = field_name not in required_fields

        if json_type == "string":
            field_type = str
        # ... (other simple types: integer, number, boolean) ...
        elif json_type == "boolean":
             field_type = bool
        elif json_type == "array":
            items_schema = field_schema.get("items")
            if not items_schema:
                logger.warning(f"⚠️ Skipping field '{field_name}' (array missing 'items')")
                continue
            
            if "$ref" in items_schema:
                ref = items_schema["$ref"]
                ref_name = ref.split("/")[-1]
                ref_schema = schema.get("$defs", {}).get(ref_name)
                if ref_schema:
                    item_model = schema_to_pydantic_model(f"{name}_{field_name}_Item", ref_schema)
                    field_type = List[item_model]
                else:
                    logger.warning(f"⚠️ Could not resolve $ref for {ref}")
                    field_type = List[Dict[str, Any]]

            elif items_schema.get("type") == "object":
                # Handle inline object definition
                if "properties" in items_schema and items_schema["properties"]:
                    item_model = schema_to_pydantic_model(f"{name}_{field_name}_Item", items_schema)
                    field_type = List[item_model]
                else:
                    field_type = List[Dict[str, Any]]

            elif items_schema.get("type") == "string":
                field_type = List[str]
            elif items_schema.get("type") == "integer":
                field_type = List[int]
            elif items_schema.get("type") == "number":
                field_type = List[float]
            elif items_schema.get("type") == "boolean":
                field_type = List[bool]
            else:
                field_type = List[Any]

        elif json_type == "object":
            if "properties" in field_schema:
                nested_model = schema_to_pydantic_model(name + "_" + field_name, field_schema)
                field_type = nested_model
            elif "$ref" in field_schema:
                ref = field_schema["$ref"]
                ref_name = ref.split("/")[-1]
                ref_schema = schema["$defs"].get(ref_name)
                if ref_schema:
                    nested_model = schema_to_pydantic_model(name + "_" + ref_name, ref_schema)
                    field_type = nested_model
                else:
                    logger.warning(f"⚠️ Could not resolve $ref for {ref}")
                    field_type = Dict[str, Any]
            else:
                field_type = Dict[str, Any]

        else: # Handle Any type
            field_type = Any

        # ... (rest of the function: optional handling, adding to namespace) ...
        if is_optional:
            field_type = Optional[field_type]

        namespace["__annotations__"][field_name] = field_type
        if field_name in required_fields:
            namespace[field_name] = Field(...)
        else:
            namespace[field_name] = Field(default=None)

    return type(name, (BaseModel,), namespace)

def summarize_recent_tool_outputs(context: dict, limit: int = 3) -> str:
    summaries = []
    for key, val in list(context.items())[-limit:]:
        if isinstance(val, (str, dict, list)):
            preview = str(val)[:500]  # Avoid large dumps
            summaries.append(f"- {key}: {preview}")
    return "\n".join(summaries)

async def call_drawio_mcp_http(method_name: str, url: str, arguments: dict = None):
    request_id = str(uuid.uuid4())
    payload = {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method_name,
        "params": arguments or {}
    }

    logger.info(f"📤 Calling Draw.io MCP via HTTP: {method_name} at {url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
            logger.info(f"✅ MCP Response: {result}")
            return result
        except httpx.HTTPError as e:
            logger.error(f"❌ MCP HTTP Error: {e}")
            return None

class MCPToolDiscovery:
    """Discovers and calls tools in MCP containers."""
    def __init__(self, container_name: str, command: List[str], discovery_method: str = "tools/discover",
                 call_method: str = "tools/call"):
        self.container_name = container_name
        self.command = command
        self.discovery_method = discovery_method
        self.call_method = call_method
        self.discovered_tools = []

    @traceable
    async def discover_tools(self, timeout=30.0) -> List[Dict[str, Any]]:
        """Discovers tools from the MCP container or HTTP endpoint."""
        print(f"🔍 Discovering tools from container: {self.container_name}")
        print(f"🕵️ Discovery Method: {self.discovery_method}")

        # --- Handle HTTP-based discovery ---
        if isinstance(self.command, str) and self.command.startswith("http"):
            url = self.command  # Full URL passed as the "command"
            try:
                result = await call_drawio_mcp_http(self.discovery_method, url=url)
                if not result:
                    logger.error("❌ No response received from HTTP MCP")
                    return []
                if "result" not in result:
                    logger.warning(f"⚠️ Unexpected HTTP response format: {result}")
                    return []
                tools = result["result"]
                print("✅ Discovered tools:", [tool.get("name", "Unnamed Tool") for tool in tools])
                self.discovered_tools = tools
                return tools
            except Exception as e:
                logger.error(f"❌ HTTP discovery error: {e}", exc_info=True)
                return []

        # --- Handle STDIO-based (subprocess) discovery ---
        if not isinstance(self.command, list):
            logger.error(f"❌ Invalid command type: {type(self.command)}. Must be list or HTTP string.")
            return []

        discovery_payload = {
            "jsonrpc": "2.0",
            "method": self.discovery_method,
            "params": {},
            "id": str(uuid.uuid4())
        }
        payload_bytes = json.dumps(discovery_payload).encode() + b"\n"
        logger.debug(f"Sending discovery payload: {discovery_payload}")

        command = ["docker", "exec", "-i", self.container_name] + self.command

        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            process.stdin.write(payload_bytes)
            await process.stdin.drain()
            process.stdin.write_eof()

            try:
                response_line_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
                response_line = response_line_bytes.decode().strip() if response_line_bytes else None
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Discovery timed out after {timeout}s for {self.container_name}")
                process.kill()
                await process.wait()
                return []

            if not response_line:
                logger.error("❌ No response line received from stdout.")
                stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                logger.error(f"Associated stderr: {stderr_data.decode()}")
                return []

            response = json.loads(response_line)

            if "error" in response:
                logger.error(f"🚨 Discovery error: {response['error']}")
                return []
            elif "result" in response:
                result_data = response["result"]
                tools = result_data if isinstance(result_data, list) else result_data.get("tools", [])
                print("✅ Discovered tools:", [tool.get("name", "Unnamed Tool") for tool in tools])
                self.discovered_tools = tools
                return tools
            else:
                logger.warning(f"⚠️ Unexpected JSON structure: {response}")
                return []

        except Exception as e:
            logger.error(f"❌ STDIO discovery exception: {e}", exc_info=True)
            return []
        finally:
            if 'process' in locals() and process.returncode is None:
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
    @traceable
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any], timeout=60.0):
        """Calls a tool in the MCP container (HTTP or STDIO)."""
        logger.info(f"🔍 Attempting to call tool: {tool_name}")
        logger.info(f"📦 Argument Keys: {list(arguments.keys()) if isinstance(arguments, dict) else type(arguments)}")

        # --- Handle HTTP-based tool call ---
        if isinstance(self.command, str) and self.command.startswith("http"):
            url = self.command
            payload = {
                "jsonrpc": "2.0",
                "method": self.call_method,
                "params": {"name": tool_name, "arguments": arguments},
                "id": str(uuid.uuid4())
            }
            try:
                response = await call_drawio_mcp_http(self.call_method, url, payload)
                if not response:
                    logger.error("❌ No response from HTTP MCP call")
                    return {"error": "No response"}
                if "error" in response:
                    logger.error(f"🚨 HTTP tool call error: {response['error']}")
                    return {"error": response["error"]}
                return response.get("result", {})
            except Exception as e:
                logger.exception(f"❌ Exception during HTTP tool call to {tool_name}")
                return {"error": str(e)}

        # --- Handle STDIO-based tool call ---
        if not isinstance(self.command, list):
            logger.error(f"❌ Invalid command type: {type(self.command)}")
            return {"error": "Invalid command"}

        # Normalize special-case arguments (e.g., GitHub tools)
        normalized_args = arguments
        if tool_name == "create_or_update_file" and normalized_args.get("sha") is None:
            del normalized_args["sha"]

        payload = {
            "jsonrpc": "2.0",
            "method": self.call_method,
            "params": {"name": tool_name, "arguments": normalized_args},
            "id": str(uuid.uuid4())
        }
        payload_bytes = json.dumps(payload).encode() + b"\n"
        command = ["docker", "exec", "-i", self.container_name] + self.command

        try:
            logger.info(f"🚀 Sending payload to {tool_name} via STDIO")
            process = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env={**os.environ, "PYTHONUNBUFFERED": "1"}
            )

            process.stdin.write(payload_bytes)
            await process.stdin.drain()
            process.stdin.write_eof()

            try:
                response_line_bytes = await asyncio.wait_for(process.stdout.readline(), timeout=timeout)
                response_line = response_line_bytes.decode().strip() if response_line_bytes else None
            except asyncio.TimeoutError:
                logger.error(f"⏱️ Timeout after {timeout}s calling tool {tool_name}")
                process.kill()
                await process.wait()
                return {"error": f"Timeout after {timeout} seconds"}

            if not response_line:
                logger.error("❌ No response from stdout")
                stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                return {"error": "No response", "stderr": stderr_data.decode()}

            logger.info(f"🔬 Raw Response Line Received: {response_line}")
            try:
                response = json.loads(response_line)
            except json.JSONDecodeError:
                stderr_data = await asyncio.wait_for(process.stderr.read(), timeout=1.0)
                logger.error("❌ JSON Decode Error")
                return {"error": "JSON Decode Error", "stderr": stderr_data.decode(), "raw": response_line}

            if "error" in response:
                logger.error(f"🚨 Tool error: {response['error']}")
                return {"error": response["error"]}
            elif "result" in response:
                return response["result"]
            else:
                logger.warning("⚠️ Unexpected response shape")
                return response

        except Exception as e:
            logger.critical(f"🔥 Exception in tool call to {tool_name}", exc_info=True)
            return {"error": str(e)}
        finally:
            if process and process.returncode is None:
                logger.info(f"🧹 Cleaning up subprocess for {tool_name}")
                try:
                    process.kill()
                except ProcessLookupError:
                    pass
                await process.wait()
            
@traceable
async def get_tools_for_service(service_name, command, discovery_method, call_method, service_discoveries):
    """Enhanced tool discovery for each service."""
    print(f"🕵️ Discovering tools for: {service_name}")
    discovery = MCPToolDiscovery(
        container_name=service_name,
        command=command,
        discovery_method=discovery_method,
        call_method=call_method
    )
    service_discoveries[service_name] = discovery  # Store for future tool calls

    tools = []
    try:
        discovered_tools = await discovery.discover_tools()
        print(f"🛠️ Tools for {service_name}: {[t.get('name', 'Unnamed') for t in discovered_tools]}")

        for tool_info in discovered_tools: # Renamed 'tool' to 'tool_info' to avoid clash
            tool_name = tool_info["name"]
            tool_description = tool_info.get("description", "")
            # Handle potential variations in schema key name
            tool_schema = tool_info.get("inputSchema") or tool_info.get("parameters", {})

            if tool_schema and isinstance(tool_schema, dict) and tool_schema.get("type") == "object":
                try:
                    # Dynamically create the Pydantic model for input validation
                    input_model = schema_to_pydantic_model(tool_name + "_Input", tool_schema)

                    # *** MODIFICATION START ***
                    # Define an async wrapper function that filters None before validation
                    async def tool_call_async_wrapper(captured_service_name=service_name, captured_tool_name=tool_name, captured_input_model=input_model, **kwargs):
                        # Filter out arguments where the value is None
                        # This allows Pydantic defaults to apply correctly for missing keys
                        filtered_kwargs = {k: v for k, v in kwargs.items() if v is not None}
                        logger.debug(f"Original kwargs for {captured_tool_name}: {kwargs}")
                        logger.debug(f"Filtered kwargs for {captured_tool_name}: {filtered_kwargs}")

                        try:
                            # Validate arguments using the Pydantic model with filtered input
                            # 👇 Add this block before calling input_model(**filtered_kwargs)
                            if captured_tool_name == "download_chart":
                                if "type" in filtered_kwargs:
                                    chart_keys = ["type", "labels", "datasets", "options", "title"]
                                    config = {k: filtered_kwargs[k] for k in chart_keys if k in filtered_kwargs}
                                    output_path = filtered_kwargs.get("outputPath", "/output/chart.png")
                                    filtered_kwargs = {"config": config, "outputPath": output_path}    

                            logger.info(f"📥 Calling tool '{captured_tool_name}' from service '{captured_service_name}' with validated args: {filtered_kwargs}")
                            validated_args = captured_input_model(**filtered_kwargs).dict()
                            # Call the actual tool execution logic (which runs in a subprocess)
                            return await service_discoveries[captured_service_name].call_tool(captured_tool_name, validated_args, timeout=120) # Increased default timeout
                        
                        except ValidationError as e:
                            # If validation fails even after filtering (e.g., missing required field)
                            logger.error(f"Pydantic validation failed for {captured_tool_name}: {e}")
                            # Return an error string compatible with ToolMessage content handling
                            return f"Tool Input Validation Error: {e}"
                        except Exception as e:
                            # Catch other unexpected errors during validation or the call setup
                            logger.error(f"Unexpected error in tool wrapper for {captured_tool_name}: {e}", exc_info=True)
                            return f"Tool Wrapper Error: {e}"

                    # Create the StructuredTool, passing the async wrapper directly
                    structured_tool = StructuredTool.from_function(
                        name=tool_name,
                        description=tool_description,
                        args_schema=input_model,
                        coroutine=tool_call_async_wrapper # Use the coroutine parameter
                        # func= THIS IS NOT NEEDED WHEN USING coroutine=
                    )
                    # *** MODIFICATION END ***

                    tools.append(structured_tool)
                    logger.info(f"✅ Created StructuredTool: {tool_name}")

                except Exception as e:
                    # Catch errors during Pydantic model creation or StructuredTool setup
                    logger.warning(f"⚠️ Failed to build structured tool {tool_name}: {e}", exc_info=True)
                    # Optionally add a fallback simple tool here if needed
            else:
                 # --- Fallback logic for non-structured tools ---
                 # Ensure this part is compatible with your async setup if you use it.
                 # The asyncio.run call here can be problematic if the main loop is async.
                 logger.warning(f"⚠️ Tool '{tool_name}' has no valid object schema. Creating basic Tool.")

                 async def fallback_tool_call_wrapper(arg_input, captured_service_name=service_name, captured_tool_name=tool_name):
                     # Basic tools often expect a single string or a simple dict.
                     # Adjust the dict structure if needed based on how your basic tools work.
                     tool_args = {"input": arg_input} if isinstance(arg_input, str) else arg_input
                     if not isinstance(tool_args, dict): # Ensure it's a dict for call_tool
                         tool_args = {"input": str(tool_args)}
                     return await service_discoveries[captured_service_name].call_tool(captured_tool_name, tool_args)

                 # If your main application runs with asyncio.run(run_cli_interaction()),
                 # you should await the async function directly instead of using asyncio.run inside the lambda.
                 # However, Langchain's Tool expects a sync func. A common workaround is needed if the main loop is async.
                 # For now, assuming the asyncio.run might be acceptable in your specific context or needs adjustment later.
                 fallback_tool = Tool(
                     name=tool_name,
                     description=tool_description + " (Note: Accepts simplified input)",
                     # This lambda calls an async function using asyncio.run - check compatibility!
                     func=lambda x, tn=tool_name: asyncio.run(fallback_tool_call_wrapper(x, captured_tool_name=tn))
                     # Consider if a synchronous wrapper or adapter is needed here depending on main event loop.
                 )
                 tools.append(fallback_tool)


    except Exception as e:
        logger.error(f"❌ Tool discovery error in {service_name}: {e}", exc_info=True)

    return tools

embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

vector_store = InMemoryVectorStore(embedding=embedding)

@traceable
async def load_all_tools():
    """Async function to load tools from different MCP services and local files."""
    print("🚨 COMPREHENSIVE TOOL DISCOVERY STARTING 🚨")

    tool_services = [
        ("pyats-mcp", ["python3", "pyats_mcp_server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("github-mcp", ["node", "dist/index.js"], "list_tools", "call_tool"),
        # ("google-maps-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("sequentialthinking-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("slack-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("excalidraw-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("filesystem-mcp", ["node", "/app/dist/index.js", "/projects"], "tools/list", "tools/call"),
        # ("netbox-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("google-search-mcp", ["node", "/app/build/index.js"], "tools/list", "tools/call"),
        # ("servicenow-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("email-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),
        # ("chatgpt-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("quickchart-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),
        # ("vegalite-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("mermaid-mcp", ["node", "dist/index.js"], "tools/list", "tools/call"),
        # ("rfc-mcp", ["node", "build/index.js"], "tools/list", "tools/call"),    
        # ("nist-mcp", ["python3", "server.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("drawio-mcp", "http://host.docker.internal:11434/rpc", "tools/list", "tools/call"),
        # ("subnet-calculator-mcp", ["python3", "main.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("ise-mcp", ["python3", "main.py", "--oneshot"], "tools/discover", "tools/call"),
        # ("wikipedia-mcp", ["python3", "main.py"], "tools/discover", "tools/call"),
        # ("aci-mcp", ["python3", "main.py"], "tools/discover", "tools/call")
    ]

    try:
        # Run docker ps to verify containers
        docker_ps_result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
        print(docker_ps_result.stdout)

        service_discoveries = {}
        local_tools_lists  = await asyncio.gather(
            *[get_tools_for_service(service, command, discovery_method, call_method, service_discoveries)
              for service, command, discovery_method, call_method in tool_services]
        )

        all_tools = []
        for tools_list in local_tools_lists :
            all_tools.extend(tools_list)

        # ✅ Peer discovery inside this function
        # ✅ Peer discovery inside this function
        peer_agents = {}

        # ✅ Finalize tool sets
        local_tools = []
        for tools_list in local_tools_lists:
            local_tools.extend(tools_list)

        all_tools = local_tools
        
        # ✅ Index only local tools for tool selection
        tool_documents = [
            Document(
                page_content=f"Tool name: {tool.name}. Tool purpose: {tool.description}",
                metadata={"tool_name": tool.name}
            )
            for tool in all_tools if hasattr(tool, "description")
        ]
        vector_store.add_documents(tool_documents)

        return all_tools, local_tools

    except Exception as e:
        print(f"❌ CRITICAL TOOL DISCOVERY ERROR: {e}")
        import traceback
        traceback.print_exc()
        return []

# Load tools
all_tools, local_tools = asyncio.run(load_all_tools())

def format_tool_descriptions(tools: List[Tool]) -> str:
    return "\n".join(
        f"- `{tool.name}`: {tool.description or 'No description provided.'}"
        for tool in tools
    )

print("🔧 All bound tools:", [t.name for t in all_tools])

#llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.0)

llm = ChatOpenAI(model_name="gpt-4o", temperature="0.1")

llm_with_tools = llm.bind_tools(all_tools)

@traceable
class ContextAwareToolNode(ToolNode):
    """
    A specialized ToolNode that handles tool execution and updates the graph state
    based on the tool's response.  It assumes that tools return a dictionary.
    """

    async def ainvoke(
        self, state: GraphState, config: Optional[RunnableConfig] = None, **kwargs: Any
    ):
        """
        Executes the tool call specified in the last AIMessage, updates the state,
        and correctly formats the ToolMessage content for both success and error.
        """ # Updated docstring
        messages = state["messages"]
        last_message = messages[-1]

        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            raise ValueError("Expected an AIMessage with tool_calls in the last message.")

        tool_calls = last_message.tool_calls
        context = state.get("context", {})
        new_tool_messages = [] # Store new messages separately

        for tool_call in tool_calls:
            tool_name = tool_call['name']
            tool_call_id = tool_call['id'] # Get tool_call_id

            if not (tool := self.tools_by_name.get(tool_name)):
                logger.warning(
                    f"Tool '{tool_name}' requested by LLM not found in available tools. Skipping."
                )
                # Add a ToolMessage indicating the tool wasn't found
                new_tool_messages.append(ToolMessage(
                    tool_call_id=tool_call_id,
                    content=f"Error: Tool '{tool_name}' is not available.",
                    name=tool_name,
                ))
                continue

            tool_input = tool_call['args']
            # Ensure tool_input is a dictionary before filtering Nones
            if not isinstance(tool_input, dict):
                 logger.warning(f"Tool input for {tool_name} is not a dict: {tool_input}. Using as is or converting.")
                 # Attempt to handle non-dict input if necessary, or pass as is if tool expects non-dict
                 # For most structured tools, this indicates an LLM error
                 # Passing the raw input might cause downstream Pydantic errors in the tool wrapper
                 filtered_tool_input = tool_input # Or handle differently based on tool type
            else:
                 filtered_tool_input = {k: v for k, v in tool_input.items() if v is not None}

            logger.debug(f"Calling tool: {tool.name} with filtered args: {filtered_tool_input}")

            try:
                 # Invoke the tool (which now calls MCPToolDiscovery.call_tool)
                 # PATCH: Fix path for read_file to convert /output → /projects
                 if tool_name == "read_file" and isinstance(filtered_tool_input, dict):
                     # Accept either 'file_path' or 'path'
                     path_val = filtered_tool_input.pop("file_path", filtered_tool_input.get("path", ""))
                     if path_val.startswith("/output"):
                         path_val = path_val.replace("/output", "/projects")
                         logger.info(f"🔧 Remapped file path for read_file: /output → /projects → {path_val}")
                     filtered_tool_input["path"] = path_val  # overwrite normalized key

                 tool_response = await tool.ainvoke(filtered_tool_input, config=config) # Pass config
                 logger.info(f"Received response from tool {tool_name}: {type(tool_response)}")

                 # *** MODIFICATION START ***
                 tool_content_str = ""
                 is_error = False

                 if isinstance(tool_response, str) and (
                      tool_response.startswith("Error:") or
                      tool_response.startswith("Tool Error:") or
                      tool_response.startswith("Subprocess Error:") or
                      tool_response.startswith("Critical Framework Error:")
                 ):
                      # Handle specific error strings returned by call_tool
                      tool_content_str = tool_response
                      is_error = True
                      logger.error(f"Error reported by tool {tool_name}: {tool_content_str}")
                 elif isinstance(tool_response, (dict, list)):
                      # Handle successful JSON dict/list response
                      try:
                           # Attempt to dump complex structures cleanly
                           tool_content_str = json.dumps(tool_response)
                           # Update context only on success? Or always update? Let's update always for now.
                           context.update({tool_name: tool_response}) # Store structured result in context
                      except TypeError as e:
                           logger.warning(f"Could not JSON serialize tool response for {tool_name}: {e}. Using str().")
                           tool_content_str = str(tool_response)
                           context.update({tool_name: tool_content_str}) # Store string representation
                 else:
                      # Handle other types (simple strings, numbers, etc.)
                      tool_content_str = str(tool_response)
                      context.update({tool_name: tool_response}) # Store raw result

                 # Create ToolMessage with the determined content string
                 tool_message = ToolMessage(
                      tool_call_id=tool_call_id,
                      content=tool_content_str,
                      name=tool_name,
                 )
                 # *** MODIFICATION END ***

                 new_tool_messages.append(tool_message)

                 # Update used tools list
                 used = set(context.get("used_tools", []))
                 used.add(tool.name)
                 context["used_tools"] = list(used)

            except Exception as tool_exec_e:
                # Catch errors during the tool.ainvoke call itself (e.g., Pydantic validation within the wrapper)
                logger.error(f"Exception during tool.ainvoke for {tool_name}", exc_info=True)
                new_tool_messages.append(ToolMessage(
                     tool_call_id=tool_call_id,
                     content=f"Framework Error invoking tool {tool_name}: {tool_exec_e}",
                     name=tool_name,
                ))

        # Append all new messages at once
        messages.extend(new_tool_messages)

        # Decide the next step - always go back to handler which then goes to assistant
        return {
            "messages": messages,
            "context": context,
            # "__next__": "handle_tool_results" # This seems to be set by the graph edge already
        }
    
async def select_tools(state: GraphState):
    messages = state.get("messages", [])
    context = state.get("context", {})
    last_user_message = next((m for m in reversed(messages) if isinstance(m, HumanMessage)), None)

    if not last_user_message:
        logger.warning("select_tools: No user message found.")
        state["selected_tools"] = []
        return {"messages": messages, "context": context}

    query = last_user_message.content
    selected_tool_names = []

    try:
        # Step 1: Vector search
        scored_docs = vector_store.similarity_search_with_score(query, k=35)

        # Step 2: Apply threshold with fallback
        threshold = 0.50
        relevant_docs = [doc for doc, score in scored_docs if score >= threshold]

        if not relevant_docs:
            logger.warning(f"⚠️ No tools above threshold {threshold}. Falling back to top 5 by score.")
            relevant_docs = [doc for doc, _ in scored_docs[:15]]

        logger.info(f"✅ Selected {len(relevant_docs)} tools after filtering/fallback.")

        # Step 3: Build tool info for LLM
        tool_infos = {
            doc.metadata["tool_name"]: doc.page_content
            for doc in relevant_docs if "tool_name" in doc.metadata
        }

        if not tool_infos:
            logger.warning("select_tools: No valid tool_name metadata found.")
            state["selected_tools"] = []
            return {"messages": messages, "context": context}

        # Log top tools and scores for debugging
        logger.info("Top tools with scores:")
        for doc, score in scored_docs[:10]:
            if "tool_name" in doc.metadata:
                logger.info(f"- {doc.metadata['tool_name']}: {score}")

        tool_descriptions_for_prompt = "\n".join(
            f"- {name}: {desc}" for name, desc in tool_infos.items()
        )

        # Step 4: LLM refinement
        tool_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a precise Tool Selector Assistant. Your task is to choose the most relevant tools from the provided list to fulfill the user's request.

Consider these guidelines:
- Match tools to the *exact* user intent.
- Refer to tool descriptions to understand their purpose.
- Prefer specific tools over general ones if applicable.
- If multiple tools seem relevant for sequential steps *explicitly requested*, list them.
- If no tool is a good fit, output "None".
- Output *only* a comma-separated list of the chosen tool names (e.g., tool_a,tool_b) or the word "None"."""),

            ("human", "User request:\n---\n{query}\n---\n\nAvailable tools:\n---\n{tools}\n---\n\nBased *only* on the tools listed above, which are the best fit for the request? Output only the comma-separated tool names or 'None'.")
        ])

        selection_prompt_messages = tool_prompt.format_messages(
            query=query,
            tools=tool_descriptions_for_prompt
        )

        logger.info("🤖 Invoking LLM for tool selection...")
        tool_selection_response = await llm.ainvoke(selection_prompt_messages)
        raw_selection = tool_selection_response.content.strip()

        logger.info(f"📝 LLM raw tool selection: '{raw_selection}'")

        if raw_selection.lower() == "none" or not raw_selection:
            selected_tool_names = []
        else:
            potential_names = [name.strip() for name in raw_selection.split(',')]
            
            # 🔧 Normalize selection for delegated tools (e.g., ask_selector → ask_selector_via_xxx)
            normalized_tool_names = {}
            for name in tool_infos.keys():
                base_name = name.split("_via_")[0] if "_via_" in name else name
                normalized_tool_names[base_name] = name  # Always map shortest name → full name

            # Map LLM-chosen names to full tool names
            selected_tool_names = [
                normalized_tool_names.get(name, name)
                for name in potential_names
                if name in normalized_tool_names
            ]

            if len(selected_tool_names) != len(potential_names):
                logger.warning(f"⚠️ LLM selected invalid tools: {set(potential_names) - set(selected_tool_names)}")

    except Exception as e:
        logger.error(f"🔥 Error during tool selection: {e}", exc_info=True)
        selected_tool_names = []

    # Final: Update context
    context["selected_tools"] = list(set(context.get("selected_tools", [])) | set(selected_tool_names))
    logger.info(f"✅ Final selected tools: {context['selected_tools']}")
    return {
        "messages": messages,
        "context": context
    }


system_msg = """You are a computer networking expert at the CCIE level. You are a precise and helpful assistant with access to a wide range of tools for networking, GitHub automation, Slack notifications, file system operations, and ServiceNow ticketing. You must follow strict guidelines before choosing and using tools.

AVAILABLE TOOL CATEGORIES:
{tool_descriptions}

📌 TOOL USAGE GUIDELINES:

GENERAL RULES:
1. THINK step-by-step about what the user wants.
2. MATCH tools to the *exact* user intent.
3. DO NOT guess. Only use tools when the user explicitly requests an action that matches the tools purpose.
4. NEVER call a tool without all required parameters.
5. NEVER call a tool just because the output of another tool suggests a next step — unless the user explicitly asked for that.

🧠 CONTEXT MEMORY:
- You may use results of previously run tools by referring to their output in memory.
- For example, if you ran a tool like `ask_selector_via_agent2`, the result will be stored and available in memory under that name.
- When using follow-up tools like `email_send_message`, reuse this result as needed.

✅ WHEN TO USE TOOLS:

🧠 PYATS NETWORK AUTOMATION TOOLS:
- Use `pyATS_show_running_config`, `pyATS_run_show_command`, `pyATS_ping_from_network_device`, or `pyATS_configure_device` ONLY if the user requests network validation, inspection, or configuration of Cisco-style network devices.
- Do NOT use these tools for cloud or filesystem tasks.

📁 FILESYSTEM TOOLS:
- Use `write_file`, `edit_file`, `read_file`, or `create_directory` when the user asks to **create, modify, save, or read from files** in a local or mounted directory.
- Example: “Save the config to a markdown file” → `write_file`

🐙 GITHUB TOOLS:
- Use GitHub tools ONLY when the user explicitly asks to:
  - Push files
  - Create or update code or documentation in a repo
  - Open or manage GitHub issues or PRs
- Required for all GitHub actions: `owner`, `repo`, `branch`, and `commit message`
- NEVER use GitHub tools for local file management or Slack-style notifications.

💬 SLACK TOOLS:
- Use `slack_post_message`, `slack_reply_to_thread`, or `slack_add_reaction` only when the user asks to send messages to a Slack channel or thread.
- Example: “Notify the team” or “Send a message to #NOC” → `slack_post_message`

🗺️ MAPS TOOLS:
- Use `maps_geocode`, `maps_elevation`, etc., ONLY when the user asks for location-based data.
- NEVER use for IP addresses or configs.

📐 DIAGRAMMING TOOLS:
- Use `create_drawing`, `update_drawing`, `export_to_json` only when the user wants a network diagram or visual model.
- Do NOT export a drawing unless the user explicitly says so.

🧜 MERMAID DIAGRAM TOOLS:
- Use `mermaid_generate` ONLY when the user asks to create a PNG image from **Mermaid diagram code**.
  - **Purpose**: Converts Mermaid diagram code text into a PNG image file.
  - **Parameters:**
    - `code` (string): The Mermaid diagram code to render (required).
    - `theme` (string, optional): Theme for the diagram. Options: default, forest, dark, neutral. Defaults to default.
    - `backgroundColor` (string, optional): Background color for the generated PNG, e.g., white, transparent, #F0F0F0. Defaults to transparent or theme-based.
    - `name` (string): The filename for the generated PNG image (e.g., network_topology.png). **Required only if the tools environment is configured to save files to disk (CONTENT_IMAGE_SUPPORTED=false).**
    - `folder` (string): The absolute path *inside the container* where the image should be saved (e.g., /output). **Required only if the tools environment is configured to save files to disk (CONTENT_IMAGE_SUPPORTED=false).**
    - **Behavior Note:** This tools behavior depends on the `CONTENT_IMAGE_SUPPORTED` environment variable of the running container.
    - If `true` (default): The PNG image data is returned directly in the API response. `name` and `folder` parameters are ignored.
    - If `false`: The PNG image is saved to the specified `folder` with the specified `name`. The API response will contain the path to the saved file (e.g., /output/network_topology.png). `name` and `folder` parameters are **mandatory** in this mode.
    - DO NOT call this tool with a raw text string. The diagram field must be a structured JSON object that includes children, connections, title, cloudProvider, and layoutDirection.
    
🛠️ SERVICE NOW TOOLS:
- ONLY use ServiceNow tools if the user explicitly says things like:
  - “Create a problem ticket in ServiceNow”
  - “Get the state of a ServiceNow problem”
  - if asked to create a problem in service now - only call the create service now problem tool; not the other service now problem tools. You only need 1 tool to create a problem.
- NEVER use ServiceNow tools to write files, notify teams, or log internal info.
- NEVER assume a ServiceNow ticket is needed unless the user says so.
- ⚠️ If the user does NOT mention “ServiceNow” or “ticket,” DO NOT CALL ANY ServiceNow tool.

📧 EMAIL TOOLS:
- Use email tools (like `email_send_message`) ONLY when the user explicitly asks to send an email.
- Examples: "Send an email to team@example.com with the results", "Email the configuration to the network admin".
- Required: Recipient email address(es), subject line, and the body content for the email.
- Specify clearly who the email should be sent to and what information it should contain.
- DO NOT use email tools for Slack notifications, saving files, or internal logging unless specifically instructed to email that information.

🤖 CHATGPT ANALYSIS TOOLS:
- Use the `ask_chatgpt` tool ONLY when the user explicitly asks you to leverage an external ChatGPT model for specific analysis, summarization, comparison, or generation tasks that go beyond your primary function or require a separate perspective.
- Examples: "Analyze this Cisco config for security best practices using ChatGPT", "Ask ChatGPT to summarize this document", "Get ChatGPTs explanation for this routing behavior".
- Required: The `content` (e.g., configuration text, document snippet, specific question) that needs to be sent to the external ChatGPT tool.
- Clearly state *why* you are using the external ChatGPT tool (e.g., "To get a detailed security analysis from ChatGPT...").
- Do NOT use this tool for tasks you are expected to perform directly based on your core instructions or other available tools (like running a show command or saving a file). Differentiate between *your* analysis/response and the output requested *from* the external ChatGPT tool.

📊 VEGALITE VISUALIZATION TOOLS (Requires 2 Steps: Save then Visualize):
- Use these tools to create PNG charts from structured data (like parsed command output) using the Vega-Lite standard.

1.  **vegalite_save_data**
    - **Purpose**: Stores structured data under a unique name so it can be visualized later. This MUST be called *before* vegalite_visualize_data.
    - **Parameters**:
        - name (string): A unique identifier for this dataset (e.g., R1_interface_stats, packet_comparison). Choose a descriptive name.
        - data (List[Dict]): The actual structured data rows, formatted as a list of dictionaries. **CRITICAL: Ensure this data argument contains the *actual, non-empty* data extracted from previous steps (like pyATS output). Do NOT pass empty lists or lists of empty dictionaries.**
    - **Returns**: Confirmation that the data was saved successfully.

2.  **vegalite_visualize_data**
    - **Purpose**: Generates a PNG image visualization from data previously saved using vegalite_save_data. It uses a provided Vega-Lite JSON specification *template* and saves the resulting PNG to the /output directory.
    - **Parameters**:
        - data_name (string): The *exact* unique name that was used when calling vegalite_save_data.
        - vegalite_specification (string): A valid Vega-Lite v5 JSON specification string that defines the desired chart (marks, encodings, axes, etc.). **CRITICAL: This JSON string MUST NOT include the top-level data key.** The tool automatically loads the data referenced by data_name and injects it. The encodings within the spec (e.g., field, packets) must refer to keys present in the saved data.
    - **Returns**: Confirmation message including the container path where the PNG file was saved (e.g., /output/R1_interface_stats.png).

📈 QUICKCHART TOOLS (Generates Standard Chart Images/URLs):
- Use these tools for creating common chart types (bar, line, pie, etc.) using the QuickChart.io service. This requires constructing a valid Chart.js configuration object.

1.  **generate_chart**
    - **Purpose**: Creates a chart image hosted by QuickChart.io and returns a publicly accessible URL to that image. Use this when the user primarily needs a *link* to the visualization.
    - **Parameters**:
        - chart_config (dict or JSON string): A complete configuration object following the **Chart.js structure**. This object must define the chart type (e.g., bar, line, pie), the data (including labels and datasets with their values), and any desired options. Refer to Chart.js documentation for details on structuring this object. **CRITICAL: You must construct the full, valid Chart.js configuration based on the users request and available data.**
    - **Returns**: A string containing the URL pointing to the generated chart image.

2.  **download_chart**
    - **Purpose**: Creates a chart image using QuickChart.io and saves it directly as an image file (e.g., PNG) to the /output directory on the server. Use this when the user explicitly asks to **save the chart as a file**.
    - **Parameters**:
        - chart_config (dict or JSON string): The *same* complete Chart.js configuration object structure required by generate_chart. It defines the chart type, data, and options. **CRITICAL: You must construct the full, valid Chart.js configuration.**
        - file_path (string): The desired filename for the output image within the /output directory (e.g., interface_pie_chart.png, device_load.png). The tool automatically saves to the /output path.
    - **Returns**: Confirmation message including the container path where the chart image file was saved (e.g., /output/interface_pie_chart.png).

📜 RFC DOCUMENT TOOLS:
- Use `get_rfc`, `search_rfcs`, or `get_rfc_section` ONLY when the user explicitly asks to find, retrieve, or examine Request for Comments (RFC) documents.
- **Trigger Examples**:
    - Search for RFCs about HTTP/3 → `search_rfcs`
    - Get RFC 8446 or Show me the document for RFC 8446 → `get_rfc`
    - What's the metadata for RFC 2616?" → `get_rfc` with `format=metadata`
    - Find section 4.2 in RFC 791 or Get the 'Security Considerations' section of RFC 3550 → `get_rfc_section`
- **Constraints**:
    - Requires the specific RFC `number` for `get_rfc` and `get_rfc_section`.
    - Requires a `query` string for `search_rfcs`.
    - For `get_rfc_section`, requires a `section` identifier (title or number).
    - Do NOT use these tools for general web searches, code lookup, configuration files, or non-RFC standards documents. ONLY use for retrieving information directly related to official RFCs.

🛡️ NIST CVE VULNERABILITY TOOLS:
- Use `get_cve` or `search_cve` ONLY when the user explicitly asks to find or retrieve information about Common Vulnerabilities and Exposures (CVEs) from the NIST National Vulnerability Database (NVD).
- **Trigger Examples**:
    - Get details for CVE-2021-44228 or Tell me about the Log4Shell vulnerability CVE-2021-44228 → `get_cve` with `cve_id=CVE-2021-44228`
    - Search the NVD for vulnerabilities related to Apache Struts → `search_cve` with `keyword="Apache Struts"`
    - Find CVEs mentioning 'Microsoft Exchange Server' exactly → `search_cve` with `keyword="Microsoft Exchange Server"` and `exact_match=True`
    - Give me a concise summary of CVE-2019-1010218 → `get_cve` with `cve_id="CVE-2019-1010218"` and `concise=True`
    - Show me the latest 5 vulnerabilities for 'Cisco IOS XE' → `search_cve` with `keyword=Cisco IOS XE` and `results=5`
- **Constraints**:
    - Requires a valid CVE ID (e.g., `CVE-YYYY-NNNNN`) for `get_cve`.
    - Requires a `keyword` string for `search_cve`.
    - Use the `concise` parameter only if the user asks for summary information.
    - Use the `exact_match` parameter for `search_cve` only if the user specifies needing an exact phrase match.
    - Do NOT use these tools for general security advice, threat hunting outside of NVD, retrieving non-CVE vulnerability info, or fetching software patches. They are ONLY for interacting with the NIST NVD CVE database.

── DRAW.IO MCP TOOLS ────────────────────────────────────────────────

• add-rectangle
Creates a standard rectangle shape on the current page.
Parameters (JSON):
- text (string) – Label displayed inside the rectangle
- width (number) – Width in pixels
- height (number) – Height in pixels
- color (string, optional) – Fill color (e.g., "#FF0000" or "#0000FF")
- x (number, optional) – X offset from center
- y (number, optional) – Y offset from center
Returns:
{{ "success": true, "cellId": "<ID>" }}
Tip: Omit x and y to center the shape automatically.

• add-edge
Connects two existing cells with a labeled directional connector.
Parameters (JSON):
- source_id (string) – ID of the starting cell
- target_id (string) – ID of the ending cell
- text (string, optional) – Optional label for the edge
Returns:
{{ "success": true, "edgeId": "<ID>" }}

• add-cell-of-shape
Inserts a shape from the Draw.io library (used for non-rectangle components).
Parameters (JSON):
- shape_name (string) – Name of the shape (e.g., "Router", "Cloud")
- x, y (number) – Coordinates
- width, height (number) – Dimensions
Returns:
{{ "success": true, "cellId": "<ID>" }}

• get-all-cells-detailed
Retrieves every object on the canvas including all shapes, edges, interface labels, VLAN annotations, and floating text.
Parameters:
None
Returns:
An array of cell objects. Each includes:
- id: Unique Draw.io cell ID
- type: "vertex" for shapes, "edge" for connectors
- text: Extracted human-readable label (from value, label, or content)
- parsedLines: Array of individual text lines split from the label
- rawValue: The original .value (if a string), otherwise null
- attributes: All XML attributes on the node (if present)
- style: Style string used by Draw.io (e.g., fillColor, shape, endArrow)
- geometry: Object: {{ x, y, width, height }}
- vertex: Boolean flag: true if this is a shape
- edge: Boolean flag: true if this is a connector
- parent: ID of the parent group/layer (usually "1")
- source: ID of the source cell (only for edges)
- target: ID of the target cell (only for edges)

🧮 SUBNET CALCULATOR TOOL
Use the calculate_subnet tool only when the user asks for subnetting information or IP network calculations based on a given CIDR notation (e.g., 192.168.1.0/24).

✅ Trigger Examples:
“Calculate the subnet details for 10.0.0.0/16”
“Give me the network, broadcast, and usable IPs for 192.168.10.0/24”
“How many usable hosts are in 172.16.0.0/20?”
“Show me the usable host range for 192.168.100.0/25”

🔧 Tool Parameters:
cidr (string): A valid CIDR block in the format IP/Mask, e.g., 192.168.1.0/24.

📦 Output Includes:
network_address: Base network address
broadcast_address: Broadcast address for the subnet
netmask: Subnet mask in dotted-decimal format
wildcard_mask: Wildcard mask (inverted netmask)
usable_host_range: First to last usable host IPs (if applicable)
number_of_usable_hosts: Total count of usable IPs

🚫 Do NOT use this tool:
Without a valid CIDR input
For converting host IPs, MAC addresses, or general networking explanations
To validate subnets that are not CIDR formatted


── GUIDELINES ─────────────────────────────────────────────────────  
1. **Always think first**: explain why you’re choosing a tool.  
2. **Match exactly**: call only `add-rectangle` or `add-edge` for Draw.io shapes/lines.  
3. **Wait for success**: parse the returned `cellId` from `add-rectangle` before calling `add-edge`.  
4. **Recover from timeouts**: if you see a timeout error, pause and retry the same call once.  
5. **No guessing**: if you don’t have an ID yet, ask for clarification instead of defaulting to a second call.   

🎯 TOOL CHAINING:
- Do NOT chain tools together unless the user clearly describes multiple steps.
  - Example: “Save the config to GitHub and notify Slack” → You may use two tools.
- Otherwise, assume single-tool usage unless explicitly stated.

🧠 BEFORE YOU ACT:
- Pause and explain your thought process.
- Say WHY the tool youre selecting is the best fit.
- If unsure, respond with a clarification question instead of calling a tool.

🤝 A2A PEER AGENT DELEGATION TOOL:
- Use the `delegate_task_to_peer_agent` tool ONLY in the following situations:
    1. You determine you lack the necessary local tool to fulfill a specific user request (e.g., asked about a system you don't manage).
    2. A local tool execution fails with an error indicating the target is not found or not managed by this agent.
    3. The user explicitly asks you to consult or delegate the task to another specific agent.
- **Required Parameters:**
    - `peer_agent_url`: The full base URL of the agent to contact (e.g., "http://agent2-hostname:10001"). You might need to ask the user for this if it's not obvious or provided. **Known Peer Agents: [List known peer agent URLs here if applicable, e.g., Agent2 at http://<agent2-ip-or-hostname>:<port>]**
    - `task_description`: The precise question or task description to send to the peer agent. Usually, this will be the relevant part of the user's original request.
- **Example:** If the user asks "Ask selector device health for device S3" and your local `pyATS_...` tool fails because S3 is unknown, you should explain this and then call `delegate_task_to_peer_agent` with `peer_agent_url='http://<agent2_url>'` and `task_description='Ask selector device health for device S3'`.
- **DO NOT** use this tool for tasks you *can* perform locally.

"""


@traceable
async def assistant(state: GraphState):
    """Handles assistant logic and LLM interaction, with support for sequential tool calls and uploaded file processing."""
    messages = state.get("messages", [])
    context = state.get("context", {})
    selected_tool_names = context.get("selected_tools", [])
    run_mode = context.get("run_mode", "start")

    used = set(context.get("used_tools", []))
    # If selected_tool_names is empty, fall back to ALL tools not already used
    if selected_tool_names:
        tools_to_use = [
            tool for tool in all_tools 
            if tool.name in selected_tool_names and tool.name not in used
        ]
    else:
        # Broaden scope — allow Gemini to pick missed tools (Slack, GitHub, etc.)
        tools_to_use = [
            tool for tool in all_tools 
            if tool.name not in used
        ]

    # 👇 STEP 3: Handle uploaded files from context
    uploaded_files = context.get("metadata", {}).get("uploaded_files", [])
    file_contexts = []
    for path in uploaded_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            file_contexts.append(f"--- {os.path.basename(path)} ---\n{content}\n")
        except Exception as e:
            file_contexts.append(f"Could not read file {path}: {e}")

    # If we're in continuous mode, don't re-select tools
    if run_mode == "continue":
        last_tool_message = None
        for msg in reversed(messages):
            if isinstance(msg, ToolMessage):
                last_tool_message = msg
                break

        if last_tool_message:
            new_messages = [SystemMessage(content=system_msg)] + messages

            llm_with_tools = llm.bind_tools(tools_to_use)
            response = await llm_with_tools.ainvoke(new_messages, config={"tool_choice": "auto"})

            if hasattr(response, "tool_calls") and response.tool_calls:
                return {"messages": [response], "context": context, "__next__": "tools"}
            else:
                return {"messages": [response], "context": context, "__next__": "__end__"}

    # Initial processing or starting a new sequence
    llm_with_tools = llm.bind_tools(tools_to_use)
    formatted_tool_descriptions = format_tool_descriptions(tools_to_use)
    formatted_system_msg = system_msg.format(tool_descriptions=formatted_tool_descriptions)
    context_summary = summarize_recent_tool_outputs(context)

    if context_summary:
        formatted_system_msg += f"\n\n📅 RECENT TOOL RESPONSES:\n{context_summary}"

    if file_contexts:
        formatted_system_msg += f"\n\n📎 UPLOADED FILES:\n{''.join(file_contexts)}"

    new_messages = [SystemMessage(content=formatted_system_msg)] + messages

    try:
        logger.info(f"assistant: Invoking LLM with new_messages: {new_messages}")
        response = await llm_with_tools.ainvoke(new_messages, config={"tool_choice": "auto"})
        logger.info(f"Raw LLM Response: {response}")

        if not isinstance(response, AIMessage):
            response = AIMessage(content=str(response))
    except Exception as e:
        logger.error(f"Error invoking LLM: {e}", exc_info=True)
        response = AIMessage(content=f"LLM Error: {e}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        context["run_mode"] = "continue"
        return {"messages": [response], "context": context, "__next__": "tools"}
    else:
        context["run_mode"] = "start"
        return {"messages": [response], "context": context, "__next__": "__end__"}

@traceable
async def handle_tool_results(state: GraphState):
    messages = state.get("messages", [])
    context = state.get("context", {})
    run_mode = context.get("run_mode", "start")

    # 🛠 Normalize tool messages: Fix any "model" role into "agent"
    normalized_messages = []
    for m in messages:
        if isinstance(m, dict):
            if m.get("role") == "model":
                m["role"] = "agent"
            normalized_messages.append(m)
        else:
            normalized_messages.append(m)

    # Always reset run_mode to prevent infinite loops unless LLM explicitly continues
    context["run_mode"] = "start"

    return {
        "messages": normalized_messages,
        "context": context,
        "__next__": "assistant"
    }

# Graph setup
graph_builder = StateGraph(GraphState)

# Define core nodes
graph_builder.add_node("select_tools", select_tools)
graph_builder.add_node("assistant", assistant)
graph_builder.add_node("tools", ContextAwareToolNode(tools=all_tools))
graph_builder.add_node("handle_tool_results", handle_tool_results)

# Define clean and minimal edges
# Start flow
graph_builder.add_edge(START, "select_tools")

# After tool selection, go to assistant
graph_builder.add_edge("select_tools", "assistant")

# Assistant decides: use tool or end
graph_builder.add_conditional_edges(
    "assistant",
    lambda state: state.get("__next__", "__end__"),
    {
        "tools": "tools",
        "__end__": END,
    }
)

# Tools always go to handler
graph_builder.add_edge("tools", "handle_tool_results")

# Tool results always return to assistant
graph_builder.add_edge("handle_tool_results", "assistant")

# Compile graph
compiled_graph = graph_builder.compile()

async def run_cli_interaction():
    """Runs the CLI interaction loop."""
    state = {"messages": [], "context": {"used_tools": []}}
    print("🛠️ Available tools:", [tool.name for tool in all_tools])
    while True:
        user_input = input("User: ")
        if user_input.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break

        user_message = HumanMessage(content=user_input)
        state["messages"].append(user_message)
        state["context"]["used_tools"] = []

        print("🚀 Invoking graph...")
        result = await compiled_graph.ainvoke(state, config={"recursion_limit": 100})
        state = result

        for message in reversed(state["messages"]):
            if isinstance(message, AIMessage):
                print("Assistant:", message.content)
                break

if __name__ == "__main__":
    asyncio.run(run_cli_interaction())
