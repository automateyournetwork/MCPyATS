# pyats_mcp_server.py

import os
import re
import string
import sys
import json
import logging
import argparse
import textwrap
import threading # Added
import time      # Added
from pyats.topology import loader
from genie.libs.parser.utils import get_parser
from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError
from typing import List, Dict, Any, Optional
import asyncio
from functools import partial

# --- Basic Logging Setup ---
# Add thread name to logging format
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(threadName)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("PyatsMCPServer")

# --- Load Environment Variables ---
load_dotenv()
TESTBED_PATH = os.getenv("PYATS_TESTBED_PATH")

if not TESTBED_PATH or not os.path.exists(TESTBED_PATH):
    logger.critical(f"❌ CRITICAL: PYATS_TESTBED_PATH environment variable not set or file not found: {TESTBED_PATH}")
    sys.exit(1) # Exit immediately if testbed is missing

logger.info(f"✅ Using testbed file: {TESTBED_PATH}")

# --- Pydantic Models for Input Validation (Keep as is) ---
class DeviceCommandInput(BaseModel):
    device_name: str = Field(..., title="Device Name", description="The name of the device in the testbed.")
    command: str = Field(..., title="Command", description="The command to execute (e.g., 'show ip interface brief', 'ping 8.8.8.8').")

class ConfigInput(BaseModel):
    device_name: str = Field(..., title="Device Name", description="The name of the device in the testbed.")
    config_commands: str = Field(..., title="Config Commands", description="Single or multi-line configuration commands.")

class DeviceOnlyInput(BaseModel):
    device_name: str = Field(..., title="Device Name", description="The name of the device in the testbed.")

class LinuxCommandInput(BaseModel):
    device_name: str = Field(..., title="Linux Device Name", description="The name of the Linux device in the testbed.")
    command: str = Field(..., title="Linux Command", description="Linux command to execute (e.g., 'ifconfig', 'ls -l /home')")
    
def wrap_output_for_clients(output: Any) -> Dict[str, Any]:
    """Ensure VS Code-friendly response with 'content', while keeping 'text' and 'json' for clients."""
    if isinstance(output, str):
        return {
            "text": output,
            "raw": output,
            "content": [{"kind": "text", "text": output}]
        }
    if isinstance(output, dict):
        text_output = json.dumps(output, indent=2)
        return {
            "text": text_output,
            "json": output,
            "content": [{"kind": "text", "text": text_output}]
        }
    return {
        "text": str(output),
        "content": [{"kind": "text", "text": str(output)}]
    }

def _get_device(device_name: str):
    """Helper to load testbed and get/connect to a device, ensuring enable mode."""
    try:
        testbed = loader.load(TESTBED_PATH)
        device = testbed.devices.get(device_name)
        if not device:
            raise ValueError(f"Device '{device_name}' not found in testbed '{TESTBED_PATH}'.")

        if not device.is_connected():
            logger.info(f"Connecting to {device_name}...")
            device.connect(
                connection_timeout=120,
                learn_hostname=True,
                log_stdout=False,
                mit=True
            )
            logger.info(f"Connected to {device_name}")

        return device

    except Exception as e:
        logger.error(f"Error getting/connecting to device {device_name}: {e}", exc_info=True)
        raise

def _disconnect_device(device):
    """Helper to safely disconnect."""
    if device and device.is_connected():
        logger.info(f"Disconnecting from {device.name}...")
        try:
            device.disconnect()
            logger.info(f"Disconnected from {device.name}")
        except Exception as e:
             logger.warning(f"Error disconnecting from {device.name}: {e}")


def run_show_command(params: dict) -> dict:
    try:
        validated_input = DeviceCommandInput(**params)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for run_show_command: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}
    device = None
    try:
        disallowed_modifiers = ['|', 'include', 'exclude', 'begin', 'redirect', '>', '<', 'config', 'copy', 'delete', 'erase', 'reload', 'write']
        command_lower = validated_input.command.lower().strip()
        if not command_lower.startswith("show"):
            return {"status": "error", "error": f"Command '{validated_input.command}' is not a 'show' command."}
        for part in command_lower.split():
            if part in disallowed_modifiers:
                return {"status": "error", "error": f"Command '{validated_input.command}' contains disallowed term '{part}'."}

        device = _get_device(validated_input.device_name)

        try:
            logger.info(f"Attempting to parse command: '{validated_input.command}' on {validated_input.device_name}")
            parsed_output = device.parse(validated_input.command)
            logger.info(f"Successfully parsed output for '{validated_input.command}' on {validated_input.device_name}")
            return parsed_output  # Pass back exactly what the pyATS parser returns
        except Exception as parse_exc:
            logger.warning(f"Parsing failed for '{validated_input.command}' on {validated_input.device_name}: {parse_exc}. Falling back to execute.")
            raw_output = device.execute(validated_input.command)
            logger.info(f"Executed command (fallback): '{validated_input.command}' on {validated_input.device_name}")
            return {"output": str(raw_output)}
    except Exception as e:
        logger.error(f"Error in run_show_command for {validated_input.device_name}: {e}", exc_info=True)
        if "Authentication failed" in str(e) or "Timeout connecting" in str(e):
            return {"status": "error", "error": f"Connection/Auth Error: {e}"}
        return {"status": "error", "error": f"Execution error: {e}"}
    finally:
        _disconnect_device(device)


def apply_device_configuration(params: dict) -> dict:
    try:
        validated_input = ConfigInput(**params)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for apply_device_configuration: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}
    device = None
    try:
        device = _get_device(validated_input.device_name)

        if "erase" in validated_input.config_commands.lower() or "write erase" in validated_input.config_commands.lower():
             logger.warning(f"Rejected potentially dangerous command on {validated_input.device_name}: {validated_input.config_commands}")
             return {"status": "error", "error": "Potentially dangerous command detected (erase). Operation aborted."}

        cleaned_config = textwrap.dedent(validated_input.config_commands.strip())
        if not cleaned_config:
             return {"status": "error", "error": "Empty configuration provided."}

        logger.info(f"Applying configuration on {validated_input.device_name}:\n{cleaned_config}")
        output = device.configure(cleaned_config)
        logger.info(f"Configuration result on {validated_input.device_name}: {output}")
        return {"status": "success", "message": f"Configuration applied on {validated_input.device_name}.", "output": output}

    except Exception as e:
        logger.error(f"Error applying configuration on {validated_input.device_name}: {e}", exc_info=True)
        return {"status": "error", "error": f"Configuration error: {e}"}
    finally:
        _disconnect_device(device)


def execute_learn_config(params: dict) -> dict:
    try:
        validated_input = DeviceOnlyInput(**params)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for execute_learn_config: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}

    device = None
    try:
        device = _get_device(validated_input.device_name)
        logger.info(f"Learning configuration from {validated_input.device_name}...")

        device.enable()
        raw_output = device.execute("show run brief")

        # Clean the raw_output
        cleaned_output = clean_output(raw_output)

        logger.info(f"Successfully learned config from {validated_input.device_name}")

        return {
            "status": "completed_raw",
            "device": validated_input.device_name,
            "output": {
                "raw_output": cleaned_output
            }
        }

    except Exception as e:
        logger.error(f"Error learning config from {validated_input.device_name}: {e}", exc_info=True)
        return {"status": "error", "error": f"Error learning config: {e}"}
    finally:
        _disconnect_device(device)

def clean_output(output: str) -> str:
    # Remove ANSI escape sequences
    ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
    output = ansi_escape.sub('', output)

    # Remove non-printable control characters
    output = ''.join(char for char in output if char in string.printable)

    return output


def execute_learn_logging(params: dict) -> dict:
    try:
        validated_input = DeviceOnlyInput(**params)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for execute_learn_logging: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}
    device = None
    try:
        device = _get_device(validated_input.device_name)
        logger.info(f"Learning logging output from {validated_input.device_name}...")

        raw_output = device.execute("show logging last 250")
        logger.info(f"Successfully learned logs from {validated_input.device_name}")

        return {
            "status": "completed_raw",
            "device": validated_input.device_name,
            "output": {
                "raw_output": raw_output
            }
        }

    except Exception as e:
        logger.error(f"Error learning logs from {validated_input.device_name}: {e}", exc_info=True)
        return {"status": "error", "error": f"Error learning logs: {e}"}
    finally:
        _disconnect_device(device)


def run_ping_command(params: dict) -> dict:
    try:
        validated_input = DeviceCommandInput(**params)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for run_ping_command: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}
    device = None
    try:
        if not validated_input.command.lower().strip().startswith("ping"):
             return {"status": "error", "error": f"Command '{validated_input.command}' is not a 'ping' command."}

        device = _get_device(validated_input.device_name)
        logger.info(f"Executing ping: '{validated_input.command}' on {validated_input.device_name}")

        try:
             # Attempt parsing first, as ping output is often structured
             parsed_output = device.parse(validated_input.command)
             logger.info(f"Parsed ping output for '{validated_input.command}' on {validated_input.device_name}")
             json.dumps(parsed_output) # Verify serializability
             return {"status": "completed", "device": validated_input.device_name, "output": parsed_output}
        except Exception as parse_exc:
             logger.warning(f"Parsing ping failed for '{validated_input.command}' on {validated_input.device_name}: {parse_exc}. Falling back to execute.")
             raw_output = device.execute(validated_input.command)
             logger.info(f"Executed ping (fallback): '{validated_input.command}' on {validated_input.device_name}")
             return {"status": "completed_raw", "device": ..., "output": wrap_output_for_clients(raw_output)}

    except Exception as e:
        logger.error(f"Error in run_ping_command for {validated_input.device_name}: {e}", exc_info=True)
        return {"status": "error", "error": f"Ping execution error: {e}"}
    finally:
        _disconnect_device(device)

SUPPORTED_LINUX_COMMANDS = [
    "ifconfig",
    "ifconfig {interface}",
    "ip route show table all",
    "ls -l",
    "ls -l {directory}",
    "netstat -rn",
    "ps -ef",
    "ps -ef | grep {grep}",
    "route",
    "route {flag}"
]

def run_linux_command(command: str, device_name: str):
    try:
        logger.info("Loading testbed...")
        testbed = loader.load(TESTBED_PATH)

        if device_name not in testbed.devices:
            return {"status": "error", "error": f"Device '{device_name}' not found in testbed."}

        device = testbed.devices[device_name]

        if not device.is_connected():
            logger.info(f"Connecting to {device_name} via SSH...")
            device.connect()

        if ">" in command or "|" in command:
            logger.info(f"Detected redirection or pipe in command: {command}")
            command = f'sh -c "{command}"'

        try:
            parser = get_parser(command, device)
            if parser:
                logger.info(f"Parsing output for command: {command}")
                output = device.parse(command)
            else:
                raise ValueError("No parser available")
        except Exception as e:
            logger.warning(f"No parser found for command: {command}. Using `execute` instead. Error: {e}")
            output = device.execute(command)

        logger.info(f"Disconnecting from {device_name}...")
        device.disconnect()

        return {"status": "completed", "device": device_name, "output": output}

    except Exception as e:
        logger.error(f"Error executing command on {device_name}: {str(e)}")
        return {"status": "error", "error": str(e)}


def run_linux_command_tool(params: dict) -> dict:
    try:
        validated = LinuxCommandInput(**params)
        return run_linux_command(validated.command, validated.device_name)
    except ValidationError as ve:
        logger.warning(f"Input validation failed for run_linux_command_tool: {ve}")
        return {"status": "error", "error": f"Invalid input: {ve}"}

# --- Tool Definitions ---

AVAILABLE_TOOLS = {
    "pyats_run_show_command": {
        "function": run_show_command,
        "description": "Executes a general Cisco IOS/NX-OS 'show' command (e.g., 'show ip interface brief', 'show version', 'show inventory') on a specified device to gather its current operational state or specific information. Returns parsed JSON output when available, otherwise returns raw text output. Use this for general device information gathering. Requires 'device_name' and the exact 'command' string.",
        "input_model": DeviceCommandInput
    },
    "pyats_configure_device": {
        "function": apply_device_configuration,
        "description": "Applies configuration commands to a specified Cisco IOS/NX-OS device. Enters configuration mode and executes the provided commands to modify the device's settings. Use this for making changes to the device configuration. Requires 'device_name' and the 'config_commands' (can be multi-line).",
        "input_model": ConfigInput
    },
    "pyats_show_running_config": {
        "function": execute_learn_config,
         "description": "Retrieves the full running configuration from a Cisco IOS/NX-OS device using 'show running-config'. Returns raw text output as there is no parser available. Requires 'device_name'.",
        "input_model": DeviceOnlyInput
    },
    "pyats_show_logging": {
        "function": execute_learn_logging,
        "description": "Retrieves recent system logs using 'show logging last 250' on a Cisco IOS/NX-OS device. Returns raw text output. Requires 'device_name'.",
        "input_model": DeviceOnlyInput
    },
    "pyats_ping_from_network_device": {
        "function": run_ping_command,
        "description": "Executes a 'ping' command on a specified Cisco IOS/NX-OS device to test network reachability to a target IP address or hostname (e.g., 'ping 8.8.8.8', 'ping vrf MGMT 10.0.0.1'). Returns parsed JSON output for standard pings when possible, otherwise raw text. Requires 'device_name' and the exact 'command' string.",
        "input_model": DeviceCommandInput
    },
    "pyats_run_linux_command": {
        "function": run_linux_command_tool,
        "description": "Executes common Linux commands on a specified device (e.g., 'ifconfig', 'ps -ef', 'netstat -rn', including piping and redirection). Parsed output is returned when available, otherwise raw output.",
        "input_model": LinuxCommandInput
    },
}

logger.info(f"🚀 Registered {len(AVAILABLE_TOOLS)} tools at startup: {list(AVAILABLE_TOOLS.keys())}")

# --- JSON-RPC Handling ---

def discover_tools() -> List[Dict[str, Any]]:
    logger.info(f"🛠 discover_tools(): AVAILABLE_TOOLS keys: {list(AVAILABLE_TOOLS.keys())}")
    tools_list = []

    for name, tool_info in AVAILABLE_TOOLS.items():
        raw_schema = tool_info["input_model"].model_json_schema()

        cleaned_properties = {
            k: {k2: v2 for k2, v2 in v.items() if k2 != "title"}
            for k, v in raw_schema.get("properties", {}).items()
        }

        input_schema = {
            "type": "object",
            "properties": cleaned_properties,
            "required": raw_schema.get("required", []),
            "additionalProperties": False
        }

        tools_list.append({
            "name": name,
            "description": tool_info["description"],
            "parameters": input_schema
        })

    logger.info(f"✅ discover_tools() returning {len(tools_list)} tools")
    logger.info("🔍 Full tool dump for verification:")
    for tool in tools_list:
        logger.info(json.dumps(tool, indent=2))
    return tools_list

# Synchronous tool calling (unchanged, used in thread executor)
def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if tool_name not in AVAILABLE_TOOLS:
        logger.warning(f"Requested tool '{tool_name}' not found.")
        return {"error": {"code": -32601, "message": f"Method not found: {tool_name}"}}

    tool_info = AVAILABLE_TOOLS[tool_name]
    func = tool_info["function"]

    # Unwrap 'params' if present
    if isinstance(arguments, dict) and "params" in arguments and isinstance(arguments["params"], dict):
        arguments = arguments["params"]

    try:
        result_data = func(arguments)
        json.dumps(result_data)
        return result_data
    except Exception as e:
        logger.error(f"Unexpected error calling tool '{tool_name}': {e}", exc_info=True)
        return {"error": {"code": -32603, "message": f"Internal server error during tool call: {e}"}}

# Async-safe wrapper
def call_tool_async(tool_name: str, arguments: Dict[str, Any]) -> asyncio.Future:
    loop = asyncio.get_event_loop()
    return loop.run_in_executor(None, partial(call_tool, tool_name, arguments))

# --- Async version of request handler ---
async def process_request(request_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_id = request_data.get("id")

    if not isinstance(request_data, dict) or \
       request_data.get("jsonrpc") != "2.0" or \
       "method" not in request_data:
        logger.warning(f"Invalid JSON-RPC request received: {request_data}")
        return {"jsonrpc": "2.0", "error": {"code": -32600, "message": "Invalid Request"}, "id": request_id}

    method = request_data["method"]
    params = request_data.get("params", {})
    logger.info(f"📥 Incoming method: {method}, params: {params}")

    response_content = None
    error_content = None

    logger.debug(f"Processing method: {method}")

    request_id = request_data.get("id")
    method = request_data.get("method")

    # VS Code sends this notification right after starting
    if method == "notifications/initialized":
        logger.info("Received VS Code initialization notification.")
        return None  # Don't send a response for notifications

    if method == "initialize":
        logger.info("✅ Handshake: returning MCP capabilities")
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": {
                "capabilities": {
                    "tools": {
                        "discover": True,
                        "call": True
                    }
                },
                "serverInfo": {
                    "name": "pyATS MCP Server",
                    "version": "1.0.0"
                }
            }
        }

    if method in ("tools/discover", "tools/list"):
        logger.info(f"⚙️ {method} → Discovering tools...")
        tools = discover_tools()
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "result": tools
        }

    elif method == "tools/call":
        tool_name = params.get("name")
        arguments = params.get("arguments", {})
        if not tool_name or not isinstance(arguments, dict):
            logger.warning(f"Invalid params for tools/call: {params}")
            error_content = {"code": -32602, "message": "Invalid params for tools/call (missing 'name' or 'arguments')"}
        else:
            logger.info(f"Calling tool '{tool_name}' with args: {arguments}")
            result_or_error = await call_tool_async(tool_name, arguments)
            if "error" in result_or_error:
                error_content = result_or_error["error"]
            else:
                response_content = result_or_error
    else:
        logger.warning(f"Method not found: {method}")
        error_content = {"code": -32601, "message": f"Method not found: {method}"}

    if error_content:
        return {"jsonrpc": "2.0", "error": error_content, "id": request_id}
    elif response_content is not None:
        return {"jsonrpc": "2.0", "result": response_content, "id": request_id}
    else:
        logger.error(f"Request processed but resulted in no response or error content for method {method}")
        return {"jsonrpc": "2.0", "error": {"code": -32603, "message": "Internal error: Failed to generate response"}, "id": request_id}
    
# --- Stdio Server Functions ---

def send_response(response_data: Dict[str, Any]):
    try:
        response_string = json.dumps(response_data) + "\n"
        logger.info(f"Sending response: {response_string.strip()}")  # Log the response being sent
        sys.stdout.write(response_string)
        sys.stdout.flush()
        logger.debug(f"Sent response: {response_string.strip()}")
    except (TypeError, OverflowError) as e:
        logger.error(f"Failed to serialize response data: {e}", exc_info=True)
        error_response = {
            "jsonrpc": "2.0",
            "error": {"code": -32603, "message": f"Internal error: Could not serialize response - {e}"},
            "id": response_data.get("id")
        }
        sys.stdout.write(json.dumps(error_response) + "\n")
        sys.stdout.flush()
    except Exception as e:
        logger.error(f"Failed to write response to stdout: {e}", exc_info=True)

def monitor_stdin():
    logger.info("Stdin monitoring thread started.")
    while True:
        try:
            logger.debug("Waiting for input...")
            line = sys.stdin.readline()
            if not line:
                logger.warning("Stdin closed or empty line received. Keeping monitor thread alive.")
                time.sleep(1)
                continue

            line = line.strip()
            if not line:
                time.sleep(0.05)
                continue

            logger.debug(f"Received line: {line}")
            try:
                request_data = json.loads(line)
                response = asyncio.run(process_request(request_data))

                if response:
                    send_response(response)
                else:
                    logger.info("No response (likely a notification). Continuing...")

            except json.JSONDecodeError as e:
                logger.error(f"JSON decode error: {e} for line: '{line}'")
                send_response({
                    "jsonrpc": "2.0",
                    "error": {"code": -32700, "message": f"Parse error: {e}"},
                    "id": None
                })
            except Exception as e:
                logger.error(f"Error processing request: {e}", exc_info=True)
                send_response({
                    "jsonrpc": "2.0",
                    "error": {"code": -32603, "message": f"Internal server error: {e}"},
                    "id": None
                })

        except Exception as e:
            logger.error(f"Exception in monitor_stdin loop: {e}", exc_info=True)
            time.sleep(0.1)


async def run_server_oneshot():
    """Reads one JSON request from stdin and writes one JSON response to stdout."""
    logger.info("Starting pyATS MCP Server in one-shot mode...")
    response_sent = False
    try:
        input_data = sys.stdin.read()
        logger.info(f"Received raw input: {input_data[:500]}{'...' if len(input_data) > 500 else ''}")

        last_json_line = None
        for line in reversed(input_data.strip().splitlines()):
            if line.strip().startswith('{') and line.strip().endswith('}'):
                last_json_line = line.strip()
                break

        if not last_json_line:
            logger.error("No valid JSON object found in input.")
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": "Parse error: No valid JSON found"},
                "id": None
            })
            response_sent = True
            return

        logger.info(f"Processing JSON: {last_json_line}")
        request_json = json.loads(last_json_line)
        response = await process_request(request_json)
        if response:
            send_response(response)
            response_sent = True

    except json.JSONDecodeError as e:
        logger.error(f"JSON Decode Error (oneshot): {e}")
        if not response_sent:
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32700, "message": f"Parse error: {e}"},
                "id": None
            })
    except Exception as e:
        logger.error(f"Unhandled Server Error (oneshot): {e}", exc_info=True)
        if not response_sent:
            send_response({
                "jsonrpc": "2.0",
                "error": {"code": -32000, "message": f"Server error: {e}"},
                "id": None
            })
    finally:
        logger.info("pyATS MCP Server one-shot finished.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="pyATS MCP Server - Runs continuously via stdio by default.")
    parser.add_argument("--oneshot", action="store_true", help="Run in one-shot mode (read stdin, write stdout, exit)")
    args = parser.parse_args()

    if args.oneshot:
        asyncio.run(run_server_oneshot())  # 🔧 <-- this line changed
    else:
        logger.info("Starting pyATS MCP Server in continuous stdio mode...")
        stdin_thread = threading.Thread(target=monitor_stdin, name="StdinMonitorThread", daemon=True)
        stdin_thread.start()

        try:
            while stdin_thread.is_alive():
                time.sleep(0.5)
        except KeyboardInterrupt:
            logger.info("KeyboardInterrupt received. Shutting down...")
        except Exception as e:
            logger.error(f"Unexpected error in main thread: {e}", exc_info=True)
        finally:
            logger.info("Main thread exiting.")