import json
import socket
import threading
from typing import Any

from loguru import logger
from tau2.data_model.tasks import Task
from tau2.environment.environment import Environment


def _recv_json(sock: socket.socket) -> dict[str, Any] | None:
    chunks: list[bytes] = []
    while True:
        data = sock.recv(4096)
        if not data:
            break
        chunks.append(data)
        if b"\n" in data:
            break
    return (
        json.loads(b"".join(chunks).split(b"\n", 1)[0].decode("utf-8"))
        if chunks
        else None
    )


def _serialize_result(result: Any) -> Any:
    if result is None or isinstance(result, (str, int, float, bool)):
        return result
    if hasattr(result, "model_dump"):
        return result.model_dump()
    if hasattr(result, "dict"):
        return result.dict()
    if isinstance(result, list):
        return [_serialize_result(item) for item in result]
    return (
        {key: _serialize_result(value) for key, value in result.items()}
        if isinstance(result, dict)
        else result
    )


class EnvironmentSocketServer:
    def __init__(
        self,
        environment: Environment,
        task: Task = None,
        host: str = "127.0.0.1",
        port: int = 0,
    ):
        self.environment = environment
        self.task = task
        self.host = host
        self.port = port
        self.server_socket: socket.socket | None = None
        self.running = False
        self.thread: threading.Thread | None = None

    @staticmethod
    def _send_json(sock: socket.socket, payload: dict[str, Any]) -> None:
        sock.sendall((json.dumps(payload) + "\n").encode("utf-8"))

    def start(self) -> int:
        self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server_socket.bind((self.host, self.port))
        self.server_socket.listen(5)
        self.port = self.server_socket.getsockname()[1]
        self.running = True
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()
        logger.info("Environment Server listening on {}:{}", self.host, self.port)
        return self.port

    def _serve(self) -> None:
        while self.running and self.server_socket:
            try:
                client_socket, client_addr = self.server_socket.accept()
                logger.debug("Client connected: {}", client_addr)
                threading.Thread(
                    target=self._handle_client, args=(client_socket,), daemon=True
                ).start()
            except Exception as exc:
                if self.running:
                    logger.error("Server error: {}", exc)

    def _handle_client(self, client_socket: socket.socket) -> None:
        try:
            while request := _recv_json(client_socket):
                action = request.get("action")
                if action == "call_tool":
                    response = self._call_tool(
                        request["tool_name"],
                        request.get("requestor", "assistant"),
                        request.get("arguments", {}),
                    )
                elif action == "get_state":
                    response = self._get_state()
                else:
                    response = {"success": False, "error": f"Unknown action: {action}"}
                self._send_json(client_socket, response)
        except Exception as exc:
            logger.error("Client handler error: {}", exc)
        finally:
            client_socket.close()

    def _call_tool(
        self, tool_name: str, requestor: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        try:
            result = self.environment.make_tool_call(
                tool_name=tool_name, requestor=requestor, **arguments
            )
            self.environment.sync_tools()
            return {"success": True, "result": _serialize_result(result)}
        except Exception as exc:
            logger.error("Tool call error: {}", exc)
            return {"success": False, "error": str(exc)}

    def _get_state(self) -> dict[str, Any]:
        try:
            state = {
                "success": True,
                "domain": self.environment.domain_name,
                "db_hash": self.environment.get_db_hash(),
            }
            if self.task:
                state["task_id"] = self.task.id
            return state
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    def stop(self) -> None:
        self.running = False
        if self.server_socket:
            self.server_socket.close()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)
        logger.info("Environment Server stopped")

    def get_client_config(self) -> dict[str, Any]:
        return {
            "host": self.host,
            "port": self.port,
            **({"task_id": self.task.id} if self.task else {}),
        }


def create_openclaw_tool_script(tool_name: str, server_config: dict) -> str:
    host = json.dumps(server_config["host"])
    tool = json.dumps(tool_name)
    return f"""#!/usr/bin/env python
import json, socket, sys
def _recv_json(sock):
    chunks=[]
    while True:
        data=sock.recv(4096)
        if not data: break
        chunks.append(data)
        if b"\\n" in data: break
    return json.loads(b"".join(chunks).split(b"\\n",1)[0].decode("utf-8")) if chunks else None
try:
    args=json.loads(sys.argv[1]) if len(sys.argv)>1 else {{}}
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as sock:
        sock.connect(({host}, {server_config["port"]}))
        sock.sendall((json.dumps({{"action":"call_tool","tool_name":{tool},"requestor":"assistant","arguments":args}})+"\\n").encode("utf-8"))
        response=_recv_json(sock) or {{"success": False, "error": "Empty response"}}
    if not response.get("success"): raise Exception(response.get("error", "Tool call failed"))
    print(json.dumps({{"success": True, "result": response["result"]}}))
except Exception as exc:
    print(json.dumps({{"success": False, "error": str(exc)}}))
"""
