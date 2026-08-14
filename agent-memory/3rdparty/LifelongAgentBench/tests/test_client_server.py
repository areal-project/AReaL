from abc import ABC, abstractmethod
from multiprocessing import Process

from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, APIRouter
import time
import socket

from src.utils import Server, Client


class Request:
    class SetStrIdentity(BaseModel):
        str_identity: str

    class FunctionWithArgs(BaseModel):
        arg1: int
        arg2: str


class Response:
    class FunctionWithoutArgs(BaseModel):
        output_str: str

    class FunctionWithArgs(BaseModel):
        str_identity: str
        int_identity: int
        bool_identity: bool

    class GetStrIdentity(BaseModel):
        str_identity: str


class PrincipleInterface(ABC):
    @abstractmethod
    def set_str_identity(self, str_identity: str):
        pass

    @abstractmethod
    def get_str_identity(self) -> str:
        pass

    @abstractmethod
    def func_without_args(self) -> str:
        pass

    @abstractmethod
    def func_with_args(self, arg1: int, arg2: str) -> tuple[str, int, bool]:
        pass


class Principle(PrincipleInterface):
    def __init__(
        self,
        str_identity: str,
        left_parent: "Principle" = None,
        right_parent: "Principle" = None,
    ):
        assert len(str_identity) == 1
        self.str_identity = str_identity
        self.int_identity = ord(str_identity) - ord("a")
        self.bool_identity = self.int_identity % 2 == 0
        self.left_parent = left_parent
        self.right_parent = right_parent
        self.none_value = None

    def set_str_identity(self, str_identity: str):
        self.str_identity = str_identity

    def get_str_identity(self) -> str:
        return self.str_identity

    def func_without_args(self) -> str:
        result = ""
        result += f"func_without_args\n"
        result += f"self.str_identity: {self.str_identity}\n"
        result += f"self.int_identity: {self.int_identity}\n"
        result += f"self.bool_identity: {self.bool_identity}\n"
        result += f"has_left_parent: {self.left_parent is not None}\n"
        result += f"has_right_parent: {self.right_parent is not None}\n"
        return result

    def func_with_args(self, arg1: int, arg2: str):
        return self.str_identity, self.int_identity, self.bool_identity


class PrincipleServer(Server):
    def __init__(self, router: APIRouter, principle: Principle):
        Server.__init__(self, router, principle)
        self.principal = principle
        self.router.post("/get_str_identity")(self.get_str_identity)
        self.router.post("/set_str_identity")(self.set_str_identity)
        self.router.post("/func_without_args")(self.func_without_args)
        self.router.post("/func_with_args")(self.func_with_args)

    def set_str_identity(self, request: Request.SetStrIdentity):
        self.principal.set_str_identity(request.str_identity)

    def get_str_identity(self):
        return Response.GetStrIdentity(str_identity=self.principal.get_str_identity())

    def func_without_args(self):
        return Response.FunctionWithoutArgs(
            output_str=self.principal.func_without_args()
        )

    def func_with_args(self, request: Request.FunctionWithArgs):
        str_identity, int_identity, bool_identity = self.principal.func_with_args(
            request.arg1, request.arg2
        )
        return Response.FunctionWithArgs(
            str_identity=str_identity,
            int_identity=int_identity,
            bool_identity=bool_identity,
        )

    @staticmethod
    def start_server(principle: Principle, port: int):
        app = FastAPI()
        router = APIRouter()
        _ = PrincipleServer(router, principle)
        app.include_router(router)
        uvicorn.run(app, host="0.0.0.0", port=port)


class PrincipleClient(Client, PrincipleInterface):
    def __init__(self, server_address: str, request_timeout: int):
        Client.__init__(
            self, server_address=server_address, request_timeout=request_timeout
        )

    def set_str_identity(self, str_identity: str):
        _ = self._call_server(
            "/set_str_identity", Request.SetStrIdentity(str_identity=str_identity), None
        )

    def get_str_identity(self) -> str:
        response: Response.GetStrIdentity = self._call_server(
            "/get_str_identity", None, Response.GetStrIdentity
        )
        return response.str_identity

    def func_without_args(self) -> str:
        response: Response.FunctionWithoutArgs = self._call_server(
            "/func_without_args", None, Response.FunctionWithoutArgs
        )
        return response.output_str

    def func_with_args(self, arg1: int, arg2: str):
        response: Response.FunctionWithArgs = self._call_server(
            "/func_with_args",
            Request.FunctionWithArgs(arg1=arg1, arg2=arg2),
            Response.FunctionWithArgs,
        )
        return response.str_identity, response.int_identity, response.bool_identity


"""
Hierarchy:
f   g
 \ / \
  d   e
   \ /
    b   c
     \ /
      a
"""


process_list = []
client_list = []
port_list = []


def construct_hierarchy(
    principle: Principle, server_port: int
) -> tuple[Process, Principle]:
    process = Process(
        target=PrincipleServer.start_server, args=(principle, server_port)
    )
    process.start()
    time.sleep(0.1)
    client = PrincipleClient(f"http://localhost:{server_port}", 1)
    process_list.append(process)
    client_list.append(client)
    return process, client


while (
    len([port for port in port_list if port is not None]) < 7 and len(port_list) < 100
):  # The constraints of less than 100 is for avoiding infinite loop
    candidate_port = 7000 + len(port_list)
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("localhost", candidate_port))
        sock.close()
        port_list.append(None)
    except ConnectionRefusedError:
        port_list.append(candidate_port)
port_list = [port for port in port_list if port is not None][::-1]
process_g, client_g = construct_hierarchy(Principle("g"), port_list[6])
process_f, client_f = construct_hierarchy(Principle("f"), port_list[5])
process_e, client_e = construct_hierarchy(
    Principle("e", left_parent=client_g), port_list[4]
)
process_d, client_d = construct_hierarchy(
    Principle("d", left_parent=client_f, right_parent=client_g), port_list[3]
)
process_c, client_c = construct_hierarchy(Principle("c"), port_list[2])
process_b, client_b = construct_hierarchy(
    Principle("b", left_parent=client_d, right_parent=client_e), port_list[1]
)
process_a, client_a = construct_hierarchy(
    Principle("a", left_parent=client_b, right_parent=client_c), port_list[0]
)
client_list.reverse()
process_list.reverse()


class TestClass:
    def test_func_without_args(self):
        for index, client in enumerate(client_list):
            expected_client_str_identity = chr(ord("a") + index)
            outputs_str = client.func_without_args()
            assert f"self.str_identity: {expected_client_str_identity}\n" in outputs_str

    def test_func_with_args(self):
        for index, client in enumerate(client_list):
            expected_client_str_identity = chr(ord("a") + index)
            expected_client_int_identity = index
            expected_client_bool_identity = index % 2 == 0
            str_identity, int_identity, bool_identity = client.func_with_args(
                index, expected_client_str_identity
            )
            assert str_identity == expected_client_str_identity
            assert int_identity == expected_client_int_identity
            assert bool_identity == expected_client_bool_identity

    def test_immutable_type_assessment(self):
        # region int
        f_int_identity = client_a.left_parent.left_parent.left_parent.int_identity
        assert f_int_identity == 5
        f_int_identity = client_a.left_parent.left_parent.left_parent.int_identity = 50
        assert f_int_identity == 50
        client_a.left_parent.left_parent.left_parent.int_identity += (
            1  # This also works!
        )
        # endregion
        # region None, True, False
        assert client_a.left_parent.left_parent.left_parent.int_identity == 51
        assert client_a.none_value is None
        assert client_a.bool_identity is True
        client_a.bool_identity = False
        assert client_a.bool_identity is False
        # endregion
        # region str
        a_str_identity = client_a.str_identity
        assert a_str_identity == "a"
        client_a.str_identity = "new_a"
        assert client_a.str_identity == "new_a"
        assert client_a.left_parent.right_parent.str_identity == "e"
        # endregion

    def test_global_uniqueness(self):
        client_a.left_parent.left_parent.right_parent.str_identity = "new_g"
        assert client_a.left_parent.left_parent.right_parent.str_identity == "new_g"
        assert client_a.left_parent.right_parent.left_parent.str_identity == "new_g"
        # Check Whether the two clients connected to the same server
        client_g1 = client_a.left_parent.left_parent.right_parent
        client_g2 = client_a.left_parent.right_parent.left_parent
        assert client_g1.str_identity == client_g2.str_identity

    def test_finish(self):
        for process in process_list:
            process.terminate()
            process.join()
        print("All processes terminated.")
