import pytest
from pydantic import BaseModel, Field

from app.core.tools.auth import AuthPolicy
from app.core.tools.base import BaseTool, ToolContext
from app.core.tools.executor import ToolExecutor
from app.core.tools.idempotency import SQLiteIdempotencyStore
from app.core.tools.registry import ToolRegistry


class EchoArgs(BaseModel):
    text: str = Field(min_length=1)


class EchoTool(BaseTool[EchoArgs, str]):
    name = "echo"
    ArgsModel = EchoArgs
    auth_required = True

    def __init__(self) -> None:
        self.calls = 0

    async def run(self, *, args: EchoArgs, ctx: ToolContext) -> str:
        self.calls += 1
        return f"echo:{args.text}"


@pytest.mark.asyncio
async def test_run_result_not_found(tmp_path):
    reg = ToolRegistry()
    idem = SQLiteIdempotencyStore(path=str(tmp_path / "idem.db"))
    ex = ToolExecutor(registry=reg, idempotency=idem, auth=AuthPolicy(mode="off", allow_users=set()))

    res = await ex.run_result("no_such_tool", args={"a": 1}, ctx=ToolContext(mode="rag"))
    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "NOT_FOUND"


@pytest.mark.asyncio
async def test_run_result_validation_error(tmp_path):
    reg = ToolRegistry()
    tool = EchoTool()
    reg.register(tool)
    idem = SQLiteIdempotencyStore(path=str(tmp_path / "idem.db"))
    ex = ToolExecutor(registry=reg, idempotency=idem, auth=AuthPolicy(mode="off", allow_users=set()))

    res = await ex.run_result("echo", args={"text": ""}, ctx=ToolContext(mode="rag"))
    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "VALIDATION_ERROR"
    assert isinstance((res.error.detail or {}).get("errors"), list)


@pytest.mark.asyncio
async def test_run_result_permission_denied_allowlist(tmp_path):
    reg = ToolRegistry()
    tool = EchoTool()
    reg.register(tool)
    idem = SQLiteIdempotencyStore(path=str(tmp_path / "idem.db"))
    auth = AuthPolicy(mode="allowlist", allow_users={"alice"})
    ex = ToolExecutor(registry=reg, idempotency=idem, auth=auth)

    res = await ex.run_result("echo", args={"text": "hi"}, ctx=ToolContext(user_id="anonymous", mode="rag"))
    assert res.ok is False
    assert res.error is not None
    assert res.error.code == "PERMISSION_DENIED"


@pytest.mark.asyncio
async def test_idempotency_cache_hit_sqlite(tmp_path):
    reg = ToolRegistry()
    tool = EchoTool()
    reg.register(tool)
    idem = SQLiteIdempotencyStore(path=str(tmp_path / "idem.db"))
    auth = AuthPolicy(mode="allowlist", allow_users={"anonymous"})
    ex = ToolExecutor(registry=reg, idempotency=idem, auth=auth)

    ctx = ToolContext(user_id="anonymous", session_id="s1", run_id="r1", mode="rag")
    key = "k1"
    res1 = await ex.run_result("echo", args={"text": "hello"}, ctx=ctx, idempotency_key=key, idempotency_ttl_s=60)
    res2 = await ex.run_result("echo", args={"text": "hello"}, ctx=ctx, idempotency_key=key, idempotency_ttl_s=60)

    assert res1.ok is True
    assert res1.cached is False
    assert res2.ok is True
    assert res2.cached is True
    assert tool.calls == 1  # second call should be served from cache

