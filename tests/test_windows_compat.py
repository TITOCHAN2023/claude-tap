"""Regression tests for Windows-specific bugs reported in issue #83.

Patches `signal` and `shutil.which` to simulate the Windows environment so the
tests run identically on Linux/macOS CI and on real Windows.
"""

from __future__ import annotations

import asyncio
import signal
import subprocess
import sys
from pathlib import Path

import pytest

from claude_tap.cli import _start_background_update, run_client
from claude_tap.history import _rel_posix


class _DummyProc:
    def __init__(self) -> None:
        self.pid = 12345
        self.returncode: int | None = None

    async def wait(self) -> int:
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = 0

    def kill(self) -> None:
        self.returncode = -9


def _strip_sigtstp(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove SIGTSTP and force loop.add/remove_signal_handler to NotImplementedError."""
    if hasattr(signal, "SIGTSTP"):
        monkeypatch.delattr(signal, "SIGTSTP", raising=False)

    def _not_impl(*_args, **_kwargs):
        raise NotImplementedError

    monkeypatch.setattr("asyncio.AbstractEventLoop.add_signal_handler", _not_impl, raising=False)
    monkeypatch.setattr("asyncio.AbstractEventLoop.remove_signal_handler", _not_impl, raising=False)


@pytest.mark.asyncio
async def test_run_client_does_not_touch_sigtstp_when_absent(monkeypatch) -> None:
    async def fake_create_subprocess_exec(*cmd, **kwargs):
        return _DummyProc()

    _strip_sigtstp(monkeypatch)
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: r"C:\Users\x\.local\bin\claude.cmd")
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(43123, ["--version"], client="claude", proxy_mode="reverse")
    assert code == 0


@pytest.mark.asyncio
async def test_run_client_passes_resolved_path_for_cmd_shim(monkeypatch) -> None:
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        return _DummyProc()

    shim_path = r"C:\Users\x\.local\bin\claude.cmd"
    _strip_sigtstp(monkeypatch)
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: shim_path)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(43123, ["--version"], client="claude", proxy_mode="reverse")
    assert code == 0
    cmd = captured["cmd"]
    assert cmd[0] == shim_path, "resolved .cmd shim path must be preserved"
    assert cmd[1] == "--settings", "--settings must be injected before forwarded args"
    import json

    injected = json.loads(cmd[2])
    assert injected == {"env": {"ANTHROPIC_BASE_URL": "http://127.0.0.1:43123"}}
    assert cmd[3:] == ("--version",), "original args must follow --settings payload"


@pytest.mark.asyncio
async def test_run_client_uses_wrapper_provided_claude_binary(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    async def fake_create_subprocess_exec(*cmd, **kwargs):
        captured["cmd"] = cmd
        return _DummyProc()

    wrapped_claude = tmp_path / "claude"
    wrapped_claude.write_text("#!/bin/sh\n", encoding="utf-8")
    _strip_sigtstp(monkeypatch)
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: None)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fake_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(
        43123,
        ["--output-format", "stream-json"],
        client="claude",
        proxy_mode="reverse",
        client_cmd=str(wrapped_claude),
    )
    assert code == 0
    cmd = captured["cmd"]
    assert cmd[0] == str(wrapped_claude)
    assert cmd[1] == "--settings"
    assert cmd[3:] == ("--output-format", "stream-json")


@pytest.mark.asyncio
async def test_run_client_does_not_execute_wrapper_directory(monkeypatch, tmp_path: Path) -> None:
    async def fail_create_subprocess_exec(*cmd, **kwargs):
        raise AssertionError(f"directory path should not be executed: {cmd}")

    wrapped_dir = tmp_path / "claude"
    wrapped_dir.mkdir()
    _strip_sigtstp(monkeypatch)
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: None)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", fail_create_subprocess_exec)
    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    code = await run_client(
        43123,
        ["--output-format", "stream-json"],
        client="claude",
        proxy_mode="reverse",
        client_cmd=str(wrapped_dir),
    )
    assert code == 1


def test_module_import_reconfigures_stdout_to_utf8() -> None:
    import claude_tap.cli  # noqa: F401

    for stream in (sys.stdout, sys.stderr):
        encoding = getattr(stream, "encoding", "")
        assert encoding and encoding.lower().replace("-", "") == "utf8", f"expected UTF-8, got {encoding!r} on {stream}"


def test_rel_posix_uses_forward_slashes(tmp_path: Path) -> None:
    nested = tmp_path / "2026-04-29" / "trace_001234.jsonl"
    nested.parent.mkdir(parents=True)
    nested.write_text("{}\n", encoding="utf-8")
    assert _rel_posix(nested, tmp_path) == "2026-04-29/trace_001234.jsonl"


def test_start_background_update_resolves_uv_shim(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        return object()

    shim = r"C:\Users\x\AppData\Local\Programs\uv\uv.cmd"
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda name: shim if name == "uv" else None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)

    _start_background_update("uv")
    assert captured["cmd"] == [shim, "tool", "upgrade", "claude-tap"]


def test_start_background_update_hides_windows_console(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class FakeStartupInfo:
        def __init__(self) -> None:
            self.dwFlags = 0
            self.wShowWindow: int | None = None

    def fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["kwargs"] = kwargs
        return object()

    shim = r"C:\Users\x\AppData\Local\Programs\uv\uv.cmd"
    monkeypatch.setattr("claude_tap.cli_update.sys.platform", "win32")
    monkeypatch.setattr("claude_tap.cli_update.shutil.which", lambda name: shim if name == "uv" else None)
    monkeypatch.setattr(subprocess, "Popen", fake_popen)
    monkeypatch.setattr(subprocess, "CREATE_NO_WINDOW", 0x1000, raising=False)
    monkeypatch.setattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x2000, raising=False)
    monkeypatch.setattr(subprocess, "STARTF_USESHOWWINDOW", 0x4000, raising=False)
    monkeypatch.setattr(subprocess, "SW_HIDE", 0, raising=False)
    monkeypatch.setattr(subprocess, "STARTUPINFO", FakeStartupInfo, raising=False)

    _start_background_update("uv")

    assert captured["cmd"] == [shim, "tool", "upgrade", "claude-tap"]
    kwargs = captured["kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["stdin"] == subprocess.DEVNULL
    assert kwargs["stdout"] == subprocess.DEVNULL
    assert kwargs["stderr"] == subprocess.DEVNULL
    assert kwargs["creationflags"] == 0x1000 | 0x2000
    startupinfo = kwargs["startupinfo"]
    assert isinstance(startupinfo, FakeStartupInfo)
    assert startupinfo.dwFlags == 0x4000
    assert startupinfo.wShowWindow == 0


def test_start_background_update_returns_none_when_uv_missing(monkeypatch) -> None:
    monkeypatch.setattr("claude_tap.cli.shutil.which", lambda _: None)
    assert _start_background_update("uv") is None


def test_install_interrupt_handlers_uses_loop_when_available() -> None:
    from claude_tap.cli_clients import _install_interrupt_handlers

    added: list[object] = []
    removed: list[object] = []

    class FakeLoop:
        def add_signal_handler(self, sig, _handler):
            added.append(sig)

        def remove_signal_handler(self, sig):
            removed.append(sig)

    restore = _install_interrupt_handlers(FakeLoop(), lambda: None, lambda: None, 99)
    assert added == [signal.SIGINT, 99]
    restore()
    assert removed == [signal.SIGINT, 99]


def test_install_interrupt_handlers_falls_back_to_signal_signal(monkeypatch) -> None:
    """When the loop lacks add_signal_handler (Windows Proactor), SIGINT is
    installed via signal.signal and restored afterward."""
    from claude_tap import cli_clients

    class FakeLoop:
        def add_signal_handler(self, *_a, **_k):
            raise NotImplementedError

        def remove_signal_handler(self, *_a, **_k):  # pragma: no cover - must not run
            raise AssertionError("loop removal must not run on the fallback path")

    installed: dict[str, object] = {}

    def fake_signal(sig, handler):
        installed["sig"] = sig
        installed["handler"] = handler
        return "PREVIOUS"

    monkeypatch.setattr(cli_clients.signal, "signal", fake_signal)

    sigint_calls: list[str] = []
    restore = cli_clients._install_interrupt_handlers(
        FakeLoop(), lambda: sigint_calls.append("int"), lambda: None, None
    )
    assert installed["sig"] == signal.SIGINT
    # The installed OS handler must delegate to on_sigint.
    installed["handler"](signal.SIGINT, None)
    assert sigint_calls == ["int"]
    # Restoring puts the previously-saved handler back.
    restore()
    assert installed["handler"] == "PREVIOUS"


def test_default_data_dir_posix_is_unchanged(monkeypatch) -> None:
    from claude_tap import trace_store

    monkeypatch.setattr(trace_store.sys, "platform", "linux")
    assert trace_store._default_data_dir() == Path.home() / ".local" / "share" / "claude-tap"


def test_default_data_dir_windows_prefers_localappdata(monkeypatch, tmp_path: Path) -> None:
    from claude_tap import trace_store

    fake_home = tmp_path / "home"
    local = tmp_path / "AppData" / "Local"
    monkeypatch.setattr(trace_store.sys, "platform", "win32")
    monkeypatch.setattr(trace_store.Path, "home", lambda: fake_home)
    monkeypatch.setenv("LOCALAPPDATA", str(local))

    assert trace_store._default_data_dir() == local / "claude-tap"


def test_default_data_dir_windows_keeps_legacy_when_db_exists(monkeypatch, tmp_path: Path) -> None:
    from claude_tap import trace_store

    fake_home = tmp_path / "home"
    legacy = fake_home / ".local" / "share" / "claude-tap"
    legacy.mkdir(parents=True)
    (legacy / trace_store.DB_FILENAME).write_bytes(b"")
    monkeypatch.setattr(trace_store.sys, "platform", "win32")
    monkeypatch.setattr(trace_store.Path, "home", lambda: fake_home)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "AppData" / "Local"))

    assert trace_store._default_data_dir() == legacy


def test_restrict_key_file_uses_icacls_on_windows(monkeypatch, tmp_path: Path) -> None:
    from claude_tap import certs

    key = tmp_path / "ca-key.pem"
    key.write_bytes(b"x")
    captured: dict[str, object] = {}

    def fake_run(cmd, **_kwargs):
        captured["cmd"] = cmd
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(certs.sys, "platform", "win32")
    monkeypatch.setenv("USERNAME", "tester")
    monkeypatch.setattr(certs.subprocess, "run", fake_run)

    certs._restrict_key_file(key)
    assert captured["cmd"] == ["icacls", str(key), "/inheritance:r", "/grant:r", "tester:F"]


def test_restrict_key_file_uses_chmod_on_posix(monkeypatch, tmp_path: Path) -> None:
    from claude_tap import certs

    key = tmp_path / "ca-key.pem"
    key.write_bytes(b"x")
    modes: list[int] = []
    ran: list[bool] = []

    monkeypatch.setattr(certs.sys, "platform", "linux")
    monkeypatch.setattr(certs.subprocess, "run", lambda *a, **k: ran.append(True))
    monkeypatch.setattr(Path, "chmod", lambda self, mode: modes.append(mode))

    certs._restrict_key_file(key)
    assert modes == [0o600]
    assert ran == []
