"""Exercise Compose startup supervision without a Docker daemon."""

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import yaml

COMPOSE = Path(__file__).resolve().parents[2] / "docker-compose.yml"
STUB = """
import json
import os
from pathlib import Path
import signal
import sys
import time

root = Path(os.environ["STUB_ROOT"])
scenario = os.environ["STUB_SCENARIO"]
command = sys.argv[1]
if command == "serve":
    (root / "server.pid").write_text(str(os.getpid()))
    if scenario == "early_exit":
        sys.exit(7)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    (root / "list-ready").touch()
    while True:
        time.sleep(0.05)
elif command == "list":
    sys.exit(0 if (root / "list-ready").exists() else 1)
elif command == "pull":
    (root / "pull.pid").write_text(str(os.getpid()))
    (root / "model.json").write_text(json.dumps(sys.argv[2:]))
    if scenario == "pull_failure":
        sys.exit(9)
    if scenario in ("pull_blocked", "exit_during_pull"):
        if scenario == "exit_during_pull":
            os.kill(int((root / "server.pid").read_text()), signal.SIGTERM)
        while True:
            time.sleep(0.05)
    (root / "pulled").touch()
else:
    sys.exit(2)
"""


def _await_file(path: Path, process: subprocess.Popen) -> None:
    deadline = time.monotonic() + 8
    while not path.exists():
        assert process.poll() is None, "startup exited before the expected state"
        assert time.monotonic() < deadline, "startup never reached the expected state"
        time.sleep(0.05)


def _assert_children_stopped(root: Path) -> None:
    for name in ("server.pid", "pull.pid"):
        path = root / name
        if path.exists():
            with pytest.raises(ProcessLookupError):
                os.kill(int(path.read_text()), 0)


@pytest.fixture
def run_startup(tmp_path):
    services = yaml.safe_load(COMPOSE.read_text())["services"]
    service = services["ollama"]
    assert "LLM_MODEL=${LLM_MODEL:-llama3.2:3b}" in service["environment"]
    assert "LLM_MODEL=${LLM_MODEL:-llama3.2:3b}" in services["app"]["environment"]
    runtime = service["entrypoint"][2].replace("$$", "$")
    health = service["healthcheck"]["test"][1].replace("$$", "$")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    executable = bin_dir / "ollama"
    executable.write_text(f"#!{sys.executable}\n{STUB}")
    executable.chmod(0o755)
    processes = []

    def start(scenario):
        env = {
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "LLM_MODEL": "research/model:tag with spaces",
            "TMPDIR": str(tmp_path),
            "STUB_ROOT": str(tmp_path),
            "STUB_SCENARIO": scenario,
        }
        process = subprocess.Popen(
            ["/bin/sh", "-c", runtime],
            cwd=tmp_path,
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
        )
        processes.append(process)
        return process, env, health

    yield start
    for process in processes:
        if process.poll() is None:
            process.terminate()
        process.communicate(timeout=8)
    _assert_children_stopped(tmp_path)


def test_model_pull_precedes_health_and_normal_stop(run_startup, tmp_path):
    process, env, health = run_startup("normal")
    _await_file(tmp_path / "nexusrag-ollama-ready", process)

    assert (tmp_path / "pulled").exists()
    assert json.loads((tmp_path / "model.json").read_text()) == [env["LLM_MODEL"]]
    assert subprocess.run(["/bin/sh", "-c", health], env=env, check=False).returncode == 0
    process.terminate()
    process.communicate(timeout=8)

    assert process.returncode == 0
    assert not (tmp_path / "nexusrag-ollama-ready").exists()
    _assert_children_stopped(tmp_path)


@pytest.mark.parametrize("scenario", ["pull_failure", "early_exit", "exit_during_pull"])
def test_startup_failure_stops_server_and_returns_nonzero(run_startup, tmp_path, scenario):
    process, _, _ = run_startup(scenario)
    process.communicate(timeout=8)

    assert process.returncode != 0
    assert not (tmp_path / "nexusrag-ollama-ready").exists()
    _assert_children_stopped(tmp_path)


def test_stop_during_pull_cleans_up_both_processes(run_startup, tmp_path):
    process, env, health = run_startup("pull_blocked")
    _await_file(tmp_path / "pull.pid", process)

    assert subprocess.run(["/bin/sh", "-c", health], env=env, check=False).returncode != 0
    process.send_signal(signal.SIGTERM)
    process.communicate(timeout=8)

    assert process.returncode == 0
    _assert_children_stopped(tmp_path)
