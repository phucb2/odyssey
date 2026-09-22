"""Tests for the kva CLI wrapper and bash usage."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from odyssey.kva import find_kva_script

REPO = Path(__file__).resolve().parents[1]
KVA = REPO / "scripts" / "kva"


def _bare_env(tmp_path: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VAST_INSTANCE_ID", None)
    env["KVA_CONFIG"] = str(tmp_path / "kva.config")
    return env


def test_find_kva_script_from_repo():
    assert find_kva_script() == KVA
    assert KVA.is_file()


def test_kva_help():
    r = subprocess.run(["bash", str(KVA), "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0
    assert "kva new" in r.stdout
    assert "kva sessions" in r.stdout
    assert "kva upload" in r.stdout
    assert "kva download" in r.stdout
    assert "kva ssh" in r.stdout
    assert "use-context" in r.stdout
    assert "current-context" in r.stdout
    assert "KVA_VERBOSE=1" in r.stdout
    assert "id_ed25519" in r.stdout


def test_kva_unknown_command():
    r = subprocess.run(["bash", str(KVA), "nope"], capture_output=True, text=True, check=False)
    assert r.returncode == 2
    assert "unknown command" in r.stderr


def test_kva_new_help():
    r = subprocess.run(["bash", str(KVA), "new", "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0
    assert "--max-dph" in r.stdout
    assert "--size" in r.stdout
    assert "--max-disk-day" in r.stdout
    assert "medium" in r.stdout
    assert "Create this instance?" not in r.stdout


def test_kva_new_unknown_size():
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "huge", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.returncode == 1
    assert "small, medium, or large" in r.stderr


def test_kva_new_size_queries():
    small = subprocess.run(
        ["bash", str(KVA), "new", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "size=small" in small.stdout
    assert "max-dph=0.1" in small.stdout
    assert "gpu_name=RTX_3060" in small.stdout
    assert "gpu_ram>=" not in small.stdout

    medium = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "size=medium" in medium.stdout
    assert "max-dph=0.5" in medium.stdout
    assert "gpu_ram>=20" in medium.stdout
    assert "gpu_name=" not in medium.stdout

    large = subprocess.run(
        ["bash", str(KVA), "new", "--size", "LARGE", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "size=large" in large.stdout
    assert "max-dph=2.5" in large.stdout
    assert "gpu_ram>=40" in large.stdout

    named = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--gpu", "L4", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "gpu_name=L4" in named.stdout
    assert "max-dph=0.5" in named.stdout
    assert "gpu_ram>=" not in named.stdout


def test_kva_new_excludes_expensive_storage():
    default = subprocess.run(
        ["bash", str(KVA), "new", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "disk=20" in default.stdout
    assert "max-disk-day=2" in default.stdout
    assert "disk_space>=20" in default.stdout
    assert "storage_cost<3.0000" in default.stdout

    bigger = subprocess.run(
        ["bash", str(KVA), "new", "--disk", "40", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "disk_space>=40" in bigger.stdout
    assert "storage_cost<1.5000" in bigger.stdout

    uncapped = subprocess.run(
        ["bash", str(KVA), "new", "--max-disk-day", "0", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "storage_cost<" not in uncapped.stdout
    assert "disk_space>=20" in uncapped.stdout


def test_kva_exec_requires_session_and_file(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "exec"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing -s" in r.stderr


def test_kva_exec_missing_script(tmp_path, monkeypatch):
    monkeypatch.chdir(REPO)
    env = _bare_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "exec", "-s", "1", "-f", str(tmp_path / "missing.py")],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 1
    assert "script not found" in r.stderr


def _fake_vastai_env(
    tmp_path: Path,
    *,
    instances_json: str,
    destroy_log: Path | None = None,
    instance_json: str | None = None,
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = destroy_log or (tmp_path / "destroy.args")
    inst = instance_json or "{}"
    script = bin_dir / "vastai"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                f"LOG = Path({str(log)!r})",
                f"INSTANCES = {instances_json!r}",
                f"INSTANCE = {inst!r}",
                "if sys.argv[1:3] == ['show', 'instances']:",
                "    print(INSTANCES)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['show', 'instance']:",
                "    print(INSTANCE)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['destroy', 'instance']:",
                "    LOG.write_text(' '.join(sys.argv[1:]))",
                "    raise SystemExit(0)",
                "print('unexpected:', ' '.join(sys.argv[1:]), file=sys.stderr)",
                "raise SystemExit(1)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    script.chmod(0o755)
    env = os.environ.copy()
    env["PATH"] = f"{bin_dir}{os.pathsep}{env['PATH']}"
    env.pop("VAST_INSTANCE_ID", None)
    env["KVA_CONFIG"] = str(tmp_path / "kva.config")
    return env


def test_kva_sessions_lists_instances(tmp_path):
    payload = json.dumps(
        [
            {
                "id": 28394812,
                "actual_status": "running",
                "gpu_name": "RTX 3060",
                "dph_total": 0.08,
                "disk_space": 20,
                "label": "odyssey",
                "geolocation": "US, California",
            },
            {
                "id": 99,
                "actual_status": "stopped",
                "gpu_name": "L4",
                "dph_total": 0.4,
                "disk_space": 40,
                "label": "exp",
                "geolocation": "EU",
            },
        ]
    )
    env = _fake_vastai_env(tmp_path, instances_json=payload)
    r = subprocess.run(
        ["bash", str(KVA), "sessions"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "kva: 2 instance(s)" in r.stdout
    assert "28394812" in r.stdout
    assert "running" in r.stdout
    assert "RTX 3060" in r.stdout
    assert "0.0800" in r.stdout
    assert "99" in r.stdout
    assert "stopped" in r.stdout


def test_kva_sessions_empty(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]")
    r = subprocess.run(
        ["bash", str(KVA), "sessions"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "kva: no instances" in r.stdout


def test_kva_sessions_wrapped_instances(tmp_path):
    payload = json.dumps(
        {
            "instances": [
                {
                    "id": 7,
                    "actual_status": "loading",
                    "gpu_name": "RTX 4090",
                    "dph_total": 0.5,
                    "disk_space": 20,
                    "label": "odyssey",
                    "geolocation": "US",
                }
            ]
        }
    )
    env = _fake_vastai_env(tmp_path, instances_json=payload)
    r = subprocess.run(
        ["bash", str(KVA), "sessions"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "kva: 1 instance(s)" in r.stdout
    assert "7" in r.stdout
    assert "loading" in r.stdout


def test_kva_stop_requires_session(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "stop"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing -s" in r.stderr


def test_kva_stop_yes_destroys(tmp_path):
    log = tmp_path / "destroy.args"
    env = _fake_vastai_env(tmp_path, instances_json="[]", destroy_log=log)
    r = subprocess.run(
        ["bash", str(KVA), "stop", "-s", "28394812", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "destroyed instance 28394812" in r.stdout
    assert log.read_text(encoding="utf-8") == "destroy instance 28394812 -y"


def test_kva_stop_no_tty_without_yes(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]")
    r = subprocess.run(
        ["bash", str(KVA), "stop", "-s", "1"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert r.returncode == 1
    assert "without a TTY" in r.stderr


def _transfer_env(tmp_path: Path) -> dict[str, str]:
    inst = json.dumps({"public_ipaddr": "182.224.239.168", "direct_port_end": 31140})
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=inst)
    key = tmp_path / "id_ed25519"
    key.write_text("dummy\n", encoding="utf-8")
    env["KVA_SSH_IDENTITY"] = str(key)
    return env


def test_kva_upload_requires_session(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "upload", "./nope.zip"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing -s" in r.stderr


def test_kva_upload_requires_local_path(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "upload", "-s", "1"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing LOCAL" in r.stderr


def test_kva_upload_missing_file(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "upload", "-s", "1", str(tmp_path / "missing.zip")],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "local path not found" in r.stderr


def test_kva_upload_refuses_root(tmp_path):
    local = tmp_path / "coco128.zip"
    local.write_bytes(b"zip")
    r = subprocess.run(
        ["bash", str(KVA), "upload", "-s", "1", str(local), "/root"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "refusing remote path" in r.stderr


def test_kva_upload_dry_run(tmp_path):
    local = tmp_path / "coco128.zip"
    local.write_bytes(b"zip")
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "upload", "-s", "99", str(local), "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    line = r.stdout + r.stderr
    assert "rsync -avz --progress" in line
    assert "-p 31140" in line
    assert "root@182.224.239.168:/workspace/" in line
    assert str(local) in line


def test_kva_upload_relative_remote(tmp_path):
    local = tmp_path / "coco128.zip"
    local.write_bytes(b"zip")
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "upload", "-s", "99", str(local), "data/", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "root@182.224.239.168:/workspace/data/" in r.stdout


def test_kva_download_requires_remote_path(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "download", "-s", "1"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing REMOTE" in r.stderr


def test_kva_download_dry_run(tmp_path):
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        [
            "bash",
            str(KVA),
            "download",
            "-s",
            "99",
            "/workspace/coco128.zip",
            "./",
            "--dry-run",
        ],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    line = r.stdout
    assert "rsync -avz --progress" in line
    assert "-p 31140" in line
    assert "root@182.224.239.168:/workspace/coco128.zip" in line
    assert line.rstrip().endswith("./")


def test_kva_download_relative_remote(tmp_path):
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "download", "-s", "99", "runs/", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "root@182.224.239.168:/workspace/runs/" in r.stdout
    assert r.stdout.rstrip().endswith("./")


def test_kva_ssh_requires_session(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "ssh"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing -s" in r.stderr


def test_kva_ssh_dry_run_forwards_8080(tmp_path):
    env = _transfer_env(tmp_path)
    env["TERM"] = "xterm-256color"
    r = subprocess.run(
        ["bash", str(KVA), "ssh", "-s", "99", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    out = r.stdout
    assert "-L 8080:localhost:8080" in out
    assert "http://127.0.0.1:8080" in out
    assert "-p 31140" in out
    assert "root@182.224.239.168" in out
    assert " -t " in f" {out} "
    assert "terminfo" not in out


def test_kva_ssh_remaps_ghostty_term(tmp_path):
    env = _transfer_env(tmp_path)
    env["TERM"] = "xterm-ghostty"
    r = subprocess.run(
        ["bash", str(KVA), "ssh", "-s", "99", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "TERM=xterm-256color (remote has no xterm-ghostty terminfo)" in r.stdout


def test_kva_ssh_port_override(tmp_path):
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "ssh", "-s", "99", "--port", "8888", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "-L 8888:localhost:8888" in r.stdout
    assert "http://127.0.0.1:8888" in r.stdout


def test_kva_ssh_forward_spec(tmp_path):
    env = _transfer_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "ssh", "-s", "99", "-L", "8080:localhost:8888", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "-L 8080:localhost:8888" in r.stdout
    assert "http://127.0.0.1:8080" in r.stdout


def test_kva_use_and_current_context(tmp_path):
    env = _bare_env(tmp_path)
    missing = subprocess.run(
        ["bash", str(KVA), "current-context"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert missing.returncode == 1
    assert "current-context is not set" in missing.stderr

    switched = subprocess.run(
        ["bash", str(KVA), "use-context", "51997668"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert switched.returncode == 0, switched.stderr
    assert 'Switched to context "51997668"' in switched.stdout
    assert Path(env["KVA_CONFIG"]).read_text(encoding="utf-8") == "current-context=51997668\n"

    current = subprocess.run(
        ["bash", str(KVA), "current-context"],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    assert current.stdout.strip() == "51997668"

    cleared = subprocess.run(
        ["bash", str(KVA), "use-context", "--unset"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert cleared.returncode == 0, cleared.stderr
    missing2 = subprocess.run(
        ["bash", str(KVA), "current-context"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert missing2.returncode == 1


def test_kva_ssh_uses_current_context(tmp_path):
    env = _transfer_env(tmp_path)
    Path(env["KVA_CONFIG"]).write_text("current-context=99\n", encoding="utf-8")
    r = subprocess.run(
        ["bash", str(KVA), "ssh", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "-L 8080:localhost:8080" in r.stdout
    assert "root@182.224.239.168" in r.stdout


def test_kva_session_flag_overrides_context(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", destroy_log=tmp_path / "destroy.args")
    Path(env["KVA_CONFIG"]).write_text("current-context=99\n", encoding="utf-8")
    r = subprocess.run(
        ["bash", str(KVA), "stop", "-s", "28394812", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "destroy.args").read_text(encoding="utf-8") == "destroy instance 28394812 -y"
    assert Path(env["KVA_CONFIG"]).read_text(encoding="utf-8") == "current-context=99\n"


def test_kva_stop_clears_current_context(tmp_path):
    log = tmp_path / "destroy.args"
    env = _fake_vastai_env(tmp_path, instances_json="[]", destroy_log=log)
    Path(env["KVA_CONFIG"]).write_text("current-context=99\n", encoding="utf-8")
    r = subprocess.run(
        ["bash", str(KVA), "stop", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert log.read_text(encoding="utf-8") == "destroy instance 99 -y"
    assert "current-context cleared" in r.stdout
    assert not Path(env["KVA_CONFIG"]).exists()


def test_kva_sessions_marks_current_context(tmp_path):
    payload = json.dumps(
        [
            {
                "id": 28394812,
                "actual_status": "running",
                "gpu_name": "RTX 3060",
                "dph_total": 0.08,
                "disk_space": 20,
                "label": "odyssey",
                "geolocation": "US",
            },
            {
                "id": 99,
                "actual_status": "stopped",
                "gpu_name": "L4",
                "dph_total": 0.4,
                "disk_space": 40,
                "label": "exp",
                "geolocation": "EU",
            },
        ]
    )
    env = _fake_vastai_env(tmp_path, instances_json=payload)
    Path(env["KVA_CONFIG"]).write_text("current-context=99\n", encoding="utf-8")
    r = subprocess.run(
        ["bash", str(KVA), "sessions"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    lines = r.stdout.splitlines()
    assert any("99" in ln and "*" in ln for ln in lines)
    assert any("28394812" in ln and "*" not in ln for ln in lines)
