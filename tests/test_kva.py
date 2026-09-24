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
    assert "kva resume" in r.stdout
    assert "kva sessions" in r.stdout
    assert "kva upload" in r.stdout
    assert "kva download" in r.stdout
    assert "kva ssh" in r.stdout
    assert "use-context" in r.stdout
    assert "current-context" in r.stdout
    assert "KVA_VERBOSE=1" in r.stdout
    assert "id_ed25519" in r.stdout
    assert "/workspace/kva-exec/" in r.stdout


def test_kva_unknown_command():
    r = subprocess.run(["bash", str(KVA), "nope"], capture_output=True, text=True, check=False)
    assert r.returncode == 2
    assert "unknown command" in r.stderr


def test_kva_new_help():
    r = subprocess.run(["bash", str(KVA), "new", "--help"], capture_output=True, text=True, check=False)
    assert r.returncode == 0
    assert "--max-dph" in r.stdout
    assert "--size" in r.stdout
    assert "--spot" in r.stdout
    assert "--on-demand" in r.stdout
    assert "--bid-price" in r.stdout
    assert "--bid-pct" in r.stdout
    assert "--max-disk-day" in r.stdout
    assert "medium" in r.stdout
    assert "skip to the next" in r.stdout
    assert "quit" in r.stdout
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
    assert "type=bid" in small.stdout
    assert "bid-pct=10" in small.stdout
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


def test_kva_new_spot_is_default():
    default = subprocess.run(
        ["bash", str(KVA), "new", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "type=bid" in default.stdout

    explicit = subprocess.run(
        ["bash", str(KVA), "new", "--spot", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "type=bid" in explicit.stdout


def test_kva_new_on_demand_dry_run():
    r = subprocess.run(
        ["bash", str(KVA), "new", "--on-demand", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "type=on-demand" in r.stdout
    assert "type=bid" not in r.stdout


def test_kva_new_bid_pct_dry_run():
    bumped = subprocess.run(
        ["bash", str(KVA), "new", "--bid-pct", "25", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "bid-pct=25" in bumped.stdout
    assert "type=bid" in bumped.stdout

    floor = subprocess.run(
        ["bash", str(KVA), "new", "--bid-pct", "0", "--dry-run"],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "bid-pct=0" in floor.stdout


def _offer(
    oid: int,
    *,
    gpu: str = "RTX 5090",
    dph: float = 0.3,
    ram: float = 32,
) -> dict:
    return {
        "id": oid,
        "gpu_name": gpu,
        "dph_total": dph,
        "min_bid": dph,
        "geolocation": "US",
        "reliability": 0.99,
        "inet_down": 600,
        "gpu_ram": ram,
        "storage_cost": 0.2,
    }


def _kva_new_env(tmp_path: Path, offers: list, **kwargs) -> dict[str, str]:
    running = json.dumps(
        {"id": 777, "actual_status": "running", "gpu_name": "RTX 5090", "dph_total": 0.3}
    )
    env = _fake_vastai_env(
        tmp_path,
        instances_json="[]",
        instance_json=running,
        running_json=running,
        search_json=json.dumps(offers),
        **kwargs,
    )
    key = tmp_path / "id_ed25519"
    key.write_text("dummy\n", encoding="utf-8")
    env["KVA_SSH_IDENTITY"] = str(key)
    return env


def test_kva_new_on_demand_passes_type(tmp_path):
    search = tmp_path / "search.args"
    env = _kva_new_env(tmp_path, [_offer(11)], search_log=search)
    r = subprocess.run(
        ["bash", str(KVA), "new", "--on-demand", "--size", "medium", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "--type on-demand" in search.read_text(encoding="utf-8")
    assert "created instance 777" in r.stdout
    assert "--bid_price" not in (tmp_path / "create.args").read_text(encoding="utf-8")


def test_kva_new_yes_takes_first_offer(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(
        tmp_path,
        [_offer(11, gpu="RTX 5090"), _offer(22, gpu="L40S")],
        create_log=create,
    )
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "offer 1/2" in r.stdout
    assert "create instance 11" in create.read_text(encoding="utf-8")
    assert "--bid_price 0.3300" in create.read_text(encoding="utf-8")
    assert "create instance 22" not in create.read_text(encoding="utf-8")


def test_kva_new_bid_pct_applied(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(tmp_path, [_offer(11, dph=0.3)], create_log=create)
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--bid-pct", "20", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "min $0.3000 +20%" in r.stdout
    assert "--bid_price 0.3600" in create.read_text(encoding="utf-8")


def test_kva_new_bid_price_overrides_pct(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(tmp_path, [_offer(11, dph=0.3)], create_log=create)
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--bid-pct", "50", "--bid-price", "0.15", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "--bid_price 0.1500" in create.read_text(encoding="utf-8")


def test_kva_new_bid_pct_capped_at_max_dph(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(tmp_path, [_offer(11, dph=0.3)], create_log=create)
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--max-dph", "0.32", "--bid-pct", "50", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "--bid_price 0.3200" in create.read_text(encoding="utf-8")


def test_kva_new_n_selects_next_offer(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(
        tmp_path,
        [_offer(11, gpu="RTX 5090"), _offer(22, gpu="L40S", dph=0.4)],
        create_log=create,
    )
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input="n\ny\n",
    )
    assert r.returncode == 0, r.stderr
    assert "skipping offer 11" in r.stdout
    assert "offer 2/2" in r.stdout
    assert "L40S" in r.stdout
    text = create.read_text(encoding="utf-8")
    assert "create instance 22" in text
    assert "create instance 11" not in text


def test_kva_new_q_quits(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(tmp_path, [_offer(11), _offer(22)], create_log=create)
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        input="q\n",
    )
    assert r.returncode == 1
    assert "cancelled" in r.stderr
    assert not create.exists()


def test_kva_new_taken_offer_tries_next(tmp_path):
    create = tmp_path / "create.args"
    env = _kva_new_env(
        tmp_path,
        [_offer(11), _offer(22, gpu="L40S")],
        create_log=create,
        fail_create_ids="11",
    )
    r = subprocess.run(
        ["bash", str(KVA), "new", "--size", "medium", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "no longer available" in r.stdout
    text = create.read_text(encoding="utf-8")
    assert "create instance 11" in text
    assert "create instance 22" in text
    assert "created instance 777" in r.stdout


def test_kva_resume_requires_session(tmp_path):
    r = subprocess.run(
        ["bash", str(KVA), "resume"],
        capture_output=True,
        text=True,
        check=False,
        env=_bare_env(tmp_path),
    )
    assert r.returncode == 1
    assert "missing -s" in r.stderr


def _stopped_instance_json(*, dph: float = 0.08, is_bid: bool = True) -> str:
    return json.dumps(
        {
            "id": 99,
            "actual_status": "stopped",
            "gpu_name": "RTX 3060",
            "dph_total": dph,
            "is_bid": is_bid,
        }
    )


def _running_instance_json(*, dph: float = 0.08) -> str:
    return json.dumps(
        {
            "id": 99,
            "actual_status": "running",
            "gpu_name": "RTX 3060",
            "dph_total": dph,
            "is_bid": True,
        }
    )


def test_kva_resume_dry_run_raises_bid(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=_stopped_instance_json())
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "current bid: $0.0800 / hour" in r.stdout
    assert "new bid:     $0.0960 / hour" in r.stdout
    assert "change bid 99 --price 0.0960" in r.stdout
    assert "start instance 99" in r.stdout
    assert not (tmp_path / "change.args").exists()
    assert not (tmp_path / "start.args").exists()


def test_kva_resume_bid_price_override(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=_stopped_instance_json())
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99", "--bid-price", "0.15", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "new bid:     $0.1500 / hour" in r.stdout
    assert "change bid 99 --price 0.1500" in r.stdout


def test_kva_resume_bid_pct(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=_stopped_instance_json())
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99", "--bid-pct", "50", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "new bid:     $0.1200 / hour" in r.stdout
    assert "change bid 99 --price 0.1200" in r.stdout


def test_kva_resume_already_running(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=_running_instance_json())
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99", "--dry-run"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "already running ($0.0800/hr)" in r.stdout
    assert "change bid" not in r.stdout


def test_kva_resume_yes_changes_bid_and_starts(tmp_path):
    change = tmp_path / "change.args"
    start = tmp_path / "start.args"
    env = _fake_vastai_env(
        tmp_path,
        instances_json="[]",
        instance_json=_stopped_instance_json(),
        running_json=_running_instance_json(dph=0.096),
        change_log=change,
        start_log=start,
    )
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99", "--yes"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "bid 99 -> $0.0960/hr" in r.stdout
    assert "starting instance 99" in r.stdout
    assert "status=running" in r.stdout
    assert "change bid 99 --price 0.0960" in change.read_text(encoding="utf-8")
    assert "start instance 99" in start.read_text(encoding="utf-8")


def test_kva_resume_no_tty_without_yes(tmp_path):
    env = _fake_vastai_env(tmp_path, instances_json="[]", instance_json=_stopped_instance_json())
    r = subprocess.run(
        ["bash", str(KVA), "resume", "-s", "99"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
        stdin=subprocess.DEVNULL,
    )
    assert r.returncode == 1
    assert "without a TTY" in r.stderr


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


def _fake_ssh_rsync_env(tmp_path: Path) -> dict[str, str]:
    env = _transfer_env(tmp_path)
    bin_dir = Path(env["PATH"].split(os.pathsep)[0])
    ssh_log = tmp_path / "ssh.args"
    rsync_log = tmp_path / "rsync.args"
    ssh = bin_dir / "ssh"
    ssh.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                f"LOG = Path({str(ssh_log)!r})",
                "prev = LOG.read_text() if LOG.exists() else ''",
                "LOG.write_text(prev + ' '.join(sys.argv[1:]) + chr(10))",
                "raise SystemExit(0)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    ssh.chmod(0o755)
    rsync = bin_dir / "rsync"
    rsync.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                f"LOG = Path({str(rsync_log)!r})",
                "prev = LOG.read_text() if LOG.exists() else ''",
                "LOG.write_text(prev + ' '.join(sys.argv[1:]) + chr(10))",
                "raise SystemExit(0)",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    rsync.chmod(0o755)
    env["KVA_SSH_LOG"] = str(ssh_log)
    env["KVA_RSYNC_LOG"] = str(rsync_log)
    return env


def test_kva_exec_uploads_absolute_outside_path(tmp_path):
    script = tmp_path / "train.py"
    script.write_text("print(1)\n", encoding="utf-8")
    env = _fake_ssh_rsync_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "exec", "-s", "99", "-f", str(script), "--no-sync"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    assert "outside the repo" not in r.stderr
    rsync_log = Path(env["KVA_RSYNC_LOG"]).read_text(encoding="utf-8")
    assert str(script) in rsync_log
    assert "/workspace/kva-exec/train.py" in rsync_log
    ssh_log = Path(env["KVA_SSH_LOG"]).read_text(encoding="utf-8")
    assert "python3 -u /workspace/kva-exec/train.py" in ssh_log


def test_kva_exec_cwd_relative_outside_path(tmp_path, monkeypatch):
    script = tmp_path / "notebooks"
    script.mkdir()
    py = script / "colab_rsna.py"
    py.write_text("print(1)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    env = _fake_ssh_rsync_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "exec", "-s", "99", "-f", "notebooks/colab_rsna.py", "--no-sync"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    rsync_log = Path(env["KVA_RSYNC_LOG"]).read_text(encoding="utf-8")
    assert str(py.resolve()) in rsync_log
    assert "/workspace/kva-exec/colab_rsna.py" in rsync_log


def test_kva_exec_repo_relative_skips_extra_upload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = _fake_ssh_rsync_env(tmp_path)
    r = subprocess.run(
        ["bash", str(KVA), "exec", "-s", "99", "-f", "notebooks/colab_odyssey.py", "--no-sync"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert r.returncode == 0, r.stderr
    rsync_log = Path(env["KVA_RSYNC_LOG"])
    assert not rsync_log.exists() or "/workspace/kva-exec/" not in rsync_log.read_text(encoding="utf-8")
    ssh_log = Path(env["KVA_SSH_LOG"]).read_text(encoding="utf-8")
    assert "python3 -u notebooks/colab_odyssey.py" in ssh_log


def _fake_vastai_env(
    tmp_path: Path,
    *,
    instances_json: str,
    destroy_log: Path | None = None,
    instance_json: str | None = None,
    start_log: Path | None = None,
    change_log: Path | None = None,
    running_json: str | None = None,
    search_json: str | None = None,
    search_log: Path | None = None,
    create_log: Path | None = None,
    fail_create_ids: str = "",
) -> dict[str, str]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = destroy_log or (tmp_path / "destroy.args")
    start = start_log or (tmp_path / "start.args")
    change = change_log or (tmp_path / "change.args")
    search = search_log or (tmp_path / "search.args")
    create = create_log or (tmp_path / "create.args")
    state = tmp_path / "instance.state"
    inst = instance_json or "{}"
    running = running_json or inst
    offers = search_json or "[]"
    script = bin_dir / "vastai"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                f"LOG = Path({str(log)!r})",
                f"START = Path({str(start)!r})",
                f"CHANGE = Path({str(change)!r})",
                f"SEARCH = Path({str(search)!r})",
                f"CREATE = Path({str(create)!r})",
                f"STATE = Path({str(state)!r})",
                f"INSTANCES = {instances_json!r}",
                f"INSTANCE = {inst!r}",
                f"RUNNING = {running!r}",
                f"OFFERS = {offers!r}",
                f"FAIL_IDS = {{x for x in {fail_create_ids!r}.split(',') if x}}",
                "if sys.argv[1:3] == ['show', 'instances']:",
                "    print(INSTANCES)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['show', 'instance']:",
                "    print(RUNNING if STATE.exists() else INSTANCE)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['destroy', 'instance']:",
                "    LOG.write_text(' '.join(sys.argv[1:]))",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['change', 'bid']:",
                "    CHANGE.write_text(' '.join(sys.argv[1:]))",
                "    print('{\"success\": true}')",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['start', 'instance']:",
                "    START.write_text(' '.join(sys.argv[1:]))",
                "    STATE.write_text('running')",
                "    print('{\"success\": true}')",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['search', 'offers']:",
                "    SEARCH.write_text(' '.join(sys.argv[1:]))",
                "    print(OFFERS)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['create', 'instance']:",
                "    oid = sys.argv[3] if len(sys.argv) > 3 else ''",
                "    prev = CREATE.read_text() if CREATE.exists() else ''",
                "    CREATE.write_text(prev + ' '.join(sys.argv[1:]) + chr(10))",
                "    if oid in FAIL_IDS:",
                "        print('{\"error\": true, \"status_code\": 410, \"msg\": \"no_such_ask gone\"}')",
                "        raise SystemExit(0)",
                "    print('{\"success\": true, \"new_contract\": 777}')",
                "    STATE.write_text('running')",
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
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log = destroy_log or (tmp_path / "destroy.args")
    start = start_log or (tmp_path / "start.args")
    change = change_log or (tmp_path / "change.args")
    state = tmp_path / "instance.state"
    inst = instance_json or "{}"
    running = running_json or inst
    script = bin_dir / "vastai"
    script.write_text(
        "\n".join(
            [
                "#!/usr/bin/env python3",
                "import sys",
                "from pathlib import Path",
                f"LOG = Path({str(log)!r})",
                f"START = Path({str(start)!r})",
                f"CHANGE = Path({str(change)!r})",
                f"STATE = Path({str(state)!r})",
                f"INSTANCES = {instances_json!r}",
                f"INSTANCE = {inst!r}",
                f"RUNNING = {running!r}",
                "if sys.argv[1:3] == ['show', 'instances']:",
                "    print(INSTANCES)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['show', 'instance']:",
                "    print(RUNNING if STATE.exists() else INSTANCE)",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['destroy', 'instance']:",
                "    LOG.write_text(' '.join(sys.argv[1:]))",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['change', 'bid']:",
                "    CHANGE.write_text(' '.join(sys.argv[1:]))",
                "    print('{\"success\": true}')",
                "    raise SystemExit(0)",
                "if sys.argv[1:3] == ['start', 'instance']:",
                "    START.write_text(' '.join(sys.argv[1:]))",
                "    STATE.write_text('running')",
                "    print('{\"success\": true}')",
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
