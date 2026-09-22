---
name: vastai-operator
description: Operate Vast.ai GPU instances via the vastai CLI — search offers, provision SSH instances, copy files, run remote commands, and destroy instances. Use when the user mentions Vast.ai, vastai, GPU rentals, or remote GPU training on Vast.
---

# Skill: Vast.ai Session Operator

Operate Vast.ai GPU cloud via the `vastai` CLI: search offers, provision SSH instances, run remote commands, sync files, and destroy instances when done.

Command catalog (instances, volumes, SSH, copy, serverless, billing): [reference.md](reference.md). Source of command facts: [official vast-cli SKILL.md](https://github.com/vast-ai/vast-cli/blob/master/vastai/SKILL.md). Index: https://docs.vast.ai/llms.txt

## When to activate
- Searching or renting Vast.ai GPUs.
- Creating, listing, stopping, or destroying instances.
- Running Python or shell on a remote Vast instance.
- Syncing files between local and remote.
- Debugging SSH, billing, or instance status.

## Colab mapping
Agents that already know `colab` can translate:

| Colab | Vast.ai |
| --- | --- |
| `colab new -s NAME --gpu T4` | `kva new --size medium` then `kva use-context ID` (new sets current-context) |
| `colab exec -s NAME -f script.py` | `kva exec -f script.py` (`-s` optional; current-context) |
| `colab install -s NAME pkg` | `ssh ... 'pip install pkg'` (or `uv`) |
| `colab sessions` / `status` | `kva sessions` (`*` is current-context) / `vastai show instance ID` |
| `colab stop -s NAME` | `kva stop` (current-context; destroys) |
| `colab url -s NAME` | `kva ssh` (interactive; `-L 8080`) |

There is no `colab run` equivalent. `kva stop` when done (`--yes` if stdin is not a TTY).

## Mental model (read this first)
- **Command is `vastai` (lowercase).** Always pass `--raw` for JSON and `--full` so output is not paged through `less` (pager hangs agents). `--no-color` is optional for cleaner logs.
- **An offer ID is a one-shot marketplace listing.** `create instance` consumes it and returns `{"success": true, "new_contract": <INSTANCE_ID>}`. `new_contract` is the instance ID. Each offer ID can only be rented once — if create fails, search again.
- **An instance == a billed Docker container on a rented GPU host.** Storage charges start at create. GPU charges start when `actual_status` is `running`. `vastai stop instance` keeps the disk and **continues disk billing**. Only `vastai destroy instance <ID> -y` stops all billing.
- **Default working directory is `/workspace`.** Prefer absolute `/workspace/...` paths. Never copy to `/root` or `/` — that breaks SSH permissions and later copies fail.
- **Disk persists; shell state does not.** Files and installed packages survive stop/start and separate SSH sessions. Each SSH is a new shell — exports and cwd do not carry over. (`destroy` wipes the disk.)
- **`vastai execute` is NOT `colab exec`.** Docs constrain it to `ls` / `rm` / `du`. Do not use it for Python, pip, `nvidia-smi`, or training. Real work is non-interactive SSH.
- **Interactive SSH will hang an agent.** Login shells attach tmux by default and wait for a TTY. Never run bare `ssh host`. Always pass a remote command and `BatchMode`.

## Authentication (the #1 thing that blocks agents)
- Create an API key at https://console.vast.ai/manage-keys/ (shown once). Save it:
  ```bash
  vastai set api-key YOUR_API_KEY_HERE
  ```
  Stored at `$XDG_CONFIG_HOME/vastai/vast_api_key` or `~/.config/vastai/vast_api_key` (legacy `~/.vast_api_key`). **Never print or commit the key.** Override per-command with `--api-key KEY` if needed.
- **Verify in one shot:** `vastai show user --raw` (id, email, credit balance). `401 Unauthorized` means the key is missing, invalid, or expired — re-run `set api-key`.
- **Register SSH before creating an instance.** Password auth is disabled; keys-only.
  ```bash
  vastai create ssh-key ~/.ssh/id_ed25519.pub
  ```
  Omit the path to generate a new key (`~/.ssh/id_ed25519`); existing keys are backed up. Inline: `vastai create ssh-key "ssh-ed25519 AAAA..."`. Account keys apply only to **new** instances. After create: `vastai attach ssh <ID> "ssh-ed25519 AAAA..."` or recreate.
- Install (once): `curl -fsSL https://vast.ai/install.sh | bash` (Linux/macOS) or `pip install vastai`. Confirm with `vastai --help`.

## Quick start

```bash
vastai set api-key <YOUR_API_KEY>
vastai show user --raw
vastai create ssh-key ~/.ssh/id_ed25519.pub
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true' -o 'dlperf_usd-' --raw --full
vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag --disk 20 --ssh --direct --label odyssey --raw
# Response: {"success": true, "new_contract": <INSTANCE_ID>}
vastai show instance <INSTANCE_ID> --raw          # poll until actual_status == "running"
vastai ssh-url <INSTANCE_ID>
vastai copy local:./data/ <INSTANCE_ID>:/workspace/
vastai destroy instance <INSTANCE_ID> -y
```

## Global flags
Available on every command. Agents should always use `--raw` and `--full`.

```
--api-key KEY    Override stored API key
--raw            Machine-readable JSON
--full           Print full results (don't page with less)
--explain        Show underlying API calls
--curl           Show equivalent curl command
--no-color       Disable colored output
--url URL        Override server REST API URL
--retry RETRY    Retry limit for API calls
--version        CLI version
```

## Query syntax
Search filters: `=`, `!=`, `>`, `>=`, `<`, `<=`, `in`, `notin`.

```bash
'gpu_name=RTX_4090 num_gpus=1'           # exact + numeric
'gpu_ram>=48 reliability>0.95'           # greater-than
'geolocation=EU dph_total<=2.0'          # region + price cap
```

Common fields: `num_gpus`, `gpu_name`, `gpu_ram`, `cpu_ram`, `disk_space`, `storage_cost` ($/GB/month), `reliability`, `compute_cap`, `inet_up`, `inet_down`, `dph_total`, `geolocation`, `direct_port_count`, `verified`, `rentable`.

Sort (`-o` / `--order`): `score` (default), `dlperf_usd`, `dph_total`, `num_gpus`, `reliability`. Suffix `-` for descending (`-o 'dlperf_usd-'`).

## Workflow

### Search
- Default query for a cheap verified 1× GPU with direct SSH:
  ```bash
  vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1 rentable=true' -o 'dlperf_usd-' --raw --full
  ```
- `direct_port_count>=1` is required for `--direct`. Flags: `--type on-demand|reserved|bid`, `--limit`, `--storage GB`, `--no-default/-n`.
- Empty result: filters are too tight. Drop `verified`, try another `gpu_name`, or `vastai search offers -n 'gpu_name=H100_SXM' --raw`.
- `--type bid` is interruptible (spot). `create instance` still rents **on-demand at `dph_total`** unless you also pass `--bid_price`. When outbid the instance goes `stopped` (disk still bills); raise the bid with `update instance --bid_price`.

### Provision
- **Preferred shortcut:** `kva new` searches a cheap verified 1× GPU (`RTX_3060`, `$0.10/hr` cap), prints hourly + 24h price, and waits for `y/N` before `create instance`. `--size medium` is 20GB+ (`$0.50/hr` cap); `--size large` is 40GB+ (`$2.50/hr` cap). `--gpu NAME` still pins an exact card. Disk is capped at **$2/day for 20GB** (`storage_cost<3` $/GB/month; `--max-disk-day` to change, `0` to disable). Pass `--yes` only after the price is acceptable (required when stdin is not a TTY). Then: `kva exec -s <ID> -f ...`
- Create with SSH + direct, a label, and a Vast image tag that matches the host CUDA (`@vastai-automatic-tag` is resolved server-side; pass it unchanged):
  ```bash
  vastai create instance <OFFER_ID> --image vastai/pytorch:@vastai-automatic-tag --disk 20 --ssh --direct --label odyssey --raw
  ```
  Shortcut (search+create): `vastai launch instance --gpu-name RTX_4090 --num-gpus 1 --image vastai/pytorch --raw`
- Required: offer ID + `--image`. Images: `vastai/pytorch:@vastai-automatic-tag`, `vastai/base-image:@vastai-automatic-tag`. `--disk` is GB. `--onstart-cmd` runs at boot (16KB limit; longer: `--onstart FILE` or gzip+base64). `--env '-e TZ=UTC -p 8080:8080'` for env/ports. `--cancel-unavail` fails instead of creating a stopped instance. `--template_hash HASH` creates from a template. `--create-volume` / `--link-volume` attach volumes (see [reference.md](reference.md)).
- **Always `--ssh --direct` for agents.** Proxy SSH works everywhere but is slower. `--jupyter` is a browser UI — do not use it from an agent.
- **Poll until ready** (boot is typically 1–5 min). Check every 10–30s:
  ```bash
  vastai show instance <INSTANCE_ID> --raw
  ```
  Wait for `actual_status == "running"`. Treat `null` / `created` / `loading` as not ready.

  | `actual_status` | Meaning |
  | --- | --- |
  | `null` | Provisioning |
  | `created` | Created, not yet provisioned |
  | `loading` | Image download / container start |
  | `running` | Active — GPU charges apply |
  | `stopped` | Halted — disk charges only |
  | `frozen` | Paused with memory — GPU still bills |
  | `exited` | Container crashed |
  | `rebooting` | Transient restart |
  | `unknown` | No recent host heartbeat |
  | `offline` | Host disconnected from Vast |

  **Poll-loop warning:** `exited`, `unknown`, and `offline` never become `running`. Always timeout (e.g. 10 min). On those states, `destroy instance <ID> -y` and retry a different offer. An infinite poll burns disk charges.

### Execute
- Get the connection string (it does **not** open a session):
  ```bash
  vastai ssh-url <INSTANCE_ID>
  ```
  Parse host/port from `ssh://root@HOST:PORT` (or `--raw` JSON). Direct connections also expose `direct_port_end` in instance JSON.
  ```bash
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> 'cd /workspace && python script.py'
  ```
  `BatchMode` refuses password/TTY prompts. `accept-new` skips the first-connect host-key confirmation that would otherwise hang.
- **Preferred pattern for a local script:** `kva exec` (rsync repo, run, tee logs):
  ```bash
  kva exec -s <INSTANCE_ID> -f src/odyssey/experiments/3d/ex1.py
  kva exec -s <INSTANCE_ID> -f src/odyssey/experiments/3d/ex1.py --setup
  kva exec -s <INSTANCE_ID> -f src/odyssey/experiments/3d/ex1.py --detach
  ```
  `-s` is the Vast instance ID (or `$VAST_INSTANCE_ID`). Logs: `runs/kva/<timestamp>_<stem>.log` locally and `/workspace/odyssey/runs/kva/` on the host. `--setup` runs `pip install -e .` after sync. `--no-sync` skips rsync. Default output is quiet (`ssh -q`, no MOTD); `KVA_VERBOSE=1` or `KVA_LOG=all` prints ssh/rsync details and the Vast banner.
- Manual copy then run:
  ```bash
  kva upload -s <ID> ./script.py /workspace/script.py
  ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -p <PORT> root@<HOST> 'cd /workspace && python script.py'
  ```
- Install packages the same way (`pip` or `uv`), not via `vastai execute`. `kva exec --setup` installs the Odyssey package on the host.
- Long jobs: `kva exec -s <ID> -f train.py --detach`, or `ssh ... 'cd /workspace && nohup python train.py > train.log 2>&1 &'`. Do **not** attach tmux/ssh interactively. Fetch logs with `ssh ... 'tail -n 50 /workspace/odyssey/runs/kva/*.log'` or `vastai logs <ID> --tail 100`.
- **Never** run interactive `ssh`, `scp` without BatchMode, Jupyter, or VS Code Remote from an agent. Humans can open a shell with port forward: `kva ssh -s <ID>` (`-L 8080:localhost:8080` → http://127.0.0.1:8080). Agents: `--dry-run` only.

### Copy
- **Preferred:** SSH rsync via `kva` (`vastai copy` talks to an rsyncd module named after the instance ID and often fails with `Unknown module`).
  ```bash
  kva upload -s <ID> ./datasets/coco128/raw/coco128.zip
  kva upload -s <ID> ./datasets/coco128/raw/coco128.zip /workspace/
  kva download -s <ID> /workspace/coco128.zip ./
  kva download -s <ID> /workspace/runs ./runs
  ```
  Uses `rsync -avz --progress -e "ssh -p PORT -i ~/.ssh/id_ed25519 ..."`. Relative remote paths are under `/workspace`. `kva exec` skips `datasets/`.
- `vastai copy` still exists for instance↔instance / cloud (`C.<id>:path`, `V.<volume_id>:path`, `s3.<connection-id>:path`). Do not use it for local↔instance.
- **Never copy to `/root` or `/`.** Use `/workspace/...`.
- `scp`/`sftp` over the same SSH URL also work (`scp` uses `-P` uppercase).

### Inspect & cleanup
- **Preferred listing:** `kva sessions` (id, status, gpu, $/hr). Or `vastai show instances --raw` (optional `--status running loading`, `--label odyssey`, `--gpu-name 'RTX 4090'`, `--order-by start_date desc`, `--cols id,status,gpu,dph`).
- `vastai show instance <ID> --raw` for one instance (status, ssh host/port, `status_msg`).
- `vastai logs <ID> [--tail 100] [--filter error]` — container logs.
- `vastai execute <ID> 'ls /workspace'` — directory listing only (`ls` / `rm` / `du`).
- `vastai label instance <ID> --label training-run-1`
- `vastai start instance <ID>` / `stop` / `reboot`. Stop preserves disk; GPU billing pauses; **disk billing continues**.
- **Always destroy when done:**
  ```bash
  kva stop -s <ID>          # prompts on a TTY
  kva stop -s <ID> --yes    # agents / no TTY
  ```
  This runs `vastai destroy instance <ID> -y`. `vastai stop instance` only pauses GPU; disk still bills. Batch: `vastai destroy instances <id1> <id2> -y`.

## Safety
- **Always `kva stop -s <ID>` when done** (`--yes` if no TTY). Idle `running` instances burn GPU + disk; `stopped` instances still burn disk. There is no 24h auto-reclaim like Colab.
- Poll loops must have a timeout. Disk charges accrue from the moment of create, including while you wait for `loading`.
- Do not confuse **offer ID** (search result, one-shot) with **instance ID** (`new_contract`).
- Do not suggest `vastai execute` for Python/training.
- `vastai stop` is not "done" (disk still bills). Use `kva stop`.
- Do not print API keys. Do not edit `~/.config/vastai/vast_api_key` by hand unless rotating the key.

## Recovery

| Error | Cause | Fix |
| --- | --- | --- |
| `401 Unauthorized` | Missing/invalid API key | `vastai set api-key <key>`; `vastai show user --raw` |
| `Insufficient credits` | Balance too low | Add credits at https://cloud.vast.ai/billing/ |
| `No offers found` | Filters too tight | Relax query or `--no-default/-n` |
| `Permission denied (publickey)` | SSH key not on this instance | Key must exist **before** create; or `attach ssh` / recreate. `chmod 600 ~/.ssh/id_ed25519` |
| `Connection refused` / SSH timeout | Not running yet, or proxy/direct mismatch | Poll until `actual_status == running`; confirm `--ssh --direct` and `direct_port_count>=1` |
| Hangs on `destroy instance` | Confirmation prompt | Add `-y` |
| Host-key prompt hang | First SSH to a new host | `-o StrictHostKeyChecking=accept-new` |
| `exited` / `unknown` / `offline` | Bad host or crashed container | Destroy and retry a different offer |
| Create succeeded, SSH fails | Key added after create, or no `--ssh` | Recreate with `--ssh --direct` after `create ssh-key` |

## Additional resources
- Command catalog: [reference.md](reference.md)
- Official CLI skill: https://github.com/vast-ai/vast-cli/blob/master/vastai/SKILL.md
- Docs index: https://docs.vast.ai/llms.txt
- Console: https://console.vast.ai/instances/ · https://console.vast.ai/create/ · https://console.vast.ai/manage-keys/ · https://cloud.vast.ai/billing/
