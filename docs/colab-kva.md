# Colab and kva CLI

Getting started for Google Colab (`colab`) and Vast.ai (`kva`). Use when running Odyssey on a rented GPU.

Both CLIs: provision a VM, run a local script on it, then tear it down. Always stop when done — idle machines bill.

| | Colab | Vast.ai (`kva`) |
| --- | --- | --- |
| Command | `colab` | `uv run kva` or `./scripts/kva` |
| Session | named (`-s odyssey`) | instance ID (`-s 12345678`); `kva new` sets current-context |
| Work dir | `/content` | `/workspace/odyssey` |
| How code arrives | send `notebooks/colab_odyssey.py` (embedded package) | rsync the git repo (`datasets/` skipped) |
| State | kernel persists across `exec` | disk persists; each SSH is a new shell |
| Cleanup | `colab stop -s NAME` | `kva stop` (destroys; `--yes` if no TTY) |

`-s` is optional on `kva` after `kva new` / `kva use-context ID`.

---

## Colab

Self-contained bootstrap. Colab CLI sends one file; the VM does not need the git repo.

### Once

```bash
uv tool install --force --with 'jupyter-kernel-client==0.15.0' google-colab-cli
gcloud auth application-default login \
  --scopes=openid,\
https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
colab sessions   # verify auth
```

Pin `jupyter-kernel-client==0.15.0` (1.x drops `KernelClient`). All four ADC scopes are required; missing `colaboratory` 403s keep-alive. Re-run the `gcloud` line if `colab new` fails auth.

### First run

```bash
make colab-nb                                          # writes notebooks/colab_odyssey.py + .ipynb
colab new -s odyssey                                   # add --gpu T4 when training
colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
colab stop -s odyssey
```

`--timeout 600` is required (CLI default is 30s). After `src/odyssey` edits: `make update` (refreshes `ODYSSEY_B64` only; leaves experiment cells). After changing user cells: `make colab-nb`.

One-shot (new + exec + stop):

```bash
colab run --gpu T4 --timeout 600 notebooks/colab_odyssey.py
```

### Daily loop

```bash
colab new -s odyssey --gpu T4
colab exec -s odyssey -f notebooks/colab_odyssey.py --timeout 600
colab status -s odyssey
colab url -s odyssey          # attach the Colab web UI to this VM
colab stop -s odyssey
```

GPUs: `T4`, `L4`, `G4`, `H100`, `A100`. TPUs: `v5e1`, `v6e1`. Unknown `--gpu` silently falls back to A100. Most accounts only get CPU; a `400` on `new` with an accelerator means no quota — drop `--gpu` or use `T4`.

### What the bootstrap does

Unpacks `src/odyssey` from `ODYSSEY_B64` into `/content/odyssey`, installs extras, prints `hello()` / CUDA. Add experiment code below `# --- odyssey:user-cells ---` in the `.py`, or below the `<!-- odyssey:user-cells -->` cell in the `.ipynb`.

Optional UI notebook: `colab exec -s odyssey -f notebooks/colab_odyssey.ipynb --timeout 600` writes `notebooks/colab_odyssey_output.ipynb`.

### Notes

- Kernel state survives between `colab exec` calls on the same session. `colab restart-kernel -s odyssey` resets it; `colab stop` releases the VM.
- `colab install -s odyssey pkg` for extra packages. `colab auth` / `colab drivemount` are interactive (human only).
- Always `colab stop`. Idle VMs burn compute until the 24h keep-alive cap.

---

## kva (Vast.ai)

`kva` is Odyssey’s Vast.ai analogue of `colab exec`. Requires the `vastai` CLI and an SSH key.

### Once

```bash
curl -fsSL https://vast.ai/install.sh | bash    # or: pip install vastai
vastai set api-key YOUR_API_KEY                 # https://console.vast.ai/manage-keys/
vastai show user --raw                          # id, email, credit
vastai create ssh-key ~/.ssh/id_ed25519.pub     # keys-only; must exist before create
uv sync                                         # installs the `kva` entry point
```

API key is stored at `~/.config/vastai/vast_api_key`. Never print or commit it. SSH always uses `-i ~/.ssh/id_ed25519` (`KVA_SSH_IDENTITY` to override). Register the pubkey **before** `kva new`; later keys need `vastai attach ssh` or a recreate.

### First run

```bash
kva new                         # cheapest verified 1× RTX_3060, spot, ≤ $0.10/hr
# y = rent this offer, n = next, q = quit
kva exec -f notebooks/colab_odyssey.py --setup
kva sessions                    # * marks current-context
kva stop                        # destroys; prompts on a TTY
```

`kva new` sets current-context. After that, omit `-s`. Agents / no TTY: `kva new --yes` and `kva stop --yes`.

Long job (nohup; returns immediately):

```bash
kva exec -f src/odyssey/experiments/3d/ex1.py --setup --detach
```

Logs: `runs/kva/<timestamp>_<stem>.log` locally and `/workspace/odyssey/runs/kva/` on the host.

### Sizes and billing

| `--size` | GPU | Cap `$ /hr` |
| --- | --- | --- |
| `small` (default) | `RTX_3060` | 0.10 |
| `medium` | 20GB+ VRAM | 0.50 |
| `large` | 40GB+ VRAM | 2.50 |

```bash
kva new --size medium
kva new --gpu L4 --max-dph 0.4
kva new --on-demand             # dedicated (not interruptible)
kva new --dry-run               # print search query only
```

Defaults: 20GB disk, disk cost capped at **$2/day**, spot bid **10% above `min_bid`**. `--bid-pct 0` bids the floor; `--bid-price USD` sets an exact bid (never above `--max-dph`). `--max-disk-day 0` disables the disk cap.

Spot can be outbid → instance `stopped` (disk still bills). Recover with `kva resume` (raises bid 20% and starts). `kva stop` **destroys** (GPU + disk billing stop). `vastai stop instance` only pauses GPU.

### Files

`kva exec` rsyncs the repo to `/workspace/odyssey` (skips `.venv/`, `.git/`, `datasets/`, `checkpoints/`, `runs/`). `--no-sync` skips rsync. `--setup` runs `pip install -e .` after sync.

```bash
kva upload ./datasets/coco128/raw/coco128.zip      # → /workspace/
kva download /workspace/runs ./runs
kva ssh                                            # interactive; -L 8080 → http://127.0.0.1:8080
```

Relative remote paths are under `/workspace`. Never `/` or `/root`. `kva ssh` is human-only (needs a TTY).

### Notes

- Boot is 1–5 min. `kva new` waits until `running`. `exited` / `offline` never recover — `kva stop --yes` and try another offer.
- Add credits at https://cloud.vast.ai/billing/ if create fails with insufficient credits.
- `KVA_VERBOSE=1` prints ssh/rsync lines. Context file: `~/.config/kva/config` (`KVA_CONFIG` to override).
