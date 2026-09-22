# Vast.ai CLI command catalog

Read this when SKILL.md is not enough: extra flags, volumes, serverless, templates, teams, hosts. Facts from the [official vast-cli SKILL.md](https://github.com/vast-ai/vast-cli/blob/master/vastai/SKILL.md). Prefer `--raw` and `--full` on every command. Always `-y` on destroy.

## Instances

```bash
vastai show instances                                    # all pages, no prompts
vastai show instances --status running loading
vastai show instances --gpu-name 'RTX 4090'
vastai show instances --label training
vastai show instances --order-by start_date desc
vastai show instances --cols id,status,gpu,dph
vastai show instances --raw
vastai show instance <id>
vastai create instance <offer-id> --image pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime --disk 20 --ssh --direct
# Response includes "new_contract": <id>
vastai launch instance --gpu-name RTX_4090 --num-gpus 1 --image pytorch/pytorch
vastai start instance <id>
vastai change bid <id> --price 0.12                      # raise interruptible bid; omit --price for a winning bid
vastai stop instance <id>                                # disk preserved; disk still bills
vastai reboot instance <id>
vastai destroy instance <id> -y
vastai destroy instances <id1> <id2> -y
vastai label instance <id> --label "training-run-1"
vastai update instance <id>
vastai prepay instance <id>
vastai recycle instance <id>                             # destroy + recreate
```

### Images

`@vastai-automatic-tag` is resolved server-side — pass it unchanged. Model library: https://vast.ai/model-library

```bash
vastai/base-image:@vastai-automatic-tag          # Ubuntu base
vastai/pytorch:@vastai-automatic-tag             # PyTorch + CUDA
vastai/linux-desktop:@vastai-automatic-tag       # VNC/RDP

vastai create instance <offer-id> --image vastai/vllm:@vastai-automatic-tag --disk 40 --ssh --direct \
  --env '-e MODEL_NAME=Qwen/Qwen2.5-3B-Instruct -e HF_TOKEN=hf_xxx'

vastai create instance <offer-id> --image vastai/comfy:@vastai-automatic-tag --disk 40 --ssh --direct \
  --env '-e CHECKPOINT_MODEL=black-forest-labs/FLUX.1-schnell -e HF_TOKEN=hf_xxx'
```

### create instance flags

- `--image IMAGE` — Docker image
- `--disk DISK` — local disk GB
- `--ssh` / `--jupyter` — connection type
- `--direct` — faster direct connections
- `--label LABEL`
- `--env ENV` — e.g. `'-e TZ=UTC -p 8080:8080'`
- `--onstart FILE` / `--onstart-cmd CMD` (long scripts: file or gzip+base64)
- `--bid_price PRICE` — interruptible $/hr
- `--template_hash HASH`
- `--create-volume ID` / `--link-volume ID`
- `--cancel-unavail` — fail if no machine (vs. create stopped)

## Search

```bash
vastai search offers                                     # verified, on-demand, sorted by score
vastai search offers 'gpu_name=RTX_4090 num_gpus=1 verified=true direct_port_count>=1' -o 'dlperf_usd-'
vastai search offers 'num_gpus>=4 reliability>0.99' -o 'num_gpus-'
vastai search offers --type bid                          # interruptible
vastai search offers --type reserved
vastai search offers -n 'gpu_name=H100_SXM'             # no default filters
vastai search volumes
vastai search templates "pytorch"
vastai search benchmarks
vastai search invoices
```

Flags: `--type on-demand|reserved|bid`, `--order/-o FIELD[-]`, `--limit`, `--storage GB`, `--no-default/-n`.

Bid search exposes `min_bid`, but create still rents on-demand unless `--bid_price` is set.
`kva new` defaults to spot (`--type bid` + `--bid_price` at `min_bid + --bid-pct`, default 10%). `kva resume` raises the bid 20% (`--bid-pct`) and starts. `--on-demand` for dedicated.

## SSH and keys

```bash
vastai ssh-url <id>                                      # ssh:// URL; does not open a session
vastai scp-url <id>
vastai attach ssh <id> "ssh-ed25519 AAAA..."
vastai detach ssh <id> <ssh_key_id>                      # id from show ssh-keys
vastai show ssh-keys
vastai create ssh-key ~/.ssh/id_ed25519.pub              # BEFORE create instance
vastai create ssh-key                                    # generate if missing
vastai create ssh-key "ssh-ed25519 AAAA..."
vastai delete ssh-key <id>
vastai update ssh-key <id> "ssh-ed25519 AAAA..."
```

Direct port (when `--direct`): `direct_port_end` in instance JSON.

## Copy

```bash
vastai copy local:./data/ <id>:/workspace/data/
vastai copy <id>:/workspace/results/ local:./results/
vastai copy <id-a>:/workspace/ <id-b>:/workspace/
vastai copy s3.101:/data/ C.<id>:/workspace/
vastai copy V.1234:/file C.<id>:/workspace/
vastai copy V.1234:/file s3.101:/workspace/
# legacy: vastai copy 12345:./data ./local-data
vastai cloud copy --src ./data --dst s3://bucket/path \
  --instance 12345 --connection <conn-id> \
  --transfer "Instance To Cloud"
vastai cancel copy <dst-id>
```

Location formats: `[instance_id:]path` (legacy), `C.instance_id:path`, `V.volume_id:path`, `cloud_service[.id]:path`, `local:path`. Volume copy is to volumes, instances, or cloud — not local. Never copy to `/root` or `/`.

Cloud copy flags: `--src`, `--dst`, `--instance`, `--connection`, `--transfer`. Add the cloud connection in console settings first.

## Logs and execute

```bash
vastai logs <id>
vastai logs <id> --tail 100
vastai logs <id> --filter "error"
vastai execute <id> 'ls /workspace'
vastai execute <id> 'ls /workspace' --schedule DAILY
```

`execute` is documented as constrained (`ls`, `rm`, `du`) plus optional `--schedule HOURLY|DAILY|WEEKLY`. Use SSH for anything else (Python, pip, `nvidia-smi`).

## Volumes

```bash
vastai search volumes
vastai show volumes
vastai create volume <offer_id> [-s SIZE] [-n NAME]
vastai clone volume <source_id> <dest_id> [-s SIZE]
vastai delete volume <id>
vastai create network-volume ...
vastai list network-volume
vastai take snapshot <instance_id> --repo REPO --docker_login_user USER --docker_login_pass PASS
```

## Serverless and deployments

```bash
vastai show endpoints
vastai create endpoint --name "my-ep" ...
vastai update endpoint <id> ...
vastai delete endpoint <id>
vastai get endpt-logs <id>

vastai show workergroups
vastai create workergroup --name "wg" ...
vastai update workergroup <id> ...
vastai update workers <id>
vastai delete workergroup <id>
vastai get wrkgrp-logs <id>

vastai show deployments
vastai show deployment <id>
vastai show deployment-versions <id>
vastai delete deployment <id>

vastai show scheduled-jobs
vastai delete scheduled-job <id>
```

## Templates

```bash
vastai search templates "pytorch"
vastai create template --name "x" --image "img"
vastai update template <id> ...
vastai delete template <id>
```

## Account and API keys

```bash
vastai set api-key <key>
vastai show api-key <id>
vastai show api-keys
vastai create api-key --name "ci" --permissions '{...}'
vastai delete api-key <id>
vastai reset api-key
vastai show user
vastai show audit-logs
vastai show connections
vastai show ipaddrs
```

## Billing

```bash
vastai show invoices-v1
vastai show invoices-v1 --charges
vastai show invoices-v1 --invoices
vastai show invoices-v1 --start-date 2026-01-01 --end-date 2026-02-01
vastai show invoices-v1 --limit 50 --latest-first
vastai show deposit <id>
```

## Teams

```bash
vastai create team --name "myteam"
vastai show members
vastai invite member --email user@example.com
vastai remove member <id>
vastai create team-role --name "viewer" ...
vastai show team-role <id>
vastai show team-roles
vastai update team-role <id> ...
vastai remove team-role <id>
vastai destroy team
```

## Environment variables

```bash
vastai show env-vars
vastai create env-var KEY val
vastai update env-var KEY newval
vastai delete env-var KEY
```

## Machine management (hosts)

```bash
vastai show machines
vastai list machine <id>
vastai show machine <id>
vastai cleanup machine <id>
vastai schedule maint <id> ...
vastai cancel maint <id>
vastai show maints
vastai unlist machine <id>
```

## URLs

```
https://console.vast.ai/instances/
https://console.vast.ai/create/
https://console.vast.ai/manage-keys/
https://cloud.vast.ai/billing/

https://console.vast.ai/api/v0/instances/
https://console.vast.ai/api/v0/asks/
```
