# Money20/20 demo stack — operating notes

Booth demos for Money20/20 Middle East, built to run entirely on one NVIDIA GB10
(Grace Blackwell, aarch64, 121 GB unified memory, sm_121).

Everything is driven from the launcher at the repo root:

```bash
make list                # every demo, its alias, and whether it is running
make <alias>             # start one demo (stops the others first)
make <alias>-down        # stop one
make stop                # stop all demos
```

| Demo | Alias | Port |
|---|---|---|
| financial-fraud-detection | `fraud` | 8090 |
| portfolio-optimization | `portfolio` | 8501 |
| doc-element-extraction | `docs` | — |
| loan-automation | `loan` | — |
| vss-video-analytics | `vss` | 3000 |
| quant-signal-discovery | `quant` | — |
| nemotron-voice-agent | `voice` | — |

The showcase site (`showcase/`, port 8088) is nginx over a read-only bind mount:
~12 MB, no GPU. It does not compete with any demo and can stay up permanently.
Its demo links adapt to whatever host you browse from via `location.hostname`,
so they need no editing when the network changes.

## One demo at a time

All demos share a single GB10, so `make <alias>` stops the others before
starting the requested one. Demos are auto-discovered — any subdirectory with a
`Makefile` — with two optional one-line files:

- `.alias` — the short name used by `make <alias>`
- `.project` — the compose project name, when it cannot be inferred from the
  directory name. Needed for `vss-video-analytics` (runs as `mdx`) and
  `doc-element-extraction` (its `.env` sets `COMPOSE_PROJECT_NAME=name`, an
  upstream placeholder that was never changed — `name` is correct, not a bug).

## The GB10 is a shared box

**This is the single most common cause of a demo failing to start.**

Containers unrelated to this repo auto-start at boot and consume nearly all
memory. Measured 44 minutes after a reboot with nothing deliberately started:
**119 GB of 121 GB used**. The launcher only knows about demos declared here, so
it will never stop them, and its `_gpu_free` target only *prints* a status line —
it frees nothing.

The mechanism is entirely Docker restart policies (`docker.service` is enabled;
there is no `@reboot` cron, no systemd unit invoking docker, no `/etc/rc.local`).
`always` restarts a container **even if you manually stopped it** before
shutdown; `unless-stopped` does not.

Diagnose before blaming a demo:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
# map a PID to its container:
grep -oE '[0-9a-f]{64}' /proc/<pid>/cgroup | head -1 | xargs docker inspect --format '{{.Name}}'
```

Do **not** use cgroup `memory.current` for this — it reported ~3.8 GB while
114 GB was actually in use, because GPU/unified allocations are not accounted
there.

**vLLM servers claim a fraction of currently *free* memory at startup, so
stopping one hog just feeds the next.** Stop every competitor at once, then start
the demo. To disable an autostart durably without deleting anything:
`docker update --restart=no <container>` (reversible). A plain `docker stop`
holds only until the next reboot.

## VSS: config is baked in at container create time

VSS (`vss-video-analytics`, compose project `mdx`, wrapping `~/APPS/VSS-3x`)
builds its browser-facing URLs from env vars **fixed into each container when it
is created**. A network change therefore survives restarts and shows up as
websocket failures that look like a VSS bug — the UI loads, but its websocket
dials the old address.

- Source of truth: `deploy/docker/developer-profiles/dev-profile-<profile>/generated.env`
- `HOST_IP` — container→host traffic (kafka, redis, VST). Use the LAN address.
- `EXTERNAL_IP` — browser-facing (`VSS_PUBLIC_HOST=${EXTERNAL_IP}`). Use the
  **Tailscale address** (`tailscale ip -4`): it survives a venue DHCP change,
  unlike the LAN address. `haproxy` binds `*` on 7777 and both it and
  `vss-agent` are `network_mode: host`, so Tailscale reaches them.

`make vss` syncs both from live interfaces and then recreates. Verify with
`make -C vss-video-analytics urls` before demoing.

Two traps:

- **`docker compose start` re-reads nothing.** It resumes containers with their
  old env, so a config fix alone changes nothing visible. Only `up -d` recreates.
- **`make redeploy` and `make wipe` delete `generated.env`**, plus volumes and
  the entire data-dir. Never reach for them to fix a network problem.

Expect a slower first start after any IP change — all 12 containers are
recreated, so the NIMs reload. The UI reads env at runtime (`next-runtime-env`
via `/__ENV.js`), so no image rebuild is needed. One last override to know about:
`sessionStorage.webSocketURL` beats the env var, but only if someone set an
endpoint by hand in the UI's Settings dialog — closing the tab clears it.

## Demo-specific notes

**fraud** — Run `make fraud-prewarm` after a cold boot. The Triton image ships
torch cu126 whose arch list stops at `compute_90`, so the driver JIT-compiles PTX
forward on sm_121: ~85 s on first kernel launch versus 0.49 s warm. A persistent
CUDA JIT cache mount plus prewarm avoids a visible stall on stage. Retraining
regenerates `model.py` and does **not** update `demo_model_repo/` — re-apply the
`shap_n_samples` patch (16; 8 is unstable and reorders feature ranking) or
explanation latency reverts from 3.3 s to 12.5 s.

**portfolio** — CUDA 13 wheels are Blackwell-native, so there is no JIT penalty
and no prewarm. Lead with the cuOpt solve, not KDE scenario generation (cuML
measured *slower* than sklearn at demo scale). Never demo below ~500 assets ×
50,000 scenarios; at 100 × 5,000 the GPU loses on launch overhead.

## Honest framing at the booth

Two caveats should be volunteered rather than hidden:

- Fraud metrics come from a test set rebalanced to 8.09 % fraud; the true base
  rate is 0.122 %. They are a benchmark, not a production operating point.
- The cuOpt speedup is in the solver, not in scenario generation.

## Repo hygiene

- `shared_data/` must never be published — it holds real identity documents.
  It is untracked and ignored; keep it that way.
- Tracked `.env` files contain ports only. Real credentials live outside this
  repo (VSS keeps its NGC key in `VSS-3x`'s `generated.env`, which is not part of
  this repository). Keep it that way — this repo is public.
