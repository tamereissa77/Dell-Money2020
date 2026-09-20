# Nemotron Voice Agent

Speak to the GB10 and it speaks back. Speech recognition, a 30B language model and
text-to-speech all run as local sidecars on the one machine; the browser talks to it over
WebRTC. Nothing goes to the cloud at inference time.

Based on NVIDIA's [Nemotron Voice Agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent)
(BSD-2-Clause). Unlike the other blueprints in this repo, this one needed **no adaptation for
GB10** — NVIDIA ships a first-class DGX Spark recipe, and we run theirs.

---

## Read this before you promise anything in Riyadh

**It cannot hold a conversation in Arabic.** This is not a configuration gap, it is a model
limitation, and it is worth knowing before someone asks at the stand.

The text-to-speech can *speak* Arabic — Magpie Multilingual covers `ar-AR`, Chatterbox covers
`ar-SA` — and the multilingual ASR can *hear* it. But the session language offered in the UI is
the **intersection of ASR, TTS and LLM**, and no Nemotron LLM supports Arabic:

| LLM | Languages |
|---|---|
| Nemotron 3 Nano | English, German, Spanish, French, Italian, Japanese |
| Nemotron 3.5 Lightning | English, German, Spanish, French, Italian, Japanese |
| Nemotron 3 Super | the same, plus Chinese |

So Arabic will not even appear as a selectable locale. Demo it in English, and if asked, say
plainly that Arabic needs a different LLM behind the same pipeline — which the blueprint
supports, since any OpenAI-compatible endpoint can be substituted, but which we have not
tested.

---

## What the audience does

Walk up, put on the headset, and talk. They interrupt it mid-sentence and it stops and
listens — barge-in is handled by Silero VAD and Pipecat's Smart Turn detection running in
process, not by a round trip.

The demo-able point is not that it talks. It is **where** it talks: every component is on the
machine in front of them. For a financial-services audience the follow-up question is always
the same — *can this run inside our perimeter?* — and this is the answer with no asterisk.

---

## Running it

```bash
make up        # first run pulls ~60-80 GB and takes 30-60 min
make prewarm   # wait for ASR, TTS and LLM to finish loading weights
make verify    # nine checks
make open      # print the URL
make down
```

Then open **`https://<machine-ip>:7860`**, accept the self-signed certificate, choose the
microphone, and connect.

Other recipes:

```bash
make profiles                                    # list them
PROFILE=omni-assistant/dgx-spark make up         # Omni replaces ASR + LLM with one model
PROFILE=multilingual-assistant/dgx-spark make up # fixed session language per connection
```

### Booth notes, learned the hard way elsewhere

- **Use a wired headset.** A built-in laptop microphone picks up the whole stand and turn
  detection never fires cleanly. This is the single biggest cause of a bad voice demo.
- **HTTPS is mandatory**, because browsers only grant microphone access in a secure context.
  The certificate is self-signed, so the first visitor sees a warning — accept it once,
  before anyone is watching.
- **Warm it before the doors open.** The sidecars load tens of GB; the first person to speak
  otherwise waits through it.
- **Do the cert-accept and mic-permission dance on the actual demo laptop in advance.** Both
  are per-browser, per-origin.

---

## What runs where

| Component | Image | Port |
|---|---|---|
| App + Pipecat pipeline | `nemotron-voice-agent:latest` (built locally) | 7860 (HTTPS) |
| ASR — Nemotron Streaming | `nvcr.io/nim/nvidia/nemotron-asr-streaming:1.3.0` | 9001 / 50152 |
| TTS — Magpie Multilingual | `nvcr.io/nim/nvidia/magpie-tts-multilingual:1.9.0` | 9000 / 50151 |
| LLM — Nemotron 3 Nano NVFP4 | `nvcr.io/nvidia/vllm:26.05.post1-py3` | 8000 |

All three sidecar images publish `linux/arm64` manifests — verified, not assumed.

Two details that make the Spark recipe specifically ours rather than a workstation profile
bent into shape:

- The ASR NIM selects a Spark build via `NIM_TAGS_SELECTOR: mode=str,gpu=dgx_spark,batch_size=32`
- vLLM serves NVFP4 weights with the FlashInfer CUTLASS MoE kernel for `sm_121`

Rough VRAM, from NVIDIA's own docs: ASR ~15 GB, TTS ~14 GB, LLM NVFP4 ~15 GB. Comfortable
inside the GB10's 128 GB unified memory — which is exactly why the whole pipeline fits on one
machine here and needs 80 GB of discrete VRAM elsewhere.

### Port overlaps with other demos

This stack uses **6006** (tracing) and **8000** (LLM), which the `quant` demo also uses. Only
one demo runs at a time on this machine, so it does not collide in practice — but do not try
to run both.

---

## Honest notes

- **Credentials are needed to deploy, not to infer.** `NVIDIA_API_KEY` authenticates the
  `nvcr.io` pulls and NIM runtime licensing, and `HF_TOKEN` fetches the vLLM weights. Once
  images and weights are on disk, conversation happens entirely locally. It is
  network-isolated at inference time, not air-gapped at install time. Say it that way.
- **First-turn latency is higher than steady state** while the sidecars warm. `make prewarm`
  exists for this.
- **This is a cascaded pipeline**: speech in, text through an LLM, speech out. It is not a
  speech-native model, so it has the failure modes of the chain — a transcription error
  becomes a confidently wrong answer.
- Upstream's own docs disagree on the workstation VRAM floor (72 GB vs 80 GB). Irrelevant on
  a 128 GB Spark, noted in case someone asks what it needs on other hardware.

## Attribution

Upstream: [NVIDIA-AI-Blueprints/nemotron-voice-agent](https://github.com/NVIDIA-AI-Blueprints/nemotron-voice-agent),
BSD-2-Clause — see `LICENSE` and `README-upstream.md`. Built on
[Pipecat](https://github.com/pipecat-ai/pipecat).
