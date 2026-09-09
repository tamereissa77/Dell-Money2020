# Fine-tuned LoRA adapter goes here

Efadatek supplies a directory named `best/` containing the PEFT output:

    best/
      adapter_model.safetensors
      adapter_config.json
      (plus any tokenizer/processor files saved alongside)

Place it so the final path is:

    nbe_finetune/output/checkpoints/best/

Then rebuild is NOT needed — the directory is bind-mounted:
    docker compose restart vision_api

## Verify it actually loaded

    curl -s http://localhost:8005/health

Expect `"adapter_loaded": true` and
`"model": "Qwen2.5-VL-7B-Instruct + LoRA"`.

If it says `(BASE — not fine-tuned)` the adapter was not picked up, and
extraction quality on Egyptian documents will be noticeably worse.

## Booth safety

Set `REQUIRE_ADAPTER=1` in `.env` so the service refuses to start on the base
model rather than silently degrading:

    REQUIRE_ADAPTER=1

The base model must match what the adapter was trained on:
`Qwen/Qwen2.5-VL-7B-Instruct` (already cached in `hf-cache/`).
