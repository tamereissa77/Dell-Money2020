#!/usr/bin/env bash
# Warm the CUDA PTX JIT cache. The Triton image ships CUDA 12.6 PyTorch whose
# arch list stops at compute_90; on GB10 (sm_121) every kernel is JIT-compiled
# from PTX at first launch. Cold: ~85s. Warm: ~0.5s. Run after every cold boot.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
CACHE="$PWD/cache/nvcache"; mkdir -p "$CACHE"; chmod 777 "$CACHE"
IMG=ffd-triton:latest
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "building $IMG first..."; docker compose build triton; }
echo "[prewarm] cache before: $(du -sh "$CACHE" 2>/dev/null | cut -f1)"
docker run --rm --gpus all -v "$CACHE:/nvcache" \
  -e CUDA_CACHE_PATH=/nvcache -e CUDA_CACHE_MAXSIZE=4294967296 \
  --entrypoint /bin/bash "$IMG" -c '
python - <<PY
import torch, time
from torch_geometric.nn import SAGEConv, TransformerConv
from torch_geometric.utils import scatter
from captum.attr import IntegratedGradients
dev="cuda"; N,E,F=20000,80000,64
x=torch.randn(N,F,device=dev); ei=torch.randint(0,N,(2,E),device=dev)
t=time.time()
SAGEConv(F,64).to(dev)(x,ei).sum().backward()
TransformerConv(F,32,heads=2).to(dev)(x,ei).sum().backward()
scatter(torch.randn(E,F,device=dev), ei[0], dim=0, dim_size=N, reduce="mean")
lin=torch.nn.Sequential(torch.nn.Linear(F,32),torch.nn.ReLU(),torch.nn.Linear(32,2)).to(dev)
IntegratedGradients(lin).attribute(x[:256], target=1, n_steps=16)
torch.cuda.synchronize()
print(f"[prewarm] kernels compiled in {time.time()-t:.1f}s")
PY' 2>&1 | grep -viE "^\[W|UserWarning|warnings.warn|not compatible|current PyTorch install|pytorch.org"
echo "[prewarm] cache after : $(du -sh "$CACHE" 2>/dev/null | cut -f1)"
echo "[prewarm] done — first inference will now be fast."
