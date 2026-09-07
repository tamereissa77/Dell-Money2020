"""Export a compact, well-sampled 2D UMAP of the transaction embedding space
for the booth UI. Keeps every fraud, samples the legitimate majority.
"""
import numpy as np, pandas as pd, json, time
import cupy as cp
from cuml.manifold import UMAP

E = "data/embeddings"
lab = np.load(f"{E}/labels.npy")
ids = np.concatenate([np.load(f"{E}/{s}_row_ids.npy") for s in ("train","val","test")])
emb = np.load(f"{E}/embeddings.npy", mmap_mode="r")
print(f"pool: {emb.shape}  frauds: {int(lab.sum()):,}  rate: {100*lab.mean():.3f}%")

rng = np.random.default_rng(42)
fraud_idx = np.where(lab == 1)[0]
legit_idx = rng.choice(np.where(lab == 0)[0], size=min(34000, int((lab==0).sum())), replace=False)
sel = np.sort(np.concatenate([fraud_idx, legit_idx]))
print(f"sampled {len(sel):,} points ({len(fraud_idx):,} fraud + {len(legit_idx):,} legit)")

X = np.asarray(emb[sel], dtype=np.float32)
t = time.time()
um = UMAP(n_components=2, n_neighbors=25, min_dist=0.12, random_state=42, verbose=False)
xy = um.fit_transform(cp.asarray(X))
xy = cp.asnumpy(xy) if hasattr(xy, "get") else np.asarray(xy)
print(f"cuML UMAP on GPU: {time.time()-t:.1f}s")

# normalise to a clean 0..1000 box
xy = xy - xy.min(0); xy = xy / xy.max() * 1000.0

raw = pd.read_csv("data/TabFormer/raw/card_transaction.v1.csv",
                  usecols=["Amount","Merchant City","Merchant State","MCC","Use Chip","Errors?",
                           "Year","Month","Day"], low_memory=False)
r = raw.iloc[ids[sel]].reset_index(drop=True)
amt = r["Amount"].astype(str).str.replace("$","",regex=False).astype(float)

pts = []
for i in range(len(sel)):
    pts.append([round(float(xy[i,0]),1), round(float(xy[i,1]),1), int(lab[sel[i]]),
                round(float(amt.iloc[i]),2), int(r["MCC"].iloc[i]),
                str(r["Merchant City"].iloc[i]).strip()[:20],
                str(r["Merchant State"].iloc[i]).strip()[:14],
                str(r["Use Chip"].iloc[i]).replace(" Transaction",""),
                ("" if str(r["Errors?"].iloc[i]).strip() in ("nan","XX") else str(r["Errors?"].iloc[i]).strip()[:18]),
                f'{int(r["Year"].iloc[i])}-{int(r["Month"].iloc[i]):02d}-{int(r["Day"].iloc[i]):02d}'])

out = {"schema": ["x","y","fraud","amount","mcc","city","state","chip","errors","date"],
       "n": len(pts), "n_fraud": int(lab[sel].sum()),
       "pool_size": int(len(lab)), "pool_fraud_rate": round(100*float(lab.mean()),3),
       "points": pts}
import os; os.makedirs("results", exist_ok=True)
json.dump(out, open("results/umap_points.json","w"), separators=(",",":"))
print(f"wrote results/umap_points.json  ({os.path.getsize('results/umap_points.json')/1e6:.1f} MB)")
