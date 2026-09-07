"""The demo artefact: which frauds did raw features MISS that the foundation
model embeddings CAUGHT? Produces concrete transactions you can put on screen.
"""
import numpy as np, pandas as pd, json, xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import average_precision_score

E="data/embeddings"
tr_e=np.load(f"{E}/train_embeddings.npy"); tr_y=np.load(f"{E}/train_labels.npy"); tr_id=np.load(f"{E}/train_row_ids.npy")
te_e=np.load(f"{E}/test_embeddings.npy");  te_y=np.load(f"{E}/test_labels.npy");  te_id=np.load(f"{E}/test_row_ids.npy")
print(f"train {tr_e.shape}  test {te_e.shape}  test frauds {int(te_y.sum())}")

# raw features straight from TabFormer, aligned by row id
cols=["User","Card","Amount","Use Chip","Merchant Name","Merchant City","Merchant State",
      "Zip","MCC","Errors?","Year","Month","Day"]
raw=pd.read_csv("data/TabFormer/raw/card_transaction.v1.csv",
                usecols=cols+["Is Fraud?","Time"], low_memory=False)
def prep(idx):
    d=raw.iloc[idx].copy()
    d["Amount"]=d["Amount"].astype(str).str.replace("$","",regex=False).astype(float)
    for c in ["Use Chip","Merchant City","Merchant State","Errors?"]:
        d[c]=d[c].astype("category").cat.codes
    d["Merchant Name"]=pd.factorize(d["Merchant Name"])[0]
    return d[cols].astype("float32").values
Xtr_raw, Xte_raw = prep(tr_id), prep(te_id)

pca=PCA(n_components=64, random_state=0).fit(tr_e)
Xtr_emb, Xte_emb = pca.transform(tr_e), pca.transform(te_e)

def fit(Xtr,Xte,name):
    m=xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05,
        tree_method="hist", device="cuda", eval_metric="aucpr", n_jobs=-1)
    m.fit(Xtr,tr_y)
    p=m.predict_proba(Xte)[:,1]
    print(f"  {name:<22} AP={average_precision_score(te_y,p):.4f}")
    return p
print("\ntraining:")
p_raw = fit(Xtr_raw, Xte_raw, "raw features (13d)")
p_comb= fit(np.hstack([Xtr_raw,Xtr_emb]), np.hstack([Xte_raw,Xte_emb]), "combined (13+64d)")

# threshold each model at the same alert budget (top 0.5% of transactions)
k=int(len(te_y)*0.005)
thr_raw, thr_comb = np.sort(p_raw)[-k], np.sort(p_comb)[-k]
raw_hit, comb_hit = p_raw>=thr_raw, p_comb>=thr_comb
caught = np.where((te_y==1) & (~raw_hit) & comb_hit)[0]
lost   = np.where((te_y==1) & raw_hit & (~comb_hit))[0]
print(f"\nAt an equal alert budget (top {k} of {len(te_y)} transactions):")
print(f"  frauds caught by raw only      : {int(((te_y==1)&raw_hit).sum())}")
print(f"  frauds caught by combined      : {int(((te_y==1)&comb_hit).sum())}")
print(f"  MISSED by raw, CAUGHT by embeds: {len(caught)}")
print(f"  caught by raw, lost by combined: {len(lost)}")

if len(caught):
    print("\n=== transactions the baseline missed, embeddings caught ===")
    rows=[]
    for i in caught[np.argsort(-p_comb[caught])][:8]:
        r=raw.iloc[te_id[i]]
        rows.append({"amount":r["Amount"],"merchant_city":r["Merchant City"],
                     "state":str(r["Merchant State"]).strip(),"mcc":int(r["MCC"]),
                     "chip":str(r["Use Chip"]).replace(" Transaction",""),
                     "when":f'{int(r["Year"])}-{int(r["Month"]):02d}-{int(r["Day"]):02d}',
                     "raw_score":round(float(p_raw[i]),4),"combined_score":round(float(p_comb[i]),4)})
        print(f"  {r['Amount']:>9}  {str(r['Merchant City']).strip()[:18]:<18} "
              f"MCC {int(r['MCC']):<5} | raw {p_raw[i]:.4f} -> combined {p_comb[i]:.4f}")
    json.dump(rows, open("results/caught_cases.json","w"), indent=2, default=str)
    print("\nsaved results/caught_cases.json")
