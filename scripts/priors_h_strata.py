"""implementer H: re-run the stacker (same protocol as modeng.stack.run: dev 5-fold grouped OOF, confirm = model trained on dev)
and evaluate per-task metrics on target strata from modet's arrival_devall file (temporal only):
  gw = gwas_only arrivals, ng = clinvar_only / curated_only / mixed_non_gwas arrivals (other targets removed from the ranking).
Writes results/temporal/<split>/hS_<tag>.parquet (per task, columns gw_* / ng_* added). Usage: priors_h_strata.py <tag> <channels> [bench]
Also used for forward (no strata there). Leaderboard rows are appended as hS_<tag>."""
import sys, numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH, data as D
from modeng import stack as ST

ARR = "/home/ph/code/modet/data/processed/groundtruth/arrival_devall_t0_string115_phys_notm_400.parquet"
STRATA = {"gw": ["gwas_only"], "ng": ["clinvar_only", "curated_only", "mixed_non_gwas"]}


def add_strata(B):
    a = pd.read_parquet(ARR)
    for k, cls in STRATA.items():
        sets = a[a.evidence_class.isin(cls)].groupby("disease_id")["gene"].apply(list).to_dict()
        for t in B.tasks:
            pos = np.intersect1d(D.gene_idx(sets.get(t.disease_id, [])), t.targets)
            if len(pos) >= 3:
                t.alt[k] = (pos, np.union1d(t.exclude, np.setdiff1d(t.targets, pos)))


def main(tag, channels, folds=5, neg=3000, rounds=400, bench="temporal"):
    import time
    t0 = time.time()
    ST.LGB["num_threads"] = int(__import__("os").environ.get("H_THREADS", 4))         # shared machine (load > 100): 4 threads; paired comparisons only within hS_ runs
    names = CH.all_channel_names(tuple(channels.split(",")))
    Bf = T.load(bench)
    Bd, Bc = Bf.split("dev"), Bf.split("confirm")
    md, rd, nd = ST.build_feats(Bd, "dev", names)
    clusters = np.array([t.cluster or t.disease_id for t in Bd.tasks])
    uc = np.unique(clusters); rng = np.random.default_rng(0); rng.shuffle(uc)
    fold_of = {c: k % folds for k, c in enumerate(uc)}
    fid = np.array([fold_of[c] for c in clusters])
    S = np.zeros((len(Bd.tasks), Bd.n_genes), np.float32)
    for k in range(folds):
        tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
        X, y, w = ST.training_rows(Bd, tr, md, rd, nd, neg)
        ST.predict_all(ST.fit(X, y, w, rounds), Bd, te, md, rd, nd, S)
    X, y, w = ST.training_rows(Bd, np.arange(len(Bd.tasks)), md, rd, nd, neg)
    m = ST.fit(X, y, w, rounds)
    mc, rc, nc = ST.build_feats(Bc, "confirm", names)
    Sc = np.zeros((len(Bc.tasks), Bc.n_genes), np.float32)
    ST.predict_all(m, Bc, np.arange(len(Bc.tasks)), mc, rc, nc, Sc)
    for split, B_, S_ in (("dev", Bd, S), ("confirm", Bc, Sc)):
        if bench == "temporal":
            add_strata(B_)
        df = BM.evaluate(S_, B_, jobs=4)
        for k in STRATA:
            if f"{k}_ap" not in df:
                df[f"{k}_ap"] = np.nan; df[f"{k}_nuap"] = np.nan
        df.to_parquet(BM.RESULTS / bench / split / f"hS_{tag}.parquet", index=False)
        row = dict(bench=bench, split=split, method=f"hS_{tag}", n=int(df["ap"].notna().sum()), ap=df["ap"].mean(),
                   nuap=df["nuap"].mean(), auroc=df["auroc"].mean(), r100=df["recall@100"].mean(),
                   onnet_ap=df["onnet_ap"].mean(), onnet_nuap=df["onnet_nuap"].mean(),
                   sa_ap=df["sa_ap"].mean() if "sa_ap" in df else np.nan, seconds=round(time.time() - t0, 1),
                   timestamp=time.strftime("%Y-%m-%d %H:%M:%S"), git=BM.git_sha())
        pd.DataFrame([row]).to_csv(BM.LEADERBOARD, sep="\t", mode="a", header=False, index=False, float_format="%.4f")
        print(f"{tag:14s} {split:8s} nuAP {df.nuap.mean():.4f} AP {df.ap.mean():.4f} | gw nuAP {df.gw_nuap.mean():.4f} "
              f"(n={df.gw_nuap.notna().sum()}) | ng nuAP {df.ng_nuap.mean():.4f} (n={df.ng_nuap.notna().sum()})", flush=True)
        for k in STRATA:   # strata are mutated on the shared cached tasks: remove them again
            for t in B_.tasks:
                t.alt.pop(k, None)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], bench=sys.argv[3] if len(sys.argv) > 3 else "temporal")
