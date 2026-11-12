"""R1 red-team: endpoints for the headline claim from saved score matrices (results/review/S_*.npy)."""
import os, sys
os.environ["OMP_NUM_THREADS"] = "2"
import numpy as np, pandas as pd
sys.path.insert(0, "src")
from modeng import tasks as T, metrics as M
from modeng.methods import baselines as BL
R = "results/review/"; C = "data/cache/channels/"
Bf = T.load("temporal"); deg = np.asarray(Bf.degree)
# degree bins: 0 = off-network, then deciles of positive degree
q = np.quantile(deg[deg > 0], np.linspace(0, 1, 11)[1:-1]); dbin = np.where(deg > 0, 1 + np.searchsorted(q, deg, side="left"), 0)

def within_bin_ap(s, pos, exclude):
    mask = np.ones(len(s), bool); mask[exclude] = False
    y = np.zeros(len(s), bool); y[pos] = True
    s = np.where(np.isfinite(s), s, -1e30) + np.random.default_rng(0).uniform(0, 1e-12, len(s))
    num = den = 0.0
    for b in np.unique(dbin[pos]):
        m = mask & (dbin == b); yy = y[m]; n = int(yy.sum())
        if n == 0 or n == m.sum(): continue
        o = np.argsort(-s[m]); r = np.flatnonzero(yy[o]); ap = ((np.arange(n) + 1) / (r + 1)).mean()
        base = n / m.sum()
        num += n * ap; den += n
    return num / den if den else np.nan

def endpoints(S, B):
    rows = []
    for i, t in enumerate(B.tasks):
        s = S[i]; r = {}
        if not np.isfinite(s).any():
            rows.append({}); continue
        m = M.rank_metrics(s, t.targets, t.exclude, deg)
        for k in ("ap", "nuap", "auroc", "recall@50", "recall@100", "recall@200", "ndcg@100", "first_hit"): r[k] = m[k]
        # precision@20
        mask = np.ones(len(s), bool); mask[t.exclude] = False
        ss = np.where(np.isfinite(s), s, -1e30); cand = np.flatnonzero(mask); top = cand[np.argsort(-ss[cand], kind="stable")[:20]]
        r["p@20"] = np.isin(top, t.targets).mean(); r["hit@10"] = float(m["first_hit"] <= 10)
        if "onnet" in t.alt:
            mo = M.rank_metrics(s, *t.alt["onnet"], deg); r["onnet_ap"], r["onnet_nuap"], r["onnet_r100"] = mo["ap"], mo["nuap"], mo["recall@100"]
        if "sa" in t.alt:
            ms = M.rank_metrics(s, *t.alt["sa"], deg); r["sa_ap"], r["sa_nuap"] = ms["ap"], ms["nuap"]
        if "sa" in t.alt:
            kept, sexc = t.alt["sa"]; off = np.flatnonzero(deg == 0); ko = np.setdiff1d(kept, off)
            if len(ko):
                mo2 = M.rank_metrics(s, ko, np.union1d(sexc, np.setdiff1d(off, ko)), deg); r["sa_onnet_ap"], r["sa_onnet_nuap"], r["sa_onnet_r100"] = mo2["ap"], mo2["nuap"], mo2["recall@100"]
            na = np.setdiff1d(t.targets, kept)
            if len(na):
                mn = M.rank_metrics(s, na, np.union1d(t.exclude, kept), deg); r["nonA_ap"], r["nonA_nuap"], r["nonA_r100"] = mn["ap"], mn["nuap"], mn["recall@100"]
                nao = np.setdiff1d(na, off)
                if len(nao):
                    mn = M.rank_metrics(s, nao, np.union1d(np.union1d(t.exclude, kept), np.setdiff1d(off, nao)), deg); r["nonA_onnet_ap"], r["nonA_onnet_nuap"] = mn["ap"], mn["nuap"]
        mt = np.unique(t.mask_c); keep = np.setdiff1d(t.targets, mt)
        if len(keep):
            mm = M.rank_metrics(s, keep, np.union1d(t.exclude, mt), deg); r["norewire_ap"], r["norewire_nuap"] = mm["ap"], mm["nuap"]
        r["degbin_ap"] = within_bin_ap(s, t.targets, t.exclude)
        on = t.alt["onnet"][0] if "onnet" in t.alt else np.array([], int)
        if len(on) >= 4:
            md = np.median(deg[on]); hi = on[deg[on] > md]; lo = on[deg[on] <= md]
            for nm_, g in (("hub", hi), ("nonhub", lo)):
                if len(g):
                    r[f"{nm_}_ap"] = M.rank_metrics(s, g, np.union1d(t.exclude, np.setdiff1d(t.targets, g)), None)["ap"]
                    r[f"{nm_}_r100"] = M.rank_metrics(s, g, np.union1d(t.exclude, np.setdiff1d(t.targets, g)), None)["recall@100"]
        rows.append(r)
    df = pd.DataFrame(rows); df["cluster"] = [t.cluster for t in B.tasks]; df["disease_id"] = [t.disease_id for t in B.tasks]
    df["name"] = [t.name for t in B.tasks]; df["n_seeds"] = [len(t.seeds) for t in B.tasks]; df["n_targets"] = [len(t.targets) for t in B.tasks]
    return df

def boot(a, b, cl, n=10000):
    ok = ~(np.isnan(a) | np.isnan(b)); d = (a - b)[ok]; g = pd.DataFrame(dict(d=d, c=cl[ok])).groupby("c")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy(); idx = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
    bt = s[idx].sum(1) / c[idx].sum(1)
    return a[ok].mean(), b[ok].mean(), d.mean(), np.quantile(bt, .025), np.quantile(bt, .975), (bt <= 0).mean(), (d > 0).mean(), int(ok.sum())

models = [m for m in ("full", "clean", "pt19lag", "p025", "p0") if os.path.exists(f"{R}S_{m}_confirm.npy")]
out = []
for split in ("dev", "confirm"):
    B = Bf.split(split); cl = np.array([t.cluster for t in B.tasks])
    E = {"transfer": endpoints(np.load(f"{C}temporal_{split}_transfer_jaccard.npy"), B), "headline": endpoints(BL.modet_headline(B.tasks, B), B)}
    for m in models: E[m] = endpoints(np.load(f"{R}S_{m}_{split}.npy"), B)
    for k, v in E.items(): v.to_parquet(f"{R}endpoints_{k}_{split}.parquet")
    cols = [c for c in E["transfer"].columns if c not in ("cluster", "disease_id", "name", "n_seeds", "n_targets")]
    for m in models:
        for ref in ("transfer", "headline"):
            for c in cols:
                if c not in E[m] or c not in E[ref]: continue
                a, b, d, lo, hi, p, w, n = boot(E[m][c].to_numpy(float), E[ref][c].to_numpy(float), cl)
                out.append(dict(split=split, model=m, ref=ref, metric=c, model_mean=a, ref_mean=b, delta=d, lo=lo, hi=hi, p_le0=p, win=w, n=n))
    if split == "confirm":
        # concentration of the nuAP / AP gain of `full` over transfer
        for c in ("nuap", "ap"):
            d = (E["full"][c] - E["transfer"][c]).to_numpy(); o = np.argsort(-d)
            print(f"[concentration {c}] mean delta {d.mean():+.4f} median {np.median(d):+.4f} win {np.mean(d>0):.2f}; sum of top5 = {d[o[:5]].sum()/d.sum():.2f} of total, top10 = {d[o[:10]].sum()/d.sum():.2f}; "
                  f"mean without top5 {np.delete(d, o[:5]).mean():+.4f}, without top10 {np.delete(d, o[:10]).mean():+.4f}; worst5 sum {d[o[-5:]].sum():+.3f}")
            print(pd.DataFrame(dict(name=E["full"]["name"].str[:40], n_seeds=E["full"]["n_seeds"], n_t=E["full"]["n_targets"], full=E["full"][c], transfer=E["transfer"][c], d=d)).iloc[np.r_[o[:6], o[-6:]]].round(3).to_string())
        g = pd.DataFrame(dict(c=cl, d=(E["full"]["nuap"] - E["transfer"]["nuap"]).to_numpy())).groupby("c")["d"].agg(["mean", "count"]).sort_values("count", ascending=False)
        print("clusters:", len(g), "largest cluster sizes", g["count"].head(5).tolist(), "; cluster-mean delta >0 in", float((g["mean"] > 0).mean()))
        ns = E["full"]["n_seeds"].to_numpy(); d = (E["full"]["nuap"] - E["transfer"]["nuap"]).to_numpy(); da = (E["full"]["ap"] - E["transfer"]["ap"]).to_numpy()
        for lo_, hi_ in ((0, 20), (20, 100), (100, 10**6)):
            mk = (ns >= lo_) & (ns < hi_); print(f"n_seeds [{lo_},{hi_}): n={mk.sum()} d_nuAP {d[mk].mean():+.4f} d_AP {da[mk].mean():+.4f}")
        fh = {k: E[k]["first_hit"] for k in E}
        print("first_hit median:", {k: float(v.median()) for k, v in fh.items()}, "| share first_hit<=10:", {k: round(float((v <= 10).mean()), 3) for k, v in fh.items()})
res = pd.DataFrame(out); res.to_csv(f"{R}endpoints_summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
for m in models:
    for ref in ("transfer", "headline"):
        x = res[(res.model == m) & (res.ref == ref)]
        print(f"\n=== {m} vs {ref} ===")
        print(x.pivot(index="metric", columns="split", values=["model_mean", "ref_mean", "delta", "lo", "hi"]).swaplevel(axis=1).sort_index(axis=1).round(4).reindex(columns=["confirm", "dev"], level=0).to_string())
