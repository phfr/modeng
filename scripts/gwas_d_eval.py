"""Implementer D (GWAS sub-threshold channels): compute channels, evaluate alone on covered tasks and on the GWAS-only stratum.
Usage: PYTHONPATH=src python scripts/gwas_d_eval.py [alone|stack]"""
import sys, time, copy
import numpy as np, pandas as pd
from modeng import tasks as T, bench as BM, channels as CH, metrics as M, data as D
from modeng.ext import gwas_d as G

CHS = ["gwas_own", "gwas_sem", "gwas_xpow", "gwas_seedw", "gwas_catlag", "gwas_sem_perm", "gwas_seedw_perm"]
OUT = BM.RESULTS / "gwas_d"; OUT.mkdir(parents=True, exist_ok=True)


def gwas_only_tasks(B):
    """Temporal: targets restricted to evidence_class == gwas_only (other targets excluded from the ranking)."""
    a = pd.read_parquet(D.GT / "arrival_devall_t0_string115_phys_notm_400.parquet", columns=["disease_id", "gene", "evidence_class"])
    a = a[a.evidence_class == "gwas_only"]
    go = a.groupby("disease_id").gene.apply(list).to_dict()
    out = []
    for t in B.tasks:
        g = np.intersect1d(D.gene_idx(go.get(t.disease_id, [])), t.targets)
        if len(g) < 1:
            out.append(None); continue
        t2 = copy.copy(t); t2.targets = g; t2.alt = {}
        t2.exclude = np.union1d(t.exclude, np.setdiff1d(t.targets, g))
        out.append(t2)
    return out


def eval_rows(S, B, idx, tasks=None):
    rows = []
    for i in idx:
        t = (tasks or B.tasks)[i]
        if t is None:
            continue
        r = M.task_metrics(S[i], t, B.degree); r["disease_id"] = t.disease_id; rows.append(r)
    return pd.DataFrame(rows)


def alone():
    res = []
    for bench, splits in (("temporal", ("dev", "confirm")), ("forward", ("dev", "confirm"))):
        Bf = T.load(bench)
        for sp in splits:
            B = Bf.split(sp)
            tj = CH.compute(B, "transfer_jaccard", sp)
            go = gwas_only_tasks(B) if bench == "temporal" else None
            for ch in CHS:
                t0 = time.time()
                S = CH.compute(B, ch, sp)
                cov = np.flatnonzero(~np.all(np.isnan(S), 1))
                for name, X in ((ch, S), ("transfer_jaccard", tj)):
                    df = eval_rows(np.nan_to_num(X, nan=-1e9) if name == ch else X, B, cov)
                    row = dict(bench=bench, split=sp, channel=ch, scorer=name, n_cov=len(cov), n=len(B.tasks),
                               **df[["ap", "nuap", "auroc", "recall@100", "onnet_ap", "onnet_nuap"]].mean().round(4).to_dict())
                    if go is not None:
                        dg = eval_rows(np.nan_to_num(X, nan=-1e9) if name == ch else X, B, cov, go)
                        if len(dg):
                            row.update(go_n=len(dg), go_ap=round(dg.ap.mean(), 4), go_auroc=round(dg.auroc.mean(), 4),
                                       go_r100=round(dg["recall@100"].mean(), 4))
                    res.append(row); print(row, f"{time.time()-t0:.0f}s", flush=True)
    pd.DataFrame(res).to_csv(OUT / "alone.tsv", sep="\t", index=False)


def compare(pairs):
    """Paired bootstrap (cluster) on all tasks, on covered tasks (any gwas_* channel non-NaN) and the GWAS-only stratum."""
    from modeng import compare as C
    rows = []
    for bench, split, a, b in pairs:
        Bf = T.load(bench); B = Bf.split(split)
        cov_m = np.zeros(len(B.tasks), bool)
        for ch in ("gwas_own", "gwas_sem", "gwas_catlag"):
            cov_m |= ~np.all(np.isnan(CH.compute(B, ch, split)), 1)
        cov_ids = {t.disease_id for t, c in zip(B.tasks, cov_m) if c}
        da, db = C.load(bench, split, a), C.load(bench, split, b)
        for subset, f in (("all", lambda d: d), ("covered", lambda d: d[d.disease_id.isin(cov_ids)]),
                          ("uncovered", lambda d: d[~d.disease_id.isin(cov_ids)])):
            for met in ("ap", "nuap", "auroc", "recall@100"):
                r = C.paired_bootstrap(f(da), f(db), met, n=5000)
                rows.append(dict(bench=bench, split=split, a=a, b=b, subset=subset, **r))
        if bench == "temporal":
            go = gwas_only_tasks(B)
            Sa = np.load(OUT / "scores" / f"{bench}_{split}_{a}.npy"); Sb = np.load(OUT / "scores" / f"{bench}_{split}_{b}.npy")
            ga = eval_rows(Sa, B, range(len(B.tasks)), go); gb = eval_rows(Sb, B, range(len(B.tasks)), go)
            cl = {t.disease_id: t.cluster for t in B.tasks}
            ga["cluster"] = ga.disease_id.map(cl); gb["cluster"] = gb.disease_id.map(cl)
            for met in ("ap", "auroc", "recall@100"):
                r = C.paired_bootstrap(ga, gb, met, n=5000)
                rows.append(dict(bench=bench, split=split, a=a, b=b, subset="gwas_only", **r))
        print(pd.DataFrame(rows[-14:]).round(4).to_string(), flush=True)
    df = pd.DataFrame(rows)
    f = OUT / "compare.tsv"
    if f.exists():
        df = pd.concat([pd.read_csv(f, sep="\t"), df]).drop_duplicates(["bench", "split", "a", "b", "subset", "metric"], keep="last")
    df.to_csv(f, sep="\t", index=False)


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "compare":
        compare([tuple(x.split(":")) for x in sys.argv[2:]])
    else:
        alone()
