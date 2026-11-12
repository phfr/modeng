"""r3 strict-a analysis: gated stacker variants vs modeng-v1 (stack_v3ptr_p0.35) on dev OOF and temporal confirm.

Selection rule (written before any confirm number of these variants was looked at):
  head blend final = (1 - a) * log p_main + a * log p_head, a in {0.25, 0.5, 0.75}; per family (main = gate_ctx, main = gate_noctx)
  a* = argmax on DEV of mean(sa_ap, sa_nuap, sa_onnet_ap, sa_onnet_nuap) among a with dev ap AND nuap >= reference - 0.003;
  if none qualifies or none beats a = 0 on that criterion, a* = 0 (no blend). Grid points other than a* are never scored on confirm.
  Each variant's confirm is computed once (this script); the reported variant count includes every grid point looked at on dev.
Second batch (added 07:55, after gate_ctx dev+confirm were seen, before the new runs finished): gate_ctxperm (control) and
  gate_ctx_aw3 (class-A positives x3 weight); aw3 is preferred over gate_ctx only if its dev SA4 mean is higher and dev ap, nuap >= ref - 0.003.
Endpoints = scripts/review_analyze.py `endpoints` (same as scripts/leader_endpoints.py); paired cluster bootstrap (5000) vs reference.
Usage: python scripts/r3_strict_a.py   (from anywhere; absolute paths)
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "8")
ROOT = "/home/ph/code/modeng"
sys.path.insert(0, ROOT + "/src")
import numpy as np, pandas as pd
src = open(ROOT + "/scripts/review_analyze.py").read(); src = src[:src.index("def boot")]
exec(src)
from modeng import bench as BM

REF = "stack_v3ptr_p0.35"
MAINS = ["stack_gate_ctx", "stack_gate_noctx", "stack_gate_ctxperm", "stack_gate_ctxta", "stack_gate_ctx_aw3"]
HEADS = {"stack_gate_ctx": "stack_gate_ctx_headA", "stack_gate_noctx": "stack_gate_noctx_headA"}
GRID = (0.25, 0.5, 0.75)
MET = ["ap", "nuap", "onnet_ap", "onnet_nuap", "sa_ap", "sa_nuap", "sa_onnet_ap", "sa_onnet_nuap", "nonA_ap", "nonA_nuap", "degbin_ap"]
SA4 = ["sa_ap", "sa_nuap", "sa_onnet_ap", "sa_onnet_nuap"]
OUT = ROOT + "/results/r3_strict_a"


def load(split, m):
    f = BM.RESULTS / "temporal" / split / f"{m}.scores.npy"
    return np.load(f).astype(np.float32) if f.exists() else None


def blend(m, h, a):
    return (1 - a) * np.log(np.maximum(m, 1e-7)) + a * np.log(np.maximum(h, 1e-7))


def boot(a, b, cl, n=5000):
    ok = np.isfinite(a) & np.isfinite(b); d = (a - b)[ok]
    g = pd.DataFrame(dict(d=d, c=cl[ok])).groupby("c")["d"].agg(["sum", "count"])
    s, c = g["sum"].to_numpy(), g["count"].to_numpy(); idx = np.random.default_rng(0).integers(0, len(g), (n, len(g)))
    bt = s[idx].sum(1) / c[idx].sum(1)
    return a[ok].mean(), b[ok].mean(), d.mean(), np.quantile(bt, .025), np.quantile(bt, .975), int(ok.sum())


def summarise(E, ref, split, cl, extra_ref=None):
    rows = []
    for name, e in E.items():
        for c in MET:
            if c not in e: continue
            for rn, r in (("ref", ref),) + ((("headline", extra_ref),) if extra_ref is not None else ()):
                if c not in r: continue
                mv, rv, d, lo, hi, n = boot(e[c].to_numpy(float), r[c].to_numpy(float), cl)
                rows.append(dict(split=split, variant=name, vs=rn, metric=c, value=mv, ref_value=rv, delta=d, lo=lo, hi=hi, n=n))
    return pd.DataFrame(rows)


def main():
    os.makedirs(OUT, exist_ok=True)
    res, chosen, devE = [], {}, {}
    # ---------------- dev
    B = Bf.split("dev"); cl = np.array([t.cluster for t in B.tasks])
    ref = endpoints(load("dev", REF), B); hl = endpoints(BL.modet_headline(B.tasks, B), B)
    E = {m: endpoints(S, B) for m in MAINS if (S := load("dev", m)) is not None}
    for m, h in HEADS.items():
        Sm, Sh = load("dev", m), load("dev", h)
        if Sm is None or Sh is None: continue
        E[h] = endpoints(Sh, B)
        base = np.nanmean([E[m][c].mean() for c in SA4]); best, besta = base, 0.0
        for a in GRID:
            e = endpoints(blend(Sm, Sh, a), B); E[f"{m}+headA{a}"] = e
            crit = np.nanmean([e[c].mean() for c in SA4])
            okc = e["ap"].mean() >= ref["ap"].mean() - 0.003 and e["nuap"].mean() >= ref["nuap"].mean() - 0.003
            print(f"[dev] {m} a={a}: SA4 {crit:.4f} (a=0: {base:.4f}) ap {e['ap'].mean():.4f} nuap {e['nuap'].mean():.4f} ok={okc}", flush=True)
            if okc and crit > best: best, besta = crit, a
        chosen[m] = besta
    print("chosen a*:", chosen, flush=True)
    res.append(summarise(E, ref, "dev", cl, hl))
    n_dev_variants = len(E)
    # ---------------- confirm (once per variant; blends only at a*)
    B = Bf.split("confirm"); cl = np.array([t.cluster for t in B.tasks])
    ref = endpoints(load("confirm", REF), B); hl = endpoints(BL.modet_headline(B.tasks, B), B)
    E = {m: endpoints(S, B) for m in MAINS if (S := load("confirm", m)) is not None}
    for m, h in HEADS.items():
        Sm, Sh = load("confirm", m), load("confirm", h)
        if Sm is None or Sh is None: continue
        E[h] = endpoints(Sh, B)
        if chosen.get(m, 0) > 0:
            a = chosen[m]; S = blend(Sm, Sh, a); E[f"{m}+headA{a}"] = endpoints(S, B)
            np.save(BM.RESULTS / "temporal" / "confirm" / f"{m}_headA{a}.scores.npy", S.astype(np.float16))
    res.append(summarise(E, ref, "confirm", cl, hl))
    R = pd.concat(res); R.to_csv(OUT + "/endpoints_vs_ref.tsv", sep="\t", index=False, float_format="%.4f")
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    for split in ("dev", "confirm"):
        x = R[(R.split == split) & (R.vs == "ref")]
        print(f"\n=== {split}: delta vs {REF} [95% CI] ===")
        x = x.assign(cell=x.apply(lambda r: f"{r.value:.3f} {r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1))
        print(x.pivot(index="variant", columns="metric", values="cell")[[c for c in MET if c in x.metric.unique()]].to_string())
        y = R[(R.split == split) & (R.vs == "headline") & R.metric.isin(["sa_onnet_ap", "sa_onnet_nuap", "onnet_ap", "onnet_nuap"])]
        print(f"--- {split}: vs modet headline (onnet) ---")
        print(y.assign(cell=y.apply(lambda r: f"{r.value:.3f} vs {r.ref_value:.3f} {r.delta:+.3f} [{r.lo:+.3f},{r.hi:+.3f}]", axis=1)).pivot(index="variant", columns="metric", values="cell").to_string())
    print(f"\nvariants looked at on dev: {n_dev_variants}; scored on confirm: {len(E)}; chosen a*: {chosen}")


if __name__ == "__main__":
    main()
