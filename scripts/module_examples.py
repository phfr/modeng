"""Human-readable descriptions of 3 example modeng-v1 modules (tau 0.2) with their leiden_anno submodules.
Example choice (rule fixed before reading the descriptions): decomposed into >= 2 submodules, module <= 150 genes, >= 3 hits among added genes,
one disease per cluster, the 3 with the most added genes on the graph (ties: more hits).
Channel flags per added gene: disease-specific modeng-v1 channels (cached confirm matrices) in which the gene ranks in the top 1 % of the
disease's candidates. Descriptive only (where the gene stands out), not a model attribution (no SHAP of the stacker).
Writes results/modules/r4_examples.md. Usage: python scripts/module_examples.py
"""
import json, re, sys, numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, "/home/ph/code/modeng/src")
from modeng import tasks as T, data as D

ROOT = Path("/home/ph/code/modeng/data/modules"); RES = Path("/home/ph/code/modeng/results/modules")
CH = Path("/home/ph/code/modeng/data/cache/channels")
B = T.load("temporal").split("confirm"); U = D.universe(); SYM = D.symbols().reindex(U).fillna("")
V1 = re.compile(r"^(transfer_jaccard|transfer_onto|rwr_string|don_.*|vmean_.*|vmax_.*|cs_donor|dA_pt|dc_ridge|dc_ridge_pos|gwas_catlag|pri_.*|hz_.*|ph_.*|pt_ridge|pt_ridge_pos)$")
chans = sorted(p.stem.replace("temporal_confirm_", "") for p in CH.glob("temporal_confirm_*.npy"))
chans = [c for c in chans if V1.match(c) and "perm" not in c]
SM = pd.read_csv(RES / "r4_submodules.tsv", sep="\t"); SM = SM[(SM.method == "modeng") & (SM.tau == 0.2)]
idx = {t.disease_id: i for i, t in enumerate(B.tasks)}
cand = []
for did, g in SM.groupby("disease_id"):
    t = B.tasks[idx[did]]
    if g.k.iloc[0] >= 2 and g["size"].sum() <= 150 and g.hits.sum() >= 3:
        cand.append(dict(did=did, cluster=t.cluster, added=g.n_added.sum(), hits=g.hits.sum()))
C = pd.DataFrame(cand).sort_values(["added", "hits"], ascending=False).drop_duplicates("cluster").head(3)
mats = {c: np.load(CH / f"temporal_confirm_{c}.npy", mmap_mode="r") for c in chans}
out = [f"# r4 example modules (modeng-v1, tau 0.2, leiden_anno submodules)\n",
       "Submodule names are the top GO / Reactome term of the submodule; leiden_anno clusters on these same T0 annotations, so names are "
       "circular (descriptive labels, not evidence). `*` = gene became an Open Targets target of the disease by 25.09 (confirm label; not used to "
       "build anything). Channel flags: modeng-v1 channels in which the gene ranks in the disease's top 1 % (descriptive, not attribution).\n"]
for _, r in C.iterrows():
    i = idx[r.did]; t = B.tasks[i]; tg = set(U[t.targets])
    ok = np.ones(B.n_genes, bool); ok[t.exclude] = False; cidx = np.flatnonzero(ok)
    mod = pd.read_csv(ROOT / "modeng" / r.did / "module_tau0.2.tsv", sep="\t").set_index("gene_id")
    sub = pd.read_csv(ROOT / "modeng" / r.did / "sub_tau0.2.tsv", sep="\t")
    meta = json.loads(str(np.load(ROOT / "modeng" / r.did / "sub_tau0.2.npz")["__meta__"]))
    names = {s["id"]: s["name"] for s in meta["submodules"]}
    pct = {}
    for c, M in mats.items():
        v = np.asarray(M[i], np.float32); vc = v[cidx]; srt = np.sort(vc)
        pct[c] = lambda gi, v=v, srt=srt: np.searchsorted(srt, v[gi], side="left") / len(srt)
    def flags(gid):
        gi = U.get_loc(gid); ps = sorted(((pct[c](gi), c) for c in chans), reverse=True)
        return ", ".join(f"{c} ({p:.3f})" for p, c in ps[:4] if p >= 0.99) or "none in top 1 %"
    add = mod[mod.role == "added"]
    out.append(f"\n## {t.name} ({r.did})\n{(mod.role == 'seed').sum()} seeds (OT 21.04), {len(add)} added genes at calibrated P >= 0.2, "
               f"{int(add.index.isin(tg).sum())} of them became targets by 25.09 (of {len(tg)} new targets in total); "
               f"{int((~add.on_graph).sum())} added genes are not on STRING-T0 and are not in any submodule.\n")
    for sid, g in sub[sub.submodule >= 0].groupby("submodule"):
        roles = mod.role.reindex(g.gene_id)
        seeds = [SYM[x] for x in g.gene_id[roles.to_numpy() == "seed"]]
        adds = g.gene_id[roles.to_numpy() == "added"]
        out.append(f"- **Submodule {sid}: {names.get(sid, '')}** ({len(g)} genes, {len(seeds)} seeds: {', '.join(seeds[:12])}{' ...' if len(seeds) > 12 else ''})")
        for gid in sorted(adds, key=lambda x: -mod.calib_p[x]):
            out.append(f"  - new: {SYM[gid] or gid}{'*' if gid in tg else ''} (P {mod.calib_p[gid]:.2f}; {flags(gid)})")
    off = add[~add.on_graph]
    if len(off):
        out.append("- **Not on the graph (no submodule):** " + "; ".join(f"{SYM[g] or g}{'*' if g in tg else ''} (P {mod.calib_p[g]:.2f}; {flags(g)})" for g in off.index))
(RES / "r4_examples.md").write_text("\n".join(out) + "\n"); print("\n".join(out))
