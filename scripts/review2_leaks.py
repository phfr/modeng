"""R3 red team: input leakage of the v3 channels on temporal (priors from OT evidence 21.04, HPO 2021-01, PubTator ridge, GWAS-catalog lag).
Brute-force recomputation of a few confirm tasks per channel with independent code + vintage / date checks. Log: results/review2/leaks.log"""
import sys, re
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
from modeng import data as D
import scipy.sparse as sp
U = D.universe(); upos = {g: i for i, g in enumerate(U)}
rng = np.random.default_rng(3); PICK = sorted(rng.choice(len(Bc.tasks), 6, replace=False).tolist())
print("brute-force tasks:", [(i, Bc.tasks[i].disease_id, Bc.tasks[i].name, len(Bc.tasks[i].seeds), len(Bc.tasks[i].targets)) for i in PICK])

# ---------------------------------------------------------------- 1. OT evidence 21.04 vs temporal targets
ev = pd.read_parquet(ROOT + "/data/cache/priors_h/ev_2104.parquet"); ev["g"] = U.get_indexer(ev.targetId); ev = ev[ev.g >= 0]
bydis = {d: x for d, x in ev.groupby("diseaseId")}
MEND = ("eva", "gene2phenotype", "genomics_england", "clingen", "uniprot_literature", "uniprot_variants", "orphanet")
rowsum = []
for split, B in (("dev", Bd), ("confirm", Bc)):
    for t in B.tasks:
        fam = {t.disease_id} | set(D.descendants(t.disease_id))
        fe = pd.concat([bydis[d] for d in fam if d in bydis]) if any(d in bydis for d in fam) else ev.iloc[:0]
        anc = set(D.relatives(t.disease_id)) - fam
        ae = pd.concat([bydis[d] for d in anc if d in bydis]) if any(d in bydis for d in anc) else ev.iloc[:0]
        excl = np.zeros(B.n_genes, bool); excl[t.exclude] = True; excl[t.targets] = True
        neg = np.flatnonzero(~excl); unk = np.setdiff1d(t.exclude, t.seeds)
        def frac(e, genes, cond=None):
            e2 = e if cond is None else e[cond(e)]
            return np.isin(genes, e2.g.unique()).mean() if len(genes) else np.nan
        gw = lambda e: (e.src == "ot_genetics_portal") & (e.s >= 0.5)
        me = lambda e: e.src.isin(MEND) & ~((e.src == "eva") & (e.s < 0.5))
        strong = lambda e: e.src.isin(MEND + ("ot_genetics_portal",))
        rowsum.append(dict(split=split, disease=t.disease_id, n_t=len(t.targets),
                           t_fam_any=frac(fe, t.targets), n_fam_any=frac(fe, neg), u_fam_any=frac(fe, unk), s_fam_any=frac(fe, t.seeds),
                           t_fam_gen=frac(fe, t.targets, strong), n_fam_gen=frac(fe, neg, strong), u_fam_gen=frac(fe, unk, strong),
                           t_fam_l2g=frac(fe, t.targets, gw), n_fam_l2g=frac(fe, neg, gw), t_fam_mend=frac(fe, t.targets, me), n_fam_mend=frac(fe, neg, me),
                           t_anc_l2g=frac(ae, t.targets, gw), n_anc_l2g=frac(ae, neg, gw), t_anc_mend=frac(ae, t.targets, me), n_anc_mend=frac(ae, neg, me),
                           n_tpairs_fam_l2g=int(np.isin(t.targets, fe[gw(fe)].g.unique()).sum()), n_tpairs_fam_mend=int(np.isin(t.targets, fe[me(fe)].g.unique()).sum()),
                           n_tpairs_fam_any=int(np.isin(t.targets, fe.g.unique()).sum())))
R = pd.DataFrame(rowsum); R.to_csv(OUT + "ev2104_target_overlap.tsv", sep="\t", index=False, float_format="%.4f")
print("\n== 1. 21.04 evidence for (task family = own + descendants, gene): share of targets / sampled-negative candidates / unknown / seeds (mean over tasks)")
print(R.groupby("split")[[c for c in R.columns if c[:2] in ("t_", "n_", "u_", "s_") and c != "n_t"]].mean().T.round(4).to_string())
print("target pairs total", R.groupby("split").n_t.sum().to_dict(), "| with family L2G>=0.5 evidence at 21.04:", R.groupby("split").n_tpairs_fam_l2g.sum().to_dict(),
      "| with family Mendelian evidence:", R.groupby("split").n_tpairs_fam_mend.sum().to_dict(), "| any source:", R.groupby("split").n_tpairs_fam_any.sum().to_dict())

# brute-force priors from the raw evidence parquet (independent of priors_h and its ev cache)
import duckdb
con = duckdb.connect(); con.execute("SET threads=2; SET memory_limit='4GB'")
EV = "/mnt/msadb/modet/downloads/ot_evidence/evidence_2104"
raw = con.execute(f"SELECT datasourceId src, diseaseId, targetId, max(score) s FROM read_parquet('{EV}/sourceId=ot_genetics_portal/*.parquet', union_by_name=true) GROUP BY ALL").df()
rawm = pd.concat([con.execute(f"SELECT datasourceId src, diseaseId, targetId, max(score) s FROM read_parquet('{EV}/sourceId={s}/*.parquet', union_by_name=true) GROUP BY ALL").df()
                  for s in MEND if __import__('os').path.isdir(f"{EV}/sourceId={s}")])
print("\nraw 21.04 ot_genetics_portal rows", len(raw), "sources", raw.src.unique(), "| mendelian rows", len(rawm), sorted(rawm.src.unique()))
try:
    yr = con.execute(f"SELECT max(publicationYear) y, count(*) n, sum(CASE WHEN publicationYear > 2021 THEN 1 ELSE 0 END) late FROM read_parquet('{EV}/sourceId=ot_genetics_portal/*.parquet', union_by_name=true)").df()
    print("ot_genetics_portal 21.04 publicationYear max / n / n(>2021):", yr.to_dict("records"))
except Exception as e:
    print("publicationYear check failed:", str(e)[:200])
for i in PICK:
    t = Bc.tasks[i]; fam = {t.disease_id} | set(D.descendants(t.disease_id))
    w = raw[(raw.s >= 0.5) & ~raw.diseaseId.isin(fam)]
    n = np.zeros(len(U)); c = w.groupby("targetId").diseaseId.nunique(); j = U.get_indexer(c.index); n[j[j >= 0]] = c.to_numpy()[j >= 0]
    mx = np.zeros(len(U)); c = raw[~raw.diseaseId.isin(fam)].groupby("targetId").s.max(); j = U.get_indexer(c.index); mx[j[j >= 0]] = c.to_numpy()[j >= 0]
    m2 = rawm[~((rawm.src == "eva") & (rawm.s < 0.5)) & ~rawm.diseaseId.isin(fam)]
    mn = np.zeros(len(U)); c = m2.groupby("targetId").diseaseId.nunique(); j = U.get_indexer(c.index); mn[j[j >= 0]] = c.to_numpy()[j >= 0]
    print(f"task {i} {t.disease_id}: max|diff| pri_l2g_n {np.abs(np.log1p(n) - cached('confirm', 'pri_l2g_n')[i]).max():.2e}  pri_l2g_max {np.abs(mx - cached('confirm', 'pri_l2g_max')[i]).max():.2e}"
          f"  pri_mend_n {np.abs(np.log1p(mn) - cached('confirm', 'pri_mend_n')[i]).max():.2e}")

# ---------------------------------------------------------------- 2. HPO vintage and phenotype matching
from modeng.ext import priors_h as PH
def hpo_genes(f, old):
    h = pd.read_csv(f, sep="\t", comment="#", header=None, usecols=[0, 3], names=["entrez", "term"]) if old else \
        pd.read_csv(f, sep="\t", usecols=["ncbi_gene_id", "hpo_id"]).rename(columns={"ncbi_gene_id": "entrez", "hpo_id": "term"})
    return h
h21 = hpo_genes(PH.HPO["temporal"], True); h25 = hpo_genes(PH.HPO["forward"], False)
uh = pd.read_parquet(D.MODET / "data/processed/hgnc_protein_coding.parquet"); e2g = {int(e): g for e, g in zip(uh.entrez_id, uh.ensembl_gene_id) if pd.notna(e)}
for h in (h21, h25):
    h["g"] = U.get_indexer(h.entrez.map(e2g))
g21, g25 = set(h21.g[h21.g >= 0]), set(h25.g[h25.g >= 0])
print(f"\n== 2. HPO: genes with a profile 2021-01-15 file {len(g21)}, 2025-09 file {len(g25)}, only-2025 {len(g25 - g21)}, only-2021 {len(g21 - g25)}; terms 2021 {h21.term.nunique()} / 2025 {h25.term.nunique()}")
tp = np.concatenate([t.targets for t in Bc.tasks]); allneg = np.setdiff1d(np.arange(len(U)), [])
print(f"confirm target pairs with a 2021 HPO profile {np.isin(tp, list(g21)).mean():.3f} (universe {len(g21)/len(U):.3f}); target pairs whose gene gained its first HPO profile after 2021-01 (in 2025 file only): {np.isin(tp, list(g25 - g21)).mean():.3f} (universe {len(g25 - g21)/len(U):.3f})")
# independent dense recompute of ph_mean / ph_top3
h = h21[h21.g >= 0].drop_duplicates(["g", "term"]); terms = {x: k for k, x in enumerate(sorted(h.term.unique()))}
G = np.zeros((len(U), len(terms)), np.float32); G[h.g.to_numpy(), h.term.map(terms).to_numpy()] = 1
has = G.sum(1) > 0; idf = np.log(has.sum() / np.maximum(G.sum(0), 1)); Gw = G * idf; Gw /= np.maximum(np.linalg.norm(Gw, axis=1, keepdims=True), 1e-9)
for i in PICK[:4]:
    t = Bc.tasks[i]; s = t.seeds[has[t.seeds]]
    if len(s) == 0:
        print(f"task {i}: no HPO seeds; cache all-NaN = {np.isnan(cached('confirm', 'ph_mean')[i]).all()}"); continue
    p = Gw[s].sum(0); p /= max(np.linalg.norm(p), 1e-9); pm = Gw @ p; C = Gw @ Gw[s].T; t3 = np.sort(C, 1)[:, -min(3, C.shape[1]):].mean(1)
    cm, c3 = cached("confirm", "ph_mean")[i], cached("confirm", "ph_top3")[i]
    print(f"task {i} {t.disease_id}: max|diff| ph_mean {np.nanmax(np.abs(pm - cm)[has]):.2e} ph_top3 {np.nanmax(np.abs(t3 - c3)[has]):.2e}; NaN pattern equal {bool((np.isnan(cm) == ~has).all())}; seeds with HPO {len(s)}/{len(t.seeds)}")
del G, Gw

# ---------------------------------------------------------------- 3. PubTator ridge
from modeng.ext.donors_a import load_any
M21, M19, M25 = load_any("a_pt21").tocsr(), load_any("a_pt19").tocsr(), load_any("a_pt25").tocsr()
print(f"\n== 3. PubTator sets: a_pt19 {M19.shape} nnz {M19.nnz}; a_pt21 {M21.shape} nnz {M21.nnz}; a_pt25 {M25.shape} nnz {M25.nnz}")
for i in PICK[:4]:
    t = Bc.tasks[i]; Md = M21.astype(np.float64); s = np.zeros(len(U)); s[t.seeds] = 1
    ov = np.asarray(Md @ s).ravel(); cand = np.flatnonzero(ov > 0); cand = cand[np.argsort(-ov[cand])[:500]]
    X = Md[cand].toarray(); X /= np.sqrt(np.maximum((X > 0).sum(1, keepdims=True), 1))
    w = np.linalg.solve(X @ X.T + np.eye(len(cand)), X @ s); u = X.T @ w; p = X.T @ np.clip(w, 0, None)
    cu, cp = cached("confirm", "pt_ridge")[i], cached("confirm", "pt_ridge_pos")[i]
    for nm, MM in (("a_pt19", M19), ("a_pt25", M25)):
        pass
    print(f"task {i} {t.disease_id}: max|diff| pt_ridge {np.abs(u - cu).max():.2e} pt_ridge_pos {np.abs(p - cp).max():.2e} (score range {cu.min():.3f}..{cu.max():.3f}); seeds get score mean {cu[t.seeds].mean():.3f}, targets {cu[t.targets].mean():.3f}, others {np.delete(cu, np.r_[t.seeds, t.targets]).mean():.4f}")

# ---------------------------------------------------------------- 4. GWAS catalog lag
a = pd.read_parquet("/home/ph/code/modet/data/raw/gwas_catalog/associations_subset.parquet")
print(f"\n== 4. GWAS catalog associations {len(a)}; added <= 2021-04-30: {(a['DATE ADDED TO CATALOG'] <= '2021-04-30').sum()}; of these with study publication DATE > 2021-04-30: {((a['DATE ADDED TO CATALOG'] <= '2021-04-30') & (a['DATE'] > '2021-04-30')).sum()}; PUBMEDID > 33750000 among them: {((a['DATE ADDED TO CATALOG'] <= '2021-04-30') & (a['PUBMEDID'].astype(float) > 33_950_000)).sum()}")
a = a[(a["P-VALUE"] <= 1e-6) & (a["DATE ADDED TO CATALOG"] <= "2021-04-30")].dropna(subset=["MAPPED_GENE", "MAPPED_TRAIT_URI"])
sym = D.symbols(); s2i = {}
for g, s in sym.items():
    if isinstance(s, str) and g in upos:
        s2i.setdefault(s, upos[g])
a["lp"] = np.minimum(-np.log10(np.maximum(a["P-VALUE"].to_numpy(), 1e-300)), 50.0)
a["tids"] = a.MAPPED_TRAIT_URI.map(lambda u: {x.strip().rsplit("/", 1)[-1] for x in u.split(",") if x.strip()})
cl = cached("confirm", "gwas_catlag"); hit = np.flatnonzero(~np.isnan(cl).all(1)); print("confirm tasks with catlag coverage:", len(hit), "of", len(Bc.tasks))
nd = {k: len(v) for k, v in D.descendants_map().items()}
for i in list(hit[:: max(1, len(hit) // 5)][:5]):
    t = Bc.tasks[i]; own = {t.disease_id} | set(D.descendants(t.disease_id)); wmap = {x: 1.0 for x in own}
    for an in set(D.relatives(t.disease_id)) - own:
        if nd.get(an, 0) <= 100:
            wmap.setdefault(an, 0.7)
    row = np.zeros(len(U), np.float32)
    for tids, genes, lp in zip(a.tids, a.MAPPED_GENE, a.lp):
        ww = max([wmap[x] for x in tids if x in wmap], default=0)
        if ww:
            for x in re.split(r"\s*[-,;]\s*|\s+x\s+", genes):
                if x in s2i:
                    row[s2i[x]] = max(row[s2i[x]], np.float32(ww * lp))
    print(f"task {i} {t.disease_id} {t.name}: max|diff| gwas_catlag {np.abs(row - cl[i]).max():.2e}; genes>0 {int((row > 0).sum())}; targets with catlag>0 {int((cl[i][t.targets] > 0).sum())}/{len(t.targets)}; "
          f"unknown with catlag>0 {int((cl[i][np.setdiff1d(t.exclude, t.seeds)] > 0).sum())}/{len(np.setdiff1d(t.exclude, t.seeds))}; seeds {int((cl[i][t.seeds] > 0).sum())}/{len(t.seeds)}")
