"""Implementer A (r1 donors): NEW virtual-donor families. Written to data/donors/a_<family>.npz with a SEPARATE manifest
data/donors/MANIFEST_A.tsv (so the default `donor` channel group / stack_bdv references do not change).

Families
  a_ps_<cell>_resp   Perturb-seq: per perturbation, the perturbed gene + its top responders (|z| >= 4, <= 200, perturbations
                     with >= 10 responders). cells: k562 (Replogle 2022 GWPS), rpe1 (Replogle 2022), cd4 (GWCD4i 2025).
  a_ps_<cell>_reg    Perturb-seq: per readout gene, the readout + its top regulators (perturbations with |z| >= 4 on it, <= 200).
  a_dm_codep         DepMap 20Q4 Chronos: per *selective* gene (SD of effect in top 25 %, not common-essential), the gene + its
                     top-25 co-dependent genes (Pearson r >= 0.2).                                     [pre21, temporal-clean]
  a_impc_orth        IMPC 13.0 MP assertions -> human via MGI HOM_MouseHumanSequence orthology classes (all orthologs, not
                     symbol upper-casing), MP terms propagated to ancestors (MP ontology is_a), sets 5..1000. [pre21 content]
  a_pt19 / a_pt21 / a_pt25   PubTator3 gene x MeSH-disease co-mention sets, PMID cut (entry date checked in PubMed):
                     <= 30,900,000 (2019-03, for `past`), <= 33,750,000 (2021-03, temporal), <= 40,850,000 (2025-08, forward).
                     Gene in set if co-mentioned in >= 2 PMIDs; top 300 genes by PMID count; sets of >= 5 genes.
Usage: python scripts/build_donors_a.py ps depmap impc pubtator
"""
import sys, gzip, numpy as np, pandas as pd, scipy.sparse as sp, h5py
from pathlib import Path

OUT = Path("/home/ph/code/modeng/data/donors")
DER = Path("/mnt/msadb/subex/derived")
U = pd.read_parquet("/home/ph/code/modet/data/processed/hgnc_protein_coding.parquet").sort_values("ensembl_gene_id").reset_index(drop=True)
N = len(U)
i_ensg = {g: i for i, g in enumerate(U.ensembl_gene_id)}
i_entrez = {int(e): i for i, e in enumerate(U.entrez_id) if pd.notna(e)}
i_sym = {s: i for i, s in enumerate(U.symbol)}
PMID_CUT = {"pt19": 30_900_000, "pt21": 33_750_000, "pt25": 40_850_000}


def save(family, sets: dict, cls, src, lo=5, hi=500):
    names, rows, cols = [], [], []
    for name, genes in sets.items():
        g = sorted({int(x) for x in genes if x is not None and x >= 0})
        if lo <= len(g) <= hi:
            rows += [len(names)] * len(g); cols += g; names.append(str(name))
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(names), N))
    np.savez_compressed(OUT / f"{family}.npz", data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape,
                        names=np.array(names), order="sorted_ensg")
    sz = M.getnnz(1)
    line = f"{family}\t{len(names)}\t{int(np.median(sz)) if len(sz) else 0}\t{int((M.getnnz(0) > 0).sum())}\t{cls}\t{src}\n"
    open(OUT / "MANIFEST_A.tsv", "a").write(line); print(line.strip(), flush=True)


def ps_sets(Z, pert, readout, zthr=4.0, cap=200, min_n=10):
    """Z [n_pert, n_readout] z-scores. pert / readout: universe indices (-1 = not in universe)."""
    A = np.abs(np.nan_to_num(Z))
    resp, reg = {}, {}
    for r in range(Z.shape[0]):
        if pert[r] < 0: continue
        hit = np.flatnonzero(A[r] >= zthr)
        if len(hit) < min_n: continue
        hit = hit[np.argsort(-A[r, hit])][:cap]
        resp[f"resp_{U.ensembl_gene_id[pert[r]]}"] = [pert[r]] + [readout[c] for c in hit]
    for c in range(Z.shape[1]):
        if readout[c] < 0: continue
        hit = np.flatnonzero((A[:, c] >= zthr) & (pert >= 0))
        if len(hit) < min_n: continue
        hit = hit[np.argsort(-A[hit, c])][:cap]
        reg[f"reg_{U.ensembl_gene_id[readout[c]]}"] = [readout[c]] + [pert[r] for r in hit]
    return resp, reg


def build_ps():
    def rd_genes(f): return [l.strip() for l in open(f) if l.strip()]
    specs = {
        "k562": (DER / "ps_k562_gwps.npy", DER / "ps_k562_gwps_genes.txt", "/mnt/msadb/subex/raw/perturbseq/K562_gwps_normalized_bulk_01.h5ad",
                 "Replogle 2022 K562 genome-wide Perturb-seq (2022)"),
        "rpe1": (DER / "ps_rpe1_ess.npy", DER / "ps_rpe1_ess_genes.txt", "/mnt/msadb/subex/raw/perturbseq/rpe1_normalized_bulk_01.h5ad",
                 "Replogle 2022 RPE1 essential Perturb-seq (2022)"),
        "cd4": (DER / "ps_cd4_2025.npy", DER / "ps_cd4_2025_genes.txt", None, "GWCD4i genome-wide CD4 T cell Perturb-seq (2025)"),
    }
    for cell, (fz, fg, h5, src) in specs.items():
        Z = np.load(fz)
        pg = rd_genes(fg)
        assert len(pg) == Z.shape[0], (cell, len(pg), Z.shape)
        if h5:
            rg = [x.decode() for x in h5py.File(h5, "r")["var"]["gene_id"][:]]
        else:
            rg = list(np.load(DER / "ps_cd4_2025_bulk.npz")["genes"])
        assert len(rg) == Z.shape[1], (cell, len(rg), Z.shape)
        pert = np.array([i_ensg.get(g, -1) for g in pg]); readout = np.array([i_ensg.get(g, -1) for g in rg])
        resp, reg = ps_sets(Z, pert, readout)
        save(f"a_ps_{cell}_resp", resp, "modern", src + ": perturbation -> responders", hi=201)
        save(f"a_ps_{cell}_reg", reg, "modern", src + ": readout -> regulators", hi=201)


def build_depmap():
    d = pd.read_csv("/mnt/msadb/subex/raw/depmap_20q4/CRISPR_gene_effect.csv", index_col=0)
    ent = [int(c.split("(")[1].rstrip(")")) for c in d.columns]
    ce = set(pd.read_csv("/mnt/msadb/subex/raw/depmap_20q4/CRISPR_common_essentials.csv")["gene"])
    idx = np.array([i_entrez.get(e, -1) for e in ent])
    keep = idx >= 0
    X = d.loc[:, keep].to_numpy(np.float32); idx = idx[keep]; cols = d.columns[keep]
    X = X - np.nanmean(X, 0); X = np.nan_to_num(X)
    sd = X.std(0)
    Xn = X / np.maximum(sd, 1e-6)
    C = (Xn.T @ Xn) / X.shape[0]
    np.fill_diagonal(C, -1)
    sel = (sd >= np.quantile(sd, 0.75)) & np.array([c not in ce for c in cols])
    sets = {}
    for j in np.flatnonzero(sel):
        top = np.argsort(-C[j])[:25]
        top = top[C[j, top] >= 0.2]
        sets[f"codep_{U.ensembl_gene_id[idx[j]]}"] = [idx[j]] + idx[top].tolist()
    save("a_dm_codep", sets, "pre21", "DepMap 20Q4 Chronos: selective gene + top-25 co-dependent genes (r>=0.2)", lo=4, hi=26)


def mp_ancestors():
    par, cur = {}, None
    for l in open("/home/ph/biotools/data/evidence/raw/mgi/MPheno_OBO.ontology"):
        l = l.strip()
        if l == "[Term]": cur = None
        elif l.startswith("id: MP:"): cur = l[4:]
        elif l.startswith("is_a: MP:") and cur: par.setdefault(cur, []).append(l[6:16])
    memo = {}
    def anc(t):
        if t in memo: return memo[t]
        s = {t}
        for p in par.get(t, []): s |= anc(p)
        memo[t] = s; return s
    return anc


def build_impc():
    h = pd.read_csv("/home/ph/biotools/data/evidence/raw/mgi/HOM_MouseHumanSequence.rpt", sep="\t", dtype=str)
    mm = h[h["NCBI Taxon ID"] == "10090"][["DB Class Key", "Mouse MGI ID"]]
    hs = h[h["NCBI Taxon ID"] == "9606"][["DB Class Key", "EntrezGene ID"]]
    hs = hs.assign(i=hs["EntrezGene ID"].astype(float).map(lambda e: i_entrez.get(int(e), -1) if pd.notna(e) else -1))
    o = mm.merge(hs[hs.i >= 0], on="DB Class Key")
    mgi2h = o.groupby("Mouse MGI ID")["i"].apply(list).to_dict()
    d = pd.read_csv("/home/ph/code/modet/data/raw/impc_release-13.0/results_genotype-phenotype-assertions-ALL.csv.gz",
                    usecols=["marker_accession_id", "mp_term_id"]).dropna().drop_duplicates()
    anc = mp_ancestors()
    sets = {}
    for mgi, t in d.itertuples(index=False):
        hg = mgi2h.get(mgi)
        if not hg: continue
        for a in anc(t):
            sets.setdefault(a, set()).update(hg)
    save("a_impc_orth", {k: list(v) for k, v in sets.items()}, "pre21",
         "IMPC 13.0 (2020-07) MP assertions, MGI orthology classes, MP is_a propagated", hi=1000)


def build_pubtator():
    """Streams both PubTator3 files in chunks (a DuckDB join was OOM-killed on the shared machine); pmid x gene and
    pmid x mesh binary sparse matrices; per cut: counts = D[pmid<=cut].T @ G[pmid<=cut] (distinct co-mentioning PMIDs)."""
    cmax = max(PMID_CUT.values()); NP = cmax + 1
    G = "/home/ph/code/modeng/data/raw/pubtator3/gene2pubtator3.gz"
    Dz = "/home/ph/biotools/data/evidence/raw/pubtator3/disease2pubtator3.gz"
    ent = {str(k): v for k, v in i_entrez.items()}
    gp, gg = [], []
    for ch in pd.read_csv(G, sep="\t", header=None, usecols=[0, 2], names=["p", "c"], dtype={"p": np.int64, "c": str},
                          chunksize=5_000_000, quoting=3, on_bad_lines="skip"):
        ch = ch[ch.p <= cmax].dropna()
        ch = ch.assign(c=ch.c.str.split(";")).explode("c")
        gi = ch.c.map(ent)
        ok = gi.notna()
        gp.append(ch.p[ok].to_numpy(np.int32)); gg.append(gi[ok].to_numpy(np.int32))
    gp, gg = np.concatenate(gp), np.concatenate(gg)
    print("gene rows", len(gp), flush=True)
    has = np.zeros(NP, bool); has[gp] = True
    dp, dm, mesh = [], [], {}
    for ch in pd.read_csv(Dz, sep="\t", header=None, usecols=[0, 2], names=["p", "c"], dtype={"p": np.int64, "c": str},
                          chunksize=5_000_000, quoting=3, on_bad_lines="skip"):
        ch = ch[(ch.p <= cmax)].dropna()
        ch = ch[has[ch.p.to_numpy()] & ch.c.str.startswith("MESH:")]
        codes = ch.c.map(lambda x: mesh.setdefault(x, len(mesh)))
        dp.append(ch.p.to_numpy(np.int32)); dm.append(codes.to_numpy(np.int32))
    dp, dm = np.concatenate(dp), np.concatenate(dm)
    print("disease rows", len(dp), "mesh", len(mesh), flush=True)
    names = np.array(sorted(mesh, key=mesh.get))
    for k, c in PMID_CUT.items():
        a = gp <= c; b = dp <= c
        Gm = sp.csr_matrix((np.ones(a.sum(), np.float32), (gp[a], gg[a])), shape=(NP, N)); Gm.data[:] = 1
        Dm = sp.csr_matrix((np.ones(b.sum(), np.float32), (dp[b], dm[b])), shape=(NP, len(mesh))); Dm.data[:] = 1
        C = (Dm.T.tocsr() @ Gm).tocsr()
        sets = {}
        for r in range(C.shape[0]):
            lo, hi = C.indptr[r], C.indptr[r + 1]
            v, ix = C.data[lo:hi], C.indices[lo:hi]
            sel = v >= 2
            if sel.sum() < 5: continue
            v, ix = v[sel], ix[sel]
            sets[names[r]] = ix[np.argsort(-v)[:300]].tolist()
        save(f"a_{k}", sets, "pre21" if k != "pt25" else "pre2509",
             f"PubTator3 gene x MeSH disease co-mention (>=2 PMIDs, top300), PMID <= {c}", hi=300)
        del Gm, Dm, C


if __name__ == "__main__":
    for w in sys.argv[1:]:
        {"ps": build_ps, "depmap": build_depmap, "impc": build_impc, "pubtator": build_pubtator}[w]()
