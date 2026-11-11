"""Build 'virtual donor' gene-set families (non-disease gene sets) on the 19,255-gene universe.

Each family -> data/donors/<family>.npz : sparse CSR set x gene (0/1) as data/indices/indptr/shape + names (str array).
MANIFEST.tsv: family, n_sets, median size, date class (pre21 = content dated <= 2021-04; modern; seq), source.
Sets are kept if 5 <= size <= 500 genes.
"""
import sys, gzip, numpy as np, pandas as pd, scipy.sparse as sp
from pathlib import Path

OUT = Path("/home/ph/code/modeng/data/donors"); OUT.mkdir(parents=True, exist_ok=True)
RAW = Path("/home/ph/code/modet/data/raw")
U = pd.read_parquet("/home/ph/code/modet/data/processed/hgnc_protein_coding.parquet").sort_values("ensembl_gene_id").reset_index(drop=True)  # harness order
N = len(U)
i_ensg = {g: i for i, g in enumerate(U.ensembl_gene_id)}
i_sym = {s: i for i, s in enumerate(U.symbol)}
i_upper = {s.upper(): i for i, s in enumerate(U.symbol)}
i_entrez = {int(e): i for i, e in enumerate(U.entrez_id) if pd.notna(e)}
i_acc = {}
for i, a in enumerate(U.uniprot_ids.fillna("")):
    for x in str(a).split("|"):
        if x: i_acc.setdefault(x, i)


def save(family, sets: dict, cls, src, lo=5, hi=500):
    names, rows, cols = [], [], []
    for name, genes in sets.items():
        g = sorted({x for x in genes if x is not None and x >= 0})
        if lo <= len(g) <= hi:
            rows += [len(names)] * len(g); cols += g; names.append(str(name))
    M = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(names), N))
    np.savez_compressed(OUT / f"{family}.npz", data=M.data, indices=M.indices, indptr=M.indptr, shape=M.shape, names=np.array(names))
    sz = M.getnnz(1)
    line = f"{family}\t{len(names)}\t{int(np.median(sz)) if len(sz) else 0}\t{int((M.getnnz(0) > 0).sum())}\t{cls}\t{src}\n"
    open(OUT / "MANIFEST.tsv", "a").write(line); print(line.strip(), flush=True)


def gmt(path):
    d = {}
    for l in open(path):
        p = l.rstrip("\n").split("\t"); d[p[0]] = [i_sym.get(s, -1) for s in p[2:]]
    return d


def main(which):
    if "msigdb" in which:
        save("msig_go", gmt(RAW / "msigdb_v7.4/c5.go.v7.4.symbols.gmt"), "pre21", "MSigDB 7.4 C5 GO (2021-03)")
        save("msig_reactome", gmt(RAW / "msigdb_v7.4/c2.cp.reactome.v7.4.symbols.gmt"), "pre21", "MSigDB 7.4 Reactome")
        save("msig_wiki", gmt(RAW / "msigdb_v7.4/c2.cp.wikipathways.v7.4.symbols.gmt"), "pre21", "MSigDB 7.4 WikiPathways")
    if "hpo" in which:
        d = pd.read_csv(RAW / "hpo_annotations_2021-01_wayback/genes_to_phenotype_ALL_SOURCES_20210115.txt", sep="\t", comment="#", header=None)
        d["i"] = d[0].map(i_entrez)
        save("hpo2101", d.dropna(subset=["i"]).groupby(3)["i"].apply(lambda s: s.astype(int).tolist()).to_dict(), "pre21",
             "HPO genes_to_phenotype 2021-01-15 (derived from OMIM/Orphanet disease genes: disease-derived donor)")
    if "impc" in which:
        d = pd.read_csv(RAW / "impc_release-13.0/results_genotype-phenotype-assertions-ALL.csv.gz", usecols=["marker_symbol", "mp_term_id", "top_level_mp_term_id"])
        d["i"] = d.marker_symbol.str.upper().map(i_upper)
        d = d.dropna(subset=["i"])
        sets = d.groupby("mp_term_id")["i"].apply(lambda s: s.astype(int).tolist()).to_dict()
        sets.update({"TOP_" + k: v for k, v in d.groupby("top_level_mp_term_id")["i"].apply(lambda s: s.astype(int).tolist()).to_dict().items()})
        save("impc13", sets, "pre21", "IMPC release 13.0 (2020-07) MP assertions, mouse->human by symbol", hi=2000)
    if "complex" in which:
        d = pd.read_csv(RAW / "complex_portal_2021-03-10/9606.tsv", sep="\t")
        col = d.columns[4]
        sets = {r["#Complex ac"]: [i_acc.get(x.split("(")[0], -1) for x in str(r[col]).split("|")] for _, r in d.iterrows()}
        save("complexportal", sets, "pre21", "Complex Portal 2021-03-10", lo=2)
    if "paralog" in which:
        rows = []
        with gzip.open(RAW / "ensembl103_compara/Compara.103.protein_default.homologies.tsv.gz", "rt") as f:
            f.readline()
            for l in f:
                p = l.split("\t")
                if p[4] == "within_species_paralog" and p[7] == "homo_sapiens" and float(p[3]) >= 30:
                    rows.append((p[0], p[5]))
        sets = {}
        for a, b in rows:
            ia, ib = i_ensg.get(a, -1), i_ensg.get(b, -1)
            if ia >= 0 and ib >= 0: sets.setdefault(a, {ia}).add(ib)
        save("paralog30", {k: list(v) for k, v in sets.items()}, "seq", "Ensembl Compara 103 paralogs identity>=30 (one set per gene)", lo=2)
    if "clusters" in which:
        from sklearn.cluster import MiniBatchKMeans
        V = Path("/home/ph/code/modeng/data/views")
        for name in ["esm2_15b", "prott5", "depmap20q4", "hpa_sc", "gtex", "ps_k562", "ps_cd4", "phylo_npp"]:
            z = np.load(V / f"{name}.npz"); X, m = z["X"], z["mask"]; idx = np.flatnonzero(m)
            Xn = X[idx] / (np.linalg.norm(X[idx], axis=1, keepdims=True) + 1e-6)
            sets = {}
            for k in (50, 200, 800):
                lab = MiniBatchKMeans(k, random_state=0, batch_size=4096, n_init=3).fit_predict(Xn)
                for c in range(k): sets[f"{name}_k{k}_{c}"] = idx[lab == c].tolist()
            save(f"km_{name}", sets, "modern" if name.startswith(("ps_", "hpa", "gtex")) else "seq", f"k-means k=50/200/800 on view {name}", hi=800)


if __name__ == "__main__":
    which = sys.argv[1:] or ["msigdb", "hpo", "impc", "complex", "paralog", "clusters"]
    main(which)
