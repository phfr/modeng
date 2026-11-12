"""r4 sub-threshold loci: extract SNPs with p <= 1e-5 from modet's 108 harmonised <= T0 summary statistics, clump to independent leads.

  ~/biotools/venvs/bio-net/bin/python scripts/subthr_extract.py [--jobs 7]

Input  (read-only): modet/data/raw/gwas_sumstats_t0/<gcst>/*.h.tsv.gz (GWAS Catalog harmonised files; studies dated <= 2021-04-23),
                    modet 1000G EUR .bim (build 37 positions; same as modet's MAGMA / lead calling).
Output: data/subthr/snps_v2/<gcst>.parquet (P, id0..idk = the rsID columns of the file; p <= 1e-5)
        data/subthr/leads.parquet (gcst, SNP, P, chr, pos, gws): greedy distance clumping over ALL p <= 1e-5 SNPs (+-500 kb, by p),
        gws = lead p <= 5e-8. Sub-threshold leads are leads with 5e-8 < p <= 1e-5 that are > 500 kb from every stronger lead.
"""
import argparse, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

R = Path("/home/ph/code/modeng")
SUM = Path("/home/ph/code/modet/data/raw/gwas_sumstats_t0")
BIM = Path("/home/ph/code/modet/data/raw/magma/g1000_eur/g1000_eur.bim")
OUT = R / "data/subthr"; (OUT / "snps_v2").mkdir(parents=True, exist_ok=True)
THR, WIN = 1e-5, 500_000


_B = None


def _bim_ids():
    global _B
    if _B is None:
        _B = pd.Index(pd.read_csv(BIM, sep="\t", header=None, usecols=[1]).iloc[:, 0].drop_duplicates())
        _B.get_indexer(["rs1"])   # build the hash engine once (threads reuse it)
    return _B


def extract(gcst: str) -> str:
    o = OUT / "snps_v2" / f"{gcst}.parquet"
    if o.exists():
        return gcst + " cached"
    f = next((SUM / gcst).glob("*.tsv.gz"))
    hdr = pd.read_csv(f, sep="\t", nrows=0).columns.tolist()
    snpcols = [c for c in ("rs_id", "rsid", "hm_rsid", "variant_id") if c in hdr]
    k = hdr.index("p_value") + 1
    cols = ",".join(f"${hdr.index(c) + 1}" for c in snpcols)
    awk = (f'BEGIN{{FS=OFS="\\t"}} {{sub(/\\r$/,"")}} NR>1 && ${k} ~ /^[0-9.eE+-]+$/ && ${k}+0 <= {THR} {{print ${k},{cols}}}')
    p = subprocess.run(f"zcat '{f}' | LC_ALL=C mawk '{awk}'", shell=True, capture_output=True, text=True, check=True)
    rows = [l.split("\t") for l in p.stdout.splitlines()]
    d = pd.DataFrame(rows, columns=["P"] + snpcols)
    # keep every rsID column; the SNP id is chosen in the single-threaded clumping step (first column whose id is in the 1000G panel)
    d["P"] = pd.to_numeric(d.P, errors="coerce").clip(lower=1e-300)
    d = d[d.P.notna()].rename(columns={c: f"id{i}" for i, c in enumerate(snpcols)})
    d.to_parquet(o, index=False)
    return f"{gcst} {len(d)}"


def clump(d: pd.DataFrame) -> pd.DataFrame:
    d = d.sort_values("P")
    kept, pos = [], {}
    for r in d.itertuples(index=False):
        k = pos.setdefault(r.chr, [])
        if all(abs(r.pos - q) > WIN for q in k):
            k.append(r.pos); kept.append(r)
    return pd.DataFrame(kept, columns=d.columns)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=7); a = ap.parse_args()
    sys.path.insert(0, str(R / "src"))
    from modeng.ext import gwas_d as G
    S = G.studies(); S = S[S.src == "t0"]
    gs = [g for g in S.gcst if any((SUM / g).glob("*.tsv.gz"))]
    print(len(gs), "studies; max date", S.date.max(), flush=True)
    with ThreadPoolExecutor(a.jobs) as ex:
        for msg in ex.map(extract, gs):
            print(msg, flush=True)
    # 1000G positions only for the extracted ids (mawk hash filter of the 22 M-row bim; pandas joins against the full bim take > 10 s per study)
    ids = set()
    for g in gs:
        d = pd.read_parquet(OUT / "snps_v2" / f"{g}.parquet")
        for c in [c for c in d.columns if c.startswith("id")]:
            ids |= set(d[c].dropna())
    (OUT / "ids.txt").write_text("\n".join(sorted(i for i in ids if str(i).startswith("rs"))) + "\n")
    subprocess.run(f"LC_ALL=C mawk 'NR==FNR{{a[$1];next}} ($2 in a){{print $1\"\\t\"$2\"\\t\"$4}}' '{OUT / 'ids.txt'}' '{BIM}' > '{OUT / 'bim_sub.tsv'}'",
                   shell=True, check=True)
    bim = pd.read_csv(OUT / "bim_sub.tsv", sep="\t", header=None, names=["chr", "SNP", "pos"], dtype={"chr": str}).drop_duplicates("SNP").set_index("SNP")
    print(len(ids), "ids;", len(bim), "in 1000G", flush=True)
    L = []
    for g in gs:
        d = pd.read_parquet(OUT / "snps_v2" / f"{g}.parquet")
        z = d.P <= 1e-300
        if z.mean() > 0.05:          # broken p_value column (GCST004461: 1.4 M rows with p = 0): treat p = 0 as missing
            print(g, "drops", int(z.sum()), "rows with p = 0", flush=True); d = d[~z]
        snp = pd.Series(None, index=d.index, dtype=object)
        for c in [c for c in d.columns if c.startswith("id")]:
            snp = snp.where(snp.notna(), d[c].where(bim.index.get_indexer(d[c].fillna("")) >= 0))
        d = pd.DataFrame({"SNP": snp, "P": d.P}).dropna().sort_values("P").drop_duplicates("SNP").join(bim, on="SNP")
        d = d[~((d.chr == "6") & d.pos.between(25e6, 34e6))]
        c = clump(d); c["gcst"] = g; L.append(c)
    L = pd.concat(L, ignore_index=True); L["gws"] = L.P <= 5e-8
    L.to_parquet(OUT / "leads.parquet", index=False)
    print(L.groupby("gws").size(), L.groupby("gcst").gws.agg(["size", "sum"]).describe(), flush=True)
