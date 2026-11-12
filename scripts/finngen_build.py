"""r6: FinnGen R4 SuSiE credible sets -> variant table with hg19 positions and 1000G rsIDs (data/finngen/cs.parquet).

Input: data/raw/finngen_r4/susie_cs/<endpoint>.cs.tsv.gz (rows of finngen_R4_<endpoint>.SUSIE.snp.bgz with cs != -1, streamed;
the full files are not kept). Positions GRCh38 (FinnGen R4) -> GRCh37 with the UCSC hg38ToHg19 chain (point liftover, own code),
rsID = modet's 1000G EUR .bim SNP at the same hg19 chr:pos with matching alleles (for cS2G). Lead of a credible set = max PIP.
"""
import gzip, subprocess, sys
from pathlib import Path
import numpy as np, pandas as pd

RAW = Path("/home/ph/code/modeng/data/raw/finngen_r4/susie_cs")
CHAIN = Path("/home/ph/code/modeng/data/raw/ucsc_chain/hg38ToHg19.over.chain.gz")
BIM = Path("/home/ph/code/modet/data/raw/magma/g1000_eur/g1000_eur.bim")
OUT = Path("/home/ph/code/modeng/data/finngen")


def read_chain():
    """per target chrom: arrays (tstart, size, qstart, qchrom, qstrand_minus, qsize, score) of ungapped blocks."""
    blocks = {}
    with gzip.open(CHAIN, "rt") as fh:
        for line in fh:
            if line.startswith("chain"):
                f = line.split(); score = float(f[1]); tn, ts = f[2], int(f[5]); qn, qsize, qstr, qs = f[7], int(f[8]), f[9], int(f[10])
                t, q = ts, qs; continue
            f = line.split()
            if not f:
                continue
            size = int(f[0])
            blocks.setdefault(tn, []).append((t, size, q, qn, qstr == "-", qsize, score))
            if len(f) == 3:
                t += size + int(f[1]); q += size + int(f[2])
    out = {}
    for c, b in blocks.items():
        b.sort(key=lambda x: (x[0], -x[6]))
        out[c] = b
    return out


def lift(chain, chrom, pos1):
    """1-based hg38 positions -> (hg19 chrom without 'chr', 1-based pos) or (None, nan)."""
    c = "chr" + chrom.replace("chr", "")
    b = chain.get(c, [])
    starts = np.array([x[0] for x in b])
    oc, op = [], []
    for p in pos1:
        p0 = p - 1
        i = np.searchsorted(starts, p0, side="right") - 1
        res = (None, np.nan)
        # scan back a few blocks (overlapping chains), take highest score containing p0
        best = -1
        for j in range(i, max(i - 50, -1), -1):
            t, size, q, qn, minus, qsize, score = b[j]
            if t <= p0 < t + size and score > best:
                q0 = q + (p0 - t)
                if minus:
                    q0 = qsize - 1 - q0
                res = (qn.replace("chr", ""), q0 + 1); best = score
        oc.append(res[0]); op.append(res[1])
    return oc, op


def main():
    rows = []
    for f in sorted(RAW.glob("*.cs.tsv.gz")):
        d = pd.read_csv(f, sep="\t", dtype={"chromosome": str})
        if len(d):
            rows.append(d)
    d = pd.concat(rows, ignore_index=True)
    d["chr38"] = d.chromosome.str.replace("chr", "").replace({"23": "X"})
    chain = read_chain()
    d["chr19"], d["pos19"] = None, np.nan
    for c, g in d.groupby("chr38"):
        oc, op = lift(chain, c, g.position.to_numpy())
        d.loc[g.index, "chr19"] = oc; d.loc[g.index, "pos19"] = op
    ok = d.chr19 == d.chr38
    print(f"{len(d)} credible-set variants, {d.trait.nunique()} endpoints; lifted to same chrom: {ok.mean():.3f}", flush=True)
    d.loc[~ok, ["chr19", "pos19"]] = [None, np.nan]
    # 1000G rsIDs at hg19 chr:pos
    keys = OUT / "keys.txt"
    dd = d.dropna(subset=["pos19"])
    pd.Series((dd.chr19 + ":" + dd.pos19.astype(int).astype(str)).unique()).to_csv(keys, index=False, header=False)
    sub = OUT / "bim_sub.tsv"
    subprocess.run(f"LC_ALL=C mawk 'NR==FNR{{a[$1];next}} (($1\":\"$4) in a){{print $1\"\\t\"$4\"\\t\"$2\"\\t\"$5\"\\t\"$6}}' '{keys}' '{BIM}' > '{sub}'",
                   shell=True, check=True)
    bim = pd.read_csv(sub, sep="\t", header=None, names=["chr19", "pos19", "rsid1kg", "a1", "a2"], dtype={"chr19": str})
    d["pos19i"] = d.pos19.fillna(-1).astype(int)
    m = d.merge(bim, left_on=["chr19", "pos19i"], right_on=["chr19", "pos19"], how="left", suffixes=("", "_b"))
    al = lambda a, b: (a == b)
    okal = (al(m.allele1, m.a1) & al(m.allele2, m.a2)) | (al(m.allele1, m.a2) & al(m.allele2, m.a1))
    m.loc[~okal.fillna(False), "rsid1kg"] = None
    m = m.sort_values("rsid1kg", na_position="last").drop_duplicates(["trait", "region", "cs", "v"])
    m = m[["trait", "region", "cs", "v", "chr38", "position", "chr19", "pos19", "rsid1kg", "maf", "beta", "se", "p", "prob"]]
    m["lead"] = m.groupby(["trait", "region", "cs"]).prob.transform("max") == m.prob
    m.to_parquet(OUT / "cs.parquet", index=False)
    print(f"rsID matched {m.rsid1kg.notna().mean():.3f}; credible sets {m.groupby(['trait','region','cs']).ngroups}; "
          f"lead p<=5e-8 share {(m[m.lead].p <= 5e-8).mean():.3f}", flush=True)
    keys.unlink(); sub.unlink()


if __name__ == "__main__":
    main()
