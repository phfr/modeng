"""r6: label-process share of FinnGen-only traced strict-a targets (temporal dev): does the OT 25.09 gwas_credible_sets evidence
for (disease or descendant, gene) include a FinnGen study (study via OT 26.09 credible-set table, studyLocusId)?"""
import glob
import numpy as np, pandas as pd
from modeng import data as D

tr = pd.read_csv("/home/ph/code/modeng/results/finngen/trace_temporal_dev.tsv", sep="\t")
u = D.universe()
pairs = [(r.disease_id, u[int(g)]) for r in tr.itertuples() if isinstance(r.fg_only_genes, str) for g in r.fg_only_genes.split(";")]
ev = pd.concat([pd.read_parquet(f, columns=["targetId", "diseaseId", "studyLocusId", "score"])
                for f in glob.glob("/mnt/msadb/modet/downloads/ot_evidence/evidence_2509/sourceId=gwas_credible_sets/*.parquet")])
cs = pd.read_parquet("/home/ph/biotools/data/evidence/raw/opentargets/26.09/credible_set_gwas_slim.parquet")
print(cs.columns.tolist()[:20])
sl = cs[["studyLocusId", "studyId"]].drop_duplicates("studyLocusId").set_index("studyLocusId").studyId
ev["studyId"] = ev.studyLocusId.map(sl)
rows = []
for d, g in pairs:
    fam = {d} | set(D.descendants(d))
    e = ev[(ev.targetId == g) & ev.diseaseId.isin(fam)]
    st = e.studyId.dropna().unique().tolist()
    rows.append(dict(disease_id=d, gene=g, symbol=D.symbols().get(g), n_ev=len(e), n_mapped=int(e.studyId.notna().sum()),
                     finngen=any(str(s).startswith("FINNGEN") for s in st), studies=";".join(map(str, st[:6]))))
df = pd.DataFrame(rows)
df.to_csv("/home/ph/code/modeng/results/finngen/labelproc_temporal_dev.tsv", sep="\t", index=False)
print(df.to_string())
print(f"FinnGen-only traced strict-a: {len(df)}; with FinnGen study in 25.09 evidence: {df.finngen.sum()}; with any credible-set evidence: {(df.n_ev>0).sum()}")
