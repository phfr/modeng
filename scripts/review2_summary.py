"""R3 red team: paired cluster-bootstrap table of the retrain variants (results/review2/retrain_*.tsv) vs the reproduced modeng-v1 (ref)
and vs transfer_jaccard. Output: results/review2/summary.tsv"""
import sys, glob, os
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
ref = pd.read_csv(OUT + "retrain_ref.tsv", sep="\t"); cl = ref.cluster.to_numpy()
tr = ep(BL.transfer_jaccard(Bc.tasks, Bc), Bc)
saved = ep(np.load(ROOT + "/results/temporal/confirm/stack_v3ptr_p0.35.scores.npy").astype(np.float32), Bc)
print(f"reproduction: saved-score AP {saved.ap.mean():.4f} nuAP {saved.nuap.mean():.4f} | retrained ref AP {ref.ap.mean():.4f} nuAP {ref.nuap.mean():.4f} | max per-task |dAP| {np.abs(saved.ap - ref.ap).max():.4f}")
rows_ = []
order = ["ref", "nogene", "permgene", "permgene3", "halves_seen", "halves_unseen", "halves5_seen", "halves5_unseen", "nolabelproc", "nov3", "pt19", "catlag12", "norel", "rand171", "rand172", "rand173", "seed1", "seed2", "seed3"]
for v in order:
    f = OUT + f"retrain_{v}.tsv"
    if not os.path.exists(f):
        continue
    e = pd.read_csv(f, sep="\t"); r = dict(variant=v)
    for m in ("ap", "nuap", "onnet_ap", "onnet_nuap", "sa_onnet_ap", "sa_onnet_nuap"):
        r[m] = e[m].mean(); r["d_ref_" + m] = fmt(boot(e[m], ref[m], cl))
        if m in ("ap", "nuap"):
            r["d_transfer_" + m] = fmt(boot(e[m], tr[m], cl))
    rows_.append(r)
R = pd.DataFrame(rows_); R.to_csv(OUT + "summary.tsv", sep="\t", index=False, float_format="%.4f")
pd.set_option("display.width", 250)
print(R[["variant", "ap", "nuap", "onnet_ap", "onnet_nuap", "sa_onnet_ap", "sa_onnet_nuap"]].round(4).to_string(index=False))
print(R[["variant", "d_ref_ap", "d_ref_nuap", "d_transfer_ap", "d_transfer_nuap"]].to_string(index=False))
print(R[["variant", "d_ref_onnet_ap", "d_ref_onnet_nuap", "d_ref_sa_onnet_nuap"]].to_string(index=False))
def pair(a, b):
    fa, fb = OUT + f"retrain_{a}.tsv", OUT + f"retrain_{b}.tsv"
    if os.path.exists(fa) and os.path.exists(fb):
        x, y = pd.read_csv(fa, sep="\t"), pd.read_csv(fb, sep="\t")
        print(f"{a} - {b}: AP {fmt(boot(x.ap, y.ap, cl))} nuAP {fmt(boot(x.nuap, y.nuap, cl))} onnet nuAP {fmt(boot(x.onnet_nuap, y.onnet_nuap, cl))}")
pair("permgene", "nogene"); pair("permgene3", "nogene"); pair("halves_seen", "halves_unseen"); pair("halves5_seen", "halves5_unseen")
