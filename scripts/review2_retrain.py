"""R3 red team: retrain modeng-v1 (train on all 53 dev tasks -> score 138 confirm tasks) under ablations that test
same-window gene memorisation, label-process channels and input vintage. Usage: review2_retrain.py <variant> [...]
Variants: ref | nogene (gene-level channels dropped) | permgene (gene-level channels: gene axis permuted with ONE fixed permutation,
dev and confirm alike = a per-gene identity hash without content) | nolabelproc | pt19 (PubTator cut 2019-03 for dA_pt/pt_ridge*) |
catlag12 | rand17<seed> (17 random dev tasks, size control for norel) | nov3 | seed<k> (other negative sample and LightGBM seed) | halves (two models, each trained on rows of one random half of the genes; 'seen' = gene scored by the model that
trained on its half, 'unseen' = by the other; per-task rank-normalised before splicing) | norel (dev tasks sharing a bootstrap cluster or an ontology relation with a confirm task dropped)
Output: results/review2/S_<variant>.npy (float16) and results/review2/retrain_<variant>.tsv (per-task endpoints)."""
import sys, time
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
from modeng import data as D

GENELEVEL = ["degree", "popularity", "pri_l2g_n", "pri_l2g_max", "pri_mend_n", "pri_hpo_n", "hz_gwas", "hz_mend"]
LABELPROC = ["gwas_catlag", "pri_l2g_n", "pri_l2g_max", "hz_gwas"]


def load(names, sub=None):
    sub = sub or {}
    return ([sub[n][0] if n in sub else cached("dev", n) for n in names], [sub[n][1] if n in sub else cached("confirm", n) for n in names])


def run(variant):
    t0 = time.time(); names = list(NAMES); sub = {}; gene_ok = None; idx = np.arange(len(Bd.tasks))
    if variant == "nogene":
        names = [n for n in names if n not in GENELEVEL]
    elif variant == "nolabelproc":
        names = [n for n in names if n not in LABELPROC]
    elif variant == "nov3":
        names = [n for n in names if not (n.startswith("pri_") or n.startswith("hz_") or n.startswith("ph_") or n.startswith("pt_ridge"))]
    elif variant.startswith("permgene"):
        p = np.random.default_rng(int(variant[8:] or 7)).permutation(Bd.n_genes)
        sub = {n: (cached("dev", n)[:, p], cached("confirm", n)[:, p]) for n in GENELEVEL}
    elif variant == "catlag12":
        sub = {"gwas_catlag": (cached("dev", "gwas_catlag12"), cached("confirm", "gwas_catlag12"))}
    elif variant == "pt19":
        from modeng.ext import donors_a as DA, lead_ptridge as PR
        DA.PT_BY_BENCH["temporal"] = "a_pt19"; PR.PT_BY_BENCH["temporal"] = "a_pt19"
        for n, fn in (("dA_pt", DA._pt), ("pt_ridge", lambda B, s: PR._ptr(B, s)), ("pt_ridge_pos", lambda B, s: PR._ptr(B, s, True))):
            sub[n] = (np.asarray(fn(Bd, "dev19"), np.float32), np.asarray(fn(Bc, "confirm19"), np.float32))
            print(n, "pt19 vs pt21 corr (confirm)", np.corrcoef(np.nan_to_num(sub[n][1]).ravel(), np.nan_to_num(cached("confirm", n)).ravel())[0, 1], flush=True)
    md, mc = load(names, sub); Fd, Fc = feats(md, Bd), feats(mc, Bc)
    if variant.startswith("halves"):
        half = np.random.default_rng(int(variant[6:] or 11)).random(Bd.n_genes) < 0.5
        SA = predict(fit(*rows(Bd, idx, Fd, gene_ok=half)), Bc, Fc); SB = predict(fit(*rows(Bd, idx, Fd, gene_ok=~half)), Bc, Fc)
        # rank-normalise each model per task so the two halves are comparable when spliced
        def rk(S):
            return np.argsort(np.argsort(S, 1), 1).astype(np.float32) / S.shape[1]
        RA, RB = rk(SA), rk(SB)
        outs = {variant + "_seen": np.where(half[None], RA, RB), variant + "_unseen": np.where(half[None], RB, RA)}
    else:
        if variant.startswith("rand17"):
            # size control for norel: 17 random dev tasks
            idx = np.sort(np.random.default_rng(int(variant[6:] or 0)).choice(len(Bd.tasks), 17, replace=False))
        if variant == "norel":
            # drop dev tasks sharing a bootstrap cluster or an ontology relation with any confirm task
            cc = {t.cluster for t in Bc.tasks}; cid = {t.disease_id for t in Bc.tasks}
            idx = np.array([i for i, t in enumerate(Bd.tasks) if t.cluster not in cc and not (set(D.relatives(t.disease_id)) & cid)])
            print("norel: dev tasks kept", len(idx), "of", len(Bd.tasks), flush=True)
        sd = int(variant[4:]) if variant.startswith("seed") else 0   # seed<k>: other negative sample + LightGBM seed
        LGB["seed"] = sd
        m = fit(*rows(Bd, idx, Fd, seed=sd)); outs = {variant: predict(m, Bc, Fc)}
        imp = pd.Series(m.feature_importance("gain"), index=names + ["r_" + n for n in names] + ["log_nseeds"]).sort_values(ascending=False)
        imp.to_csv(OUT + f"imp_{variant}.tsv", sep="\t", header=False)
    for k, S in outs.items():
        np.save(OUT + f"S_{k}.npy", S.astype(np.float16)); e = ep(S, Bc); e.to_csv(OUT + f"retrain_{k}.tsv", sep="\t", index=False, float_format="%.5f")
        print(f"{k:22s} AP {e.ap.mean():.4f} nuAP {e.nuap.mean():.4f} onnetAP {e.onnet_ap.mean():.4f} onnet nuAP {e.onnet_nuap.mean():.4f} "
              f"sa_onnet AP {e.sa_onnet_ap.mean():.4f} nuAP {e.sa_onnet_nuap.mean():.4f} AUROC {e.auroc.mean():.4f} ({time.time()-t0:.0f}s)", flush=True)


for v in sys.argv[1:]:
    run(v)
