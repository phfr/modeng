"""R3 red team, same-window test: the modeng-v1 recipe trained ONLY on temporal dev arrivals (OT 21.04 -> 25.09, 53 diseases) is applied
unchanged to the forward benchmark (25.09 -> 26.09, 797 confirm diseases, forward channel cache). If the cv-trained stacker had mainly
learned 'which genes arrive in 2021-2025' or window-specific pipeline behaviour, it should not beat transfer in another window.
pri_mend_n is dropped (NaN on forward). Variants: full | nolabelproc (gwas_catlag, pri_l2g_*, hz_gwas dropped).
Output: results/review2/crosswindow_<variant>.tsv, log to stdout."""
import sys, time
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
LABELPROC = ["gwas_catlag", "pri_l2g_n", "pri_l2g_max", "hz_gwas"]
Fw = T.load("forward"); Fc = Fw.split("confirm"); degf = np.asarray(Fw.degree, np.float32)
conf_idx = np.array([i for i, t in enumerate(Fw.tasks) if t.split == "confirm"])
print("forward confirm tasks", len(Fc.tasks), flush=True)


def fwd(name):
    return np.load(f"{ROOT}/data/cache/channels/forward_confirm_{name}.npy", mmap_mode="r")


def evalf(S_iter):
    out = []
    for i, s in S_iter:
        t = Fc.tasks[i]; m = M.rank_metrics(s, t.targets, t.exclude, degf); r = dict(disease=t.disease_id, ap=m["ap"], nuap=m["nuap"], auroc=m["auroc"], r100=m["recall@100"])
        if "onnet" in t.alt:
            mo = M.rank_metrics(s, *t.alt["onnet"], degf); r["onnet_ap"], r["onnet_nuap"] = mo["ap"], mo["nuap"]
        out.append(r)
    return pd.DataFrame(out)


tr = fwd("transfer_jaccard"); assert tr.shape[0] == len(Fc.tasks)
et = evalf((i, np.asarray(tr[i])) for i in range(len(Fc.tasks)))
print(f"forward confirm transfer_jaccard AP {et.ap.mean():.4f} nuAP {et.nuap.mean():.4f}", flush=True)
try:
    ecv = pd.read_parquet(ROOT + "/results/forward/confirm/stack_v1_p0.35.parquet")
    print(f"forward cv-trained modeng-v1 (saved): AP {ecv.ap.mean():.4f} nuAP {ecv.nuap.mean():.4f}")
except Exception as e:
    ecv = None; print("no saved forward per-task file", str(e)[:100])
for variant in sys.argv[1:]:
    t0 = time.time(); names = [n for n in NAMES if n != "pri_mend_n" and not (variant == "nolabelproc" and n in LABELPROC)]
    Fd = feats([cached("dev", n) for n in names], Bd)
    model = fit(*rows(Bd, np.arange(len(Bd.tasks)), Fd))
    mm = [fwd(n) for n in names]; ns = np.log1p(np.array([len(t.seeds) for t in Fc.tasks], np.float32)); allg = np.arange(Fc.n_genes)
    def scores():
        for a in range(0, len(Fc.tasks), 40):
            ch = np.arange(a, min(a + 40, len(Fc.tasks))); mats = [np.asarray(m[ch]) for m in mm]; ranks = [ST.pct_rank(x) for x in mats]
            for j, i in enumerate(ch):
                yield i, model.predict(ST.task_X(j, allg, mats, ranks, ns[ch]), num_threads=LGB["num_threads"])
    e = evalf(scores()); e.to_csv(OUT + f"crosswindow_{variant}.tsv", sep="\t", index=False, float_format="%.5f")
    cl = e.disease.to_numpy()
    print(f"[{variant}] temporal-dev-trained -> forward confirm: AP {e.ap.mean():.4f} nuAP {e.nuap.mean():.4f} AUROC {e.auroc.mean():.4f} R@100 {e.r100.mean():.4f} onnet AP {e.onnet_ap.mean():.4f} nuAP {e.onnet_nuap.mean():.4f} ({time.time()-t0:.0f}s)")
    print(f"   vs forward transfer (task bootstrap, tasks treated as independent): AP {fmt(boot(e.ap, et.ap, cl))} nuAP {fmt(boot(e.nuap, et.nuap, cl))}", flush=True)
    if ecv is not None and len(ecv) == len(e):
        print(f"   vs forward cv-trained modeng-v1: AP {fmt(boot(e.ap, ecv.ap.to_numpy(), cl))} nuAP {fmt(boot(e.nuap, ecv.nuap.to_numpy(), cl))}", flush=True)
