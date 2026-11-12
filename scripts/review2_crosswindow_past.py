"""R3 red team, same-window test in the other direction: the modeng-v1 recipe (minus the priors / HPO channels, which do not exist for
19.04) trained ONLY on temporal dev arrivals (21.04 -> 25.09) is applied to the 900 cached `past` episodes (seeds OT 19.04, targets new
by 21.04). Variants: full (v2 channels + pt_ridge + gwas_catlag) | nocatlag | clean (no don_* / view channels: several of those are dated
between 19.04 and 21.04, e.g. HPO 2021-01, and are transductive on past) | cleannocatlag. Reported for all episodes and for episodes whose disease
is neither a temporal dev (training) disease nor an ontology relative of one. Output: results/review2/crosswindow_past_<variant>.tsv"""
import sys, time
sys.path.insert(0, "/home/ph/code/modeng/scripts")
from review2_common import *
from modeng import data as D
P = T.load("past"); sel = np.sort(np.random.default_rng(0).choice(len(P.tasks), min(900, len(P.tasks)), replace=False)); PT = [P.tasks[i] for i in sel]
degp = np.asarray(P.degree, np.float32)
DROP = ["pri_l2g_n", "pri_l2g_max", "pri_mend_n", "pri_hpo_n", "hz_gwas", "hz_mend", "ph_mean", "ph_top3"]
pc = lambda n: np.load(f"{ROOT}/data/cache/channels/past_train900_{n}.npy", mmap_mode="r")
tr = pc("transfer_jaccard"); assert tr.shape[0] == len(PT)
# sanity: the cache rows belong to these tasks (seeds have a defined transfer score; recompute 3 rows)
chk = BL.transfer_jaccard([PT[i] for i in (0, 450, 899)], T.Benchmark(P.name, [PT[i] for i in (0, 450, 899)], P.context, P.context_ids, P.network, P.union_network, P.leakage_class, P.degree))
print("cache/task alignment, max|diff| transfer rows:", float(np.abs(chk - np.asarray(tr[[0, 450, 899]])).max()), flush=True)
def ev(it):
    return pd.DataFrame([dict(disease=PT[i].disease_id, cluster=PT[i].cluster, **{k: M.rank_metrics(s, PT[i].targets, PT[i].exclude, degp)[k] for k in ("ap", "nuap", "auroc")}) for i, s in it])
et = ev((i, np.asarray(tr[i])) for i in range(len(PT)))
dev = {t.disease_id for t in Bd.tasks}; rel = np.array([bool((set(D.relatives(t.disease_id)) | {t.disease_id}) & dev) for t in PT])
print(f"past episodes {len(PT)}; related to a temporal dev disease {rel.sum()}; transfer AP {et.ap.mean():.4f} nuAP {et.nuap.mean():.4f}", flush=True)
for variant in sys.argv[1:]:
    t0 = time.time(); names = [n for n in NAMES if n not in DROP and not (variant.endswith("nocatlag") and n == "gwas_catlag")]
    if variant.startswith("clean"):   # only inputs dated <= 19.04 on past: no virtual-donor families, no embedding views
        names = [n for n in names if not (n.startswith("don_") or n.startswith("vmean_") or n.startswith("vmax_"))]
    print(variant, len(names), "channels", flush=True)
    model = fit(*rows(Bd, np.arange(len(Bd.tasks)), feats([cached("dev", n) for n in names], Bd)))
    mm = [pc(n) for n in names]; ns = np.log1p(np.array([len(t.seeds) for t in PT], np.float32)); allg = np.arange(P.n_genes)
    def scores():
        for a in range(0, len(PT), 40):
            ch = np.arange(a, min(a + 40, len(PT))); mats = [np.asarray(m[ch]) for m in mm]; ranks = [ST.pct_rank(x) for x in mats]
            for j, i in enumerate(ch):
                yield i, model.predict(ST.task_X(j, allg, mats, ranks, ns[ch]), num_threads=LGB["num_threads"])
    e = ev(scores()); e.to_csv(OUT + f"crosswindow_past_{variant}.tsv", sep="\t", index=False, float_format="%.5f")
    for nm, s in (("all", np.ones(len(PT), bool)), ("unrelated to training diseases", ~rel)):
        print(f"[{variant}] temporal-dev-trained -> past ({nm}, n={s.sum()}): AP {e.ap[s].mean():.4f} vs transfer {et.ap[s].mean():.4f}: {fmt(boot(e.ap[s], et.ap[s], e.cluster[s]))}; "
              f"nuAP {e.nuap[s].mean():.4f} vs {et.nuap[s].mean():.4f}: {fmt(boot(e.nuap[s], et.nuap[s], e.cluster[s]))} ({time.time()-t0:.0f}s)", flush=True)
