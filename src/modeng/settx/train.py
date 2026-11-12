"""settx training / inference (temporal benchmark; train on `past` episodes only, all <= T0).

Cross-fitting: past tasks are split into K folds by modet cluster; fold k model (trained on the other folds) predicts the
past tasks of fold k (these OOF scores feed the stacker's past-trained run, so the channel is never in-sample there);
the model trained on all past episodes predicts temporal dev and confirm (never trained on any 21.04 -> 25.09 arrival).

    gpu-run --mem 12 -- python -m modeng.settx.train --variant all --views all [--no-ctx] [--permute 1]
Outputs data/cache/settx/pred_<variant>_{past,temporal_dev,temporal_confirm}.npz (ids, S float32 [n, n_genes]).
"""
from __future__ import annotations

import argparse, time
import numpy as np
import torch
import torch.nn.functional as F

from .. import tasks as T
from . import data as SD
from .model import SetTX
from .groups import leak_folds

DEV = "cuda"


def ctx_tensor(tj, pop, deg):
    """[n_ep, G] numpy -> [n_ep, G, 5] float16 on GPU."""
    tj_t = torch.from_numpy(tj).to(DEV)
    pop_t = torch.from_numpy(pop).to(DEV)
    G = tj.shape[1]
    pct = (tj_t.argsort(1).argsort(1).float() / G)
    pct[tj_t <= 0] = 0
    ld = torch.log1p(torch.from_numpy(deg.astype(np.float32)).to(DEV))[None].expand_as(tj_t)
    out = torch.stack([torch.log1p(tj_t * 10), pct, (tj_t <= 0).float(), torch.log1p(pop_t) / 5, ld / 5], -1).half()
    del tj_t, pop_t
    return out




def fold_ids(tasks, k, seed=0):
    cl = np.array([t.cluster or t.disease_id for t in tasks])
    uc = np.unique(cl); rng = np.random.default_rng(seed); rng.shuffle(uc)
    f = {c: i % k for i, c in enumerate(uc)}
    return np.array([f[c] for c in cl])


def make_batch(tasks, idx, ctx, deg, rng, max_seed=200, max_pos=64, n_neg=384, n_hard=128, hard_pool=None):
    B = len(idx)
    S = max(min(len(tasks[i].seeds), max_seed) for i in idx)
    C = max_pos + n_neg + n_hard
    seeds = np.zeros((B, S), np.int64); smask = np.ones((B, S), bool)
    cand = np.zeros((B, C), np.int64); cmask = np.zeros((B, C), bool); w = np.zeros((B, C), np.float32)
    ispos = np.zeros((B, C), bool)
    nseed = np.zeros(B, np.float32)
    G = ctx.shape[1]
    for b, i in enumerate(idx):
        t = tasks[i]
        s = t.seeds if len(t.seeds) <= max_seed else rng.choice(t.seeds, max_seed, replace=False)
        seeds[b, :len(s)] = s; smask[b, :len(s)] = False
        nseed[b] = len(t.seeds)
        tg = t.targets if len(t.targets) <= max_pos else rng.choice(t.targets, max_pos, replace=False)
        bad = np.zeros(G, bool); bad[t.exclude] = True; bad[t.targets] = True
        neg = rng.integers(0, G, n_neg * 2); neg = neg[~bad[neg]][:n_neg]
        hp = hard_pool[i]; hp = hp[~bad[hp]]
        hard = rng.choice(hp, min(n_hard, len(hp)), replace=False) if len(hp) else np.array([], np.int64)
        g = np.concatenate([tg, neg, hard])[:C]
        cand[b, :len(g)] = g; cmask[b, :len(g)] = True; ispos[b, :len(tg)] = True
        wp = 1.0 / np.maximum(deg[tg], 1.0) ** 0.5
        w[b, :len(tg)] = wp / wp.sum()
    return seeds, smask, cand, cmask, ispos, w, nseed


def listwise_loss(logit, cmask, ispos, w):
    """Sampled softmax per positive over {positive} + all negatives of its episode, weighted (deg^-0.5, sum 1 / episode)."""
    neg = cmask & ~ispos
    lneg = logit.masked_fill(~neg, -1e4)
    lse_neg = torch.logsumexp(lneg, 1, keepdim=True)                     # [B, 1]
    lse = torch.logaddexp(logit, lse_neg.expand_as(logit))
    nll = -(logit - lse)
    return (nll * w * ispos).sum() / ispos.any(1).sum()


def val_ap(model, tasks, ctx, ctx_idx, Xs, M, deg):
    """mean AP and degree-weighted AP (positives deg^-0.5) on held-out episodes, exclude removed."""
    from sklearn.metrics import average_precision_score
    S = predict(model, tasks, ctx, ctx_idx, Xs, M)
    model.train()
    aps, naps = [], []
    for i, t in enumerate(tasks):
        ok = np.ones(S.shape[1], bool); ok[t.exclude] = False
        y = np.zeros(S.shape[1], bool); y[t.targets] = True
        w = np.where(y, 1.0 / np.maximum(deg, 1) ** 0.5, 1.0)
        aps.append(average_precision_score(y[ok], S[i, ok]))
        naps.append(average_precision_score(y[ok], S[i, ok], sample_weight=w[ok]))
    return np.mean(aps), np.mean(naps)


def train_model(tasks, ctx, ctx_idx, Xs, M, deg, steps, args, seed=0, log="", val=None, aux=None):
    """aux = (aux_tasks, aux_ctx_idx): self-supervised episodes mixed in with probability args.aux_frac per episode."""
    pw = np.full(len(tasks), 1.0 / len(tasks))
    if aux is not None and args.aux_frac > 0:
        pw = np.r_[pw * (1 - args.aux_frac), np.full(len(aux[0]), args.aux_frac / len(aux[0]))]
        tasks = list(tasks) + list(aux[0]); ctx_idx = np.r_[ctx_idx, aux[1]]
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = SetTX([x.shape[1] for x in Xs], d=args.d, drop=args.drop, use_ctx=not args.no_ctx,
                  resid=args.resid).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    # hard-negative pool: top-2000 genes by transfer score per episode (non-zero only)
    tjc = ctx[ctx_idx, :, 1].float()  # percentile
    hard_pool = []
    for j in range(len(tasks)):
        v = tjc[j]
        k = int(min(2000, (v > 0).sum().item()))
        hard_pool.append(torch.topk(v, k).indices.cpu().numpy() if k > 0 else np.array([], np.int64))
    del tjc
    t0 = time.time(); run = 0.0
    model.no_genes = args.no_genes
    model.train()
    for step in range(steps):
        idx = rng.choice(len(tasks), args.batch, replace=False, p=pw)
        seeds, smask, cand, cmask, ispos, w, nseed = make_batch(tasks, idx, ctx, deg, rng, hard_pool=hard_pool)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            H = model.enc(Xs, M)
            if args.no_genes:
                H = H * 0
            sd_t = torch.from_numpy(seeds).to(DEV); cd_t = torch.from_numpy(cand).to(DEV)
            sm_t = torch.from_numpy(smask).to(DEV)
            tok, z = model.encode_seeds(H[sd_t], sm_t)
            ei = torch.from_numpy(ctx_idx[idx]).to(DEV)
            cx = ctx[ei[:, None], cd_t].float()
            logit = model.score(H[cd_t], tok, z, sm_t, cx, torch.from_numpy(nseed).to(DEV)).float()
        loss = listwise_loss(logit, torch.from_numpy(cmask).to(DEV), torch.from_numpy(ispos).to(DEV),
                             torch.from_numpy(w).to(DEV))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        run = 0.98 * run + 0.02 * loss.item() if step else loss.item()
        if step % 200 == 0 or step == steps - 1:
            msg = ""
            if val is not None and (step % 500 == 0 or step == steps - 1):
                a_, n_ = val_ap(model, val[0], ctx, val[1], Xs, M, deg)
                msg = f" val AP {a_:.4f} wAP {n_:.4f}"
            print(f"  {log} step {step} loss {run:.4f}{msg} ({time.time()-t0:.0f}s)", flush=True)
    return model


@torch.no_grad()
def predict(model, tasks, ctx, ctx_idx, Xs, M, chunk=4096):
    model.eval()
    G = ctx.shape[1]
    out = np.zeros((len(tasks), G), np.float32)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        H = model.enc(Xs, M)
        if getattr(model, "no_genes", False):
            H = H * 0
        for j, t in enumerate(tasks):
            s = torch.from_numpy(t.seeds).to(DEV)
            sm = torch.zeros(1, len(t.seeds), dtype=torch.bool, device=DEV)
            tok, z = model.encode_seeds(H[s][None], sm)
            ns = torch.tensor([float(len(t.seeds))], device=DEV)
            for a in range(0, G, chunk):
                g = torch.arange(a, min(G, a + chunk), device=DEV)
                cx = ctx[int(ctx_idx[j]), g].float()[None]
                out[j, a:a + len(g)] = model.score(H[g][None], tok, z, sm, cx, ns).float()[0].cpu().numpy()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--views", default="all", choices=["all", "clean", "seq"])
    ap.add_argument("--aux-sources", default="ctx1904,hpo2101,msig_go,msig_reactome,msig_wiki,impc13,complexportal")
    ap.add_argument("--no-ctx", action="store_true")
    ap.add_argument("--permute", type=int, default=None, help="negative control: permute gene rows of the views")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--only", default="all", help="all | final | folds | tune (train on 90 % of past, report held-out past)")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--no-genes", action="store_true", help="sanity: zero gene embeddings (context features only)")
    ap.add_argument("--aux", action="store_true", help="mix in self-supervised episodes (19.04 context rows + pre21 donor families)")
    ap.add_argument("--aux-frac", type=float, default=0.5)
    ap.add_argument("--resid", action="store_true", help="additive context-only path")
    ap.add_argument("--wd", type=float, default=0.01)
    args = ap.parse_args()
    torch.set_num_threads(4)
    t0 = time.time()
    P = T.load("past")
    Bt = T.load("temporal")
    deg = np.asarray(P.degree, np.float32)
    Xs, M, names = SD.gene_inputs(args.views, args.permute)
    print(f"views {names}", flush=True)
    Xs = [torch.from_numpy(x).to(DEV) for x in Xs]; M = torch.from_numpy(M).to(DEV)
    fp = SD.episode_features("past")
    fd = SD.episode_features("temporal", "dev"); fc = SD.episode_features("temporal", "confirm")
    ctx = torch.cat([ctx_tensor(fp["tj"], fp["pop"], deg), ctx_tensor(fd["tj"], fd["pop"], deg),
                     ctx_tensor(fc["tj"], fc["pop"], deg)], 0)
    nP, nD = len(P.tasks), len(fd["ids"])
    Pidx = np.arange(nP); Didx = nP + np.arange(nD); Cidx = nP + nD + np.arange(len(fc["ids"]))
    aux = None
    if args.aux:
        A = SD.aux_episodes(tuple(args.aux_sources.split(",")))
        n0 = ctx.shape[0]
        ctx = torch.cat([ctx, ctx_tensor(A["tj"], A["pop"], deg)], 0)
        aux = (A["tasks"], n0 + np.arange(len(A["tasks"])))
        print(f"aux episodes {len(A['tasks'])}", flush=True)
    print(f"inputs ready ({time.time()-t0:.0f}s), ctx {tuple(ctx.shape)}", flush=True)
    out = SD.CACHE
    if args.only == "tune":
        fid, info = leak_folds(P.tasks, 10)
        print("tune folds", info, flush=True)
        tr, te = np.flatnonzero(fid != 0), np.flatnonzero(fid == 0)[:120]
        from sklearn.metrics import average_precision_score
        aps = []
        for i in te:
            t = P.tasks[i]; ok = np.ones(P.n_genes, bool); ok[t.exclude] = False
            y = np.zeros(P.n_genes, bool); y[t.targets] = True
            w = np.where(y, 1.0 / np.maximum(deg, 1) ** 0.5, 1.0)
            v = fp["tj"][i]
            aps.append((average_precision_score(y[ok], v[ok]), average_precision_score(y[ok], v[ok], sample_weight=w[ok])))
        print("transfer_jaccard on val: AP %.4f wAP %.4f" % tuple(np.mean(aps, 0)), flush=True)
        train_model([P.tasks[i] for i in tr], ctx, Pidx[tr], Xs, M, deg, args.steps, args, log="tune", aux=aux,
                    val=([P.tasks[i] for i in te], Pidx[te]))
        return
    if args.only in ("all", "folds"):
        fid, info = leak_folds(P.tasks, args.folds)
        print("folds", info, flush=True)
        Sp = np.zeros((nP, P.n_genes), np.float32)
        for k in range(args.folds):
            tr, te = np.flatnonzero(fid != k), np.flatnonzero(fid == k)
            m = train_model([P.tasks[i] for i in tr], ctx, Pidx[tr], Xs, M, deg, args.steps, args, seed=k, log=f"fold{k}", aux=aux)
            Sp[te] = predict(m, [P.tasks[i] for i in te], ctx, Pidx[te], Xs, M)
            del m; torch.cuda.empty_cache()
            print(f"fold {k} done ({time.time()-t0:.0f}s)", flush=True)
        np.savez(out / f"pred_{args.variant}_past.npz", ids=np.array([t.disease_id for t in P.tasks]), S=Sp)
    if args.only in ("all", "final"):
        m = train_model(P.tasks, ctx, Pidx, Xs, M, deg, args.steps, args, seed=100, log="final", aux=aux)
        for split, idx in (("dev", Didx), ("confirm", Cidx)):
            Bs = Bt.split(split)
            S = predict(m, Bs.tasks, ctx, idx, Xs, M)
            np.savez(out / f"pred_{args.variant}_temporal_{split}.npz", ids=np.array([t.disease_id for t in Bs.tasks]), S=S)
        torch.save(m.state_dict(), out / f"model_{args.variant}.pt")
    print(f"done ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
