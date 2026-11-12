"""R1 red-team: dA_pt on temporal with the 2019-03 PubTator cut (2-year literature lag) -> results/review/pt19_{split}.npy"""
import sys, numpy as np
sys.path.insert(0, "src")
from modeng import tasks as T
from modeng.ext import donors_a as A
B = T.load("temporal"); M = A.load_any("a_pt19")
for s in ("dev", "confirm"):
    np.save(f"results/review/pt19_{s}.npy", A.hyper_transfer(B.split(s).tasks, M, **A.PARAMS))
    print(s, "done", flush=True)
