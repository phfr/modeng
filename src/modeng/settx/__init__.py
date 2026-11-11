"""settx: identity-blind episodic set-completion transformer (idea ML-5, implementer F, round 1).

episodes.py  builds self-supervised training episodes ("hide 20-40 % of a set, rank the rest") from (a) the benchmark's
             context label matrix (disease rows, family-scope masking with ancestor re-propagation, as the harness's
             holdout masking) and (b) virtual donor gene-set families (data/donors), each with its per-episode
             kNN-transfer score computed inside the episode's own family.
model.py     gene encoder over the 15 views (per-view projection + missing-view mask; no gene identity embedding),
             per-(episode, gene) context features (transfer score, popularity, degree), seed self-attention,
             candidate -> seed cross-attention (candidates never attend to each other), score head.
train.py     disease-grouped cross-fitting over the benchmark's tasks; writes data/cache/settx/pred_<bench>_<variant>.npz
"""
