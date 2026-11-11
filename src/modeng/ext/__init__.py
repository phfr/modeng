"""Extension channels written by implementer agents. Each module `modeng/ext/<name>.py` defines
CHANNELS = {"<channel_name>": fn} with fn(B, split_tag) -> np.ndarray [n_tasks, n_genes] float32 (NaN allowed = missing).
fn may use only B.tasks seeds, B.context_for(task) (masked context) and inputs allowed by the benchmark's leakage class
(state it in the module docstring). Discovered automatically by modeng.channels.compute."""
import importlib, pkgutil

def registry():
    reg = {}
    for m in pkgutil.iter_modules(__path__):
        mod = importlib.import_module(f"{__name__}.{m.name}")
        reg.update(getattr(mod, "CHANNELS", {}))
    return reg
