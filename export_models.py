"""
export_models.py — Convert Lightning checkpoints to app.py-compatible .pt files.

The app expects ./models/<name>.pt with keys:
  kind, state_dict, hparams, n_features, distance, rounds, noise_p

weights_only=False is required here because Lightning .ckpt files embed Python
objects (hparams dicts, loop state, etc.) that pickle encodes. These are
locally-generated training artifacts — not downloaded from external sources —
so unpickling them is safe in this context.
"""
from pathlib import Path
import torch

ROOT = Path(__file__).parent
CKPT_ROOT = ROOT / "checkpoints" / "iter1_mlp"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

# Validation-selected seeds sorted by val_loss (best first).
# All trained on d=3, T=3 rounds, p=1e-3 — the Phase-3 Iter-1 setting.
DISTANCE, ROUNDS, NOISE_P = 3, 3, 1e-3

seeds_by_val_loss = [
    {"seed": 3,  "val_loss": 0.00279},
    {"seed": 1,  "val_loss": 0.00281},
    {"seed": 0,  "val_loss": 0.00284},
    {"seed": 42, "val_loss": 0.00284},
    {"seed": 2,  "val_loss": 0.00288},
]

exported = []
for run in seeds_by_val_loss:
    seed = run["seed"]
    # Pick the latest checkpoint version for this seed (best-vN.ckpt > best.ckpt).
    ckpt_dir = CKPT_ROOT / f"seed{seed}"
    candidates = sorted(ckpt_dir.glob("best*.ckpt"),
                        key=lambda p: (len(p.stem), p.stem))  # best < best-v1 < best-v2
    if not candidates:
        print(f"  seed {seed}: no checkpoint found, skipping")
        continue
    ckpt_path = candidates[-1]

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    hp = ckpt["hyper_parameters"]

    blob = {
        "kind":       "mlp",
        "state_dict": ckpt["state_dict"],          # keys: net.0.weight, net.0.bias, …
        "hparams":    {k: hp[k] for k in ("hidden", "depth", "dropout")},
        "n_features": hp["n_features"],
        "distance":   DISTANCE,
        "rounds":     ROUNDS,
        "noise_p":    NOISE_P,
    }

    label = f"seed{seed}_vl{run['val_loss']:.5f}"
    out_path = MODELS_DIR / f"mlp_d{DISTANCE}_{label}.pt"
    # weights_only not applicable to torch.save — it's a save option, not load.
    torch.save(blob, str(out_path))
    exported.append(out_path.name)
    print(f"  Exported seed {seed:2d} (val_loss={run['val_loss']:.5f}) -> {out_path.name}")

print(f"\nDone. {len(exported)} model(s) written to ./models/")
