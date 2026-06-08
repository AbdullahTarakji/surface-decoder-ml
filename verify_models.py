import torch
from pathlib import Path

for p in sorted(Path("models").glob("*.pt")):
    blob = torch.load(str(p), map_location="cpu", weights_only=True)
    print(p.name, "| kind:", blob["kind"], "| n_features:", blob["n_features"],
          "| d=%d T=%d p=%.0e" % (blob["distance"], blob["rounds"], blob["noise_p"]),
          "| state_dict keys:", len(blob["state_dict"]))
