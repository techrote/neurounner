"""Post-hoc, read-only DG020 intervention: swap ONLY wet-tile forecasts.

Keeps dry-tile donor predictions, observations and receiver weights fixed.
This is an explanatory robustness check, NOT a preregistered pass criterion.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch
from experiment import Receiver


def audit(path):
    p = Path(path)
    receiver = Receiver().eval()
    receiver.load_state_dict(torch.load(p/"receiver-conditioned.pt", map_location="cpu", weights_only=True))
    results = {}
    for family in ("edge_bands", "alternating"):
        receipts = [json.loads(line) for line in (p/f"receipts-{family}.jsonl").read_text().splitlines()]
        counts = dict(wet_only_correct_high=0, dry_only_preserved_low=0,
                      wet_only_flipped=0, dry_only_flipped=0, original_low=0)
        with torch.no_grad():
            for r in receipts:
                obs = torch.tensor(np.array(r["observation"], dtype=np.float32))[None]
                low = torch.tensor(r["local_donor_predictions"][0], dtype=torch.float32)[None]
                high = torch.tensor(r["local_donor_predictions"][1], dtype=torch.float32)[None]
                wet = obs[..., 0] > .5
                only_wet = torch.where(wet, high, low)
                only_dry = torch.where(~wet, high, low)
                original = int(receiver(obs, low).argmax(-1).item())
                w = int(receiver(obs, only_wet).argmax(-1).item())
                d = int(receiver(obs, only_dry).argmax(-1).item())
                counts["wet_only_correct_high"] += int(w == r["oracle_actions"][1])
                counts["dry_only_preserved_low"] += int(d == r["oracle_actions"][0])
                counts["wet_only_flipped"] += int(w != original)
                counts["dry_only_flipped"] += int(d != original)
                counts["original_low"] += int(original == r["oracle_actions"][0])
        results[family] = dict(n=len(receipts), counts=counts)
    return results


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(json.dumps(dict(seed=Path(path).name, audit=audit(path)), sort_keys=True))
