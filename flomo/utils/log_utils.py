import json
import os
from collections import Counter

import wandb


class BrokenSampleLog:
    """Records broken-sample events surfaced by `BaseDataset.__getitem__` retries.

    Three sinks:
    - Cumulative `skipped/{ds}` counters → wandb scalars (rank 0 only).
    - Dedup-by-(clip,start_t) `wandb.Table`, re-logged at every val pass.
    - Append-only `broken_samples.rank{r}.jsonl` ground truth on disk per rank.
    """

    def __init__(self, out_dir: str, rank: int):
        self.counters: Counter = Counter()
        self.unique: dict = {}
        os.makedirs(out_dir, exist_ok=True)
        self.jsonl_path = os.path.join(out_dir, f"broken_samples.rank{rank}.jsonl")
        # Rehydrate counters + dedup table from prior runs on resume.
        if os.path.exists(self.jsonl_path):
            with open(self.jsonl_path) as f:
                for line in f:
                    self._update_state(json.loads(line))
        self.jsonl = open(self.jsonl_path, "a", buffering=1)

    def _update_state(self, r: dict):
        ds = r["dataset"]
        self.counters[ds] += 1
        key = (ds, r["clip"], r["start_t"])
        if key in self.unique:
            self.unique[key]["hit_count"] += 1
            self.unique[key]["last_err"] = r["err_msg"]
        else:
            self.unique[key] = {
                "hit_count": 1, "err_type": r["err_type"],
                "last_err": r["err_msg"], "phase": r["phase"],
            }

    def ingest(self, records, phase: str):
        for r in records:
            event = {**r, "phase": phase}
            self.jsonl.write(json.dumps(event) + "\n")
            self._update_state(event)

    def counter_log(self):
        return {f"skipped/{ds}": c for ds, c in self.counters.items()}

    def wandb_table(self):
        cols = ["dataset", "clip", "start_t", "err_type", "last_err", "hit_count", "phase"]
        data = [[ds, clip, st, info["err_type"], info["last_err"], info["hit_count"], info["phase"]]
                for (ds, clip, st), info in self.unique.items()]
        return wandb.Table(columns=cols, data=data)
