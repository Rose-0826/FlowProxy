"""Check actual neural inference causality and input contracts on both routes.
Run run_demo.py for the chosen device first to prepare its relocated runtime.
This check writes no predictions or report files; JSON is printed to stdout.
"""
from pathlib import Path
import argparse
import json
import os
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args()
    runtime = ROOT / ("runtime_" + args.device)
    if not (runtime / "system.json").is_file():
        raise SystemExit("Run run_demo.py for this device first to prepare the runtime.")
    os.environ.update(AIC_ROOT=str(runtime), OMP_NUM_THREADS="1",
                      OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    import numpy as np
    sys.path.insert(0, str(runtime / "work/evidence_upgrade_20261007"))
    from lean_entry import Predictor
    model = Predictor(runtime / "system.json")
    paths = sorted((ROOT / "demo_inputs").glob("*_input.json"))
    cases = [next(p for p in paths if p.name.startswith(prefix))
             for prefix in ["cylinder_", "ellipse_a1p4_"]]
    records = []
    for path in cases:
        info = json.loads(path.read_text(encoding="utf-8"))
        with np.load(path.with_suffix(".npz")) as arrays:
            row = {key: arrays[key] for key in ["fields", "geom", "mask", "times"]}
            values = arrays["values"]
        row.update(id=info["id"], q=info["q"])
        baseline, _ = model.predict(row, values)
        changed = values.copy()
        changed[79:] += 100.0  # NaNs stay NaN; 76 scalar budget is unchanged.
        perturbed, _ = model.predict(row, changed)
        assert np.array_equal(baseline[:80], perturbed[:80]), info["id"]
        assert np.any(baseline[80:] != perturbed[80:]), info["id"]
        invalid = values.copy()
        index = tuple(np.argwhere(np.isfinite(invalid))[0])
        invalid[index] = np.nan
        try:
            model.predict(row, invalid)
        except AssertionError:
            pass
        else:
            raise AssertionError("75 future scalars were accepted")
        malformed = dict(row, fields=row["fields"][:3])
        try:
            model.predict(malformed, values)
        except AssertionError:
            pass
        else:
            raise AssertionError("Three startup frames were accepted")
        records.append(dict(id=info["id"], future_scalars=76,
                            earlier_80_frames_bitwise_unchanged=True,
                            later_frames_respond=True,
                            invalid_75_scalar_budget_rejected=True,
                            invalid_three_frame_start_rejected=True))
    print(json.dumps(dict(status="PASSED", device=args.device, cases=records), indent=2))


if __name__ == "__main__":
    main()
