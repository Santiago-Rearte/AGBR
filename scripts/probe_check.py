"""Dev tool (inside the image): risk of the probe rule on class/module-level mutants.

For each mutant whose change is at class or module level: heavy candidates and their
probes (closest cheap coverage neighbours that are candidates).  Runs the probes; when
all pass, runs the gated heavy tests and reports any failure (a would-be miss).

  python /opt/scripts/probe_check.py <n> <seed> <tests.json> [out.jsonl]
"""

import json
import os
import pickle
import random
import sys
import time

sys.path.insert(0, "/opt/br")
sys.path.insert(0, "/opt/scripts")
import analysis  # noqa: E402
import mutants  # noqa: E402
import patchlib  # noqa: E402
import runner  # noqa: E402
import selector  # noqa: E402

REPO = "/opt/sqlglot"


def main():
    n, seed, tj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    verifiable = set(json.load(open("/opt/maps/verifiable.json"))["passed"])
    T = maps["tests"]
    sec = {t["id"]: t["seconds"] for t in json.load(open(tj))["tests"]}
    dur = {i: int(sec.get(t, 0) * 1000 + 0.5) for i, t in enumerate(T)}
    files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
    an = analysis.Analyzer(maps, REPO)
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    near = maps["near"]
    stats = {"mutants": 0, "gated": 0, "probe_failed": 0, "heavy_checked": 0, "violations": 0, "saved_s": 0.0}
    tries = 0
    while stats["mutants"] < n and tries < n * 50:
        tries += 1
        rel, ln, label, patch, new_text = mutants.mutate(rng, files)
        bits, notes, static = an.analyze(patchlib.parse(patch))
        if static or not an.trace.get("import") or an.trace.get("nonsite"):
            continue
        stats["mutants"] += 1
        cands = set(analysis.iter_bits(bits))
        heavy = [i for i in cands if dur[i] >= selector.HEAVY_SEC * 1000]
        probes, gated = set(), []
        for h in heavy:
            picks = [i for i in near.get(h, ()) if i in cands and dur[i] < selector.HEAVY_SEC * 1000 and T[i] in verifiable]
            picks = picks[: selector.PROBES_PER_HEAVY]
            if len(picks) >= selector.MIN_PROBES:
                probes.update(picks)
                gated.append(h)
        rec = {"file": rel, "line": ln, "label": label, "cands": len(cands), "gated": [T[h].split("::")[-1] for h in gated]}
        if gated:
            stats["gated"] += 1
            root = f"/tmp/probe{stats['mutants']}"
            runner.make_overlay(root, REPO, {rel: new_text})
            job = runner.Job("p", {"root": root, "nodeids": [T[i] for i in sorted(probes)], "default_timeout": 120.0, "stop_on_fail": True, "import_check": []}, 1800)
            runner.run_jobs([job], parallel=1)
            rec["probe_cpu"] = round(job.cpu, 2)
            if len(job.passed) < len(probes):
                stats["probe_failed"] += 1
                rec["probes"] = "failed"
            else:
                rec["probes"] = "passed"
                t0 = time.monotonic()
                hjob = runner.Job("h", {"root": root, "nodeids": [T[h] for h in gated], "default_timeout": 900.0}, 3600)
                runner.run_jobs([hjob], parallel=1)
                stats["heavy_checked"] += 1
                failed = sorted(T[h] for h in gated if T[h] not in hjob.passed)
                rec["heavy_failed"] = failed
                rec["heavy_secs"] = round(time.monotonic() - t0, 1)
                if failed:
                    stats["violations"] += 1
                else:
                    saved = (selector.pack(cands, dur)[0] - selector.pack(cands - set(gated), dur)[0]) / 1000
                    stats["saved_s"] += saved - job.cpu
                    rec["saved"] = round(saved, 2)
            os.system(f"rm -rf {root}")
        print(json.dumps(rec), flush=True)
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
