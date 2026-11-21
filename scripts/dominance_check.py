"""Dev tool (inside the image): does the dominance rule ever drop a heavy test that fails?

Mutates lines that heavy tests execute.  When the selector would treat a heavy test as
dominated, runs the cheap candidates; if they all pass, runs the dominated heavy tests
too and reports any that fail (a would-be miss).

  python /opt/scripts/dominance_check.py <n> <seed> <tests.json> [out.jsonl]
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
MODE = "runtime"


def main():
    n, seed, tests_json = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    out_path = sys.argv[4] if len(sys.argv) > 4 else None
    global MODE
    MODE = sys.argv[5] if len(sys.argv) > 5 else "runtime"
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    calib = json.load(open("/opt/maps/verifiable.json"))
    verifiable = set(calib["passed"])
    T = maps["tests"]
    sec_by_id = {t["id"]: t["seconds"] for t in json.load(open(tests_json))["tests"]}
    idx_sec = [sec_by_id.get(t, 0.0) for t in T]
    cheap_all = 0
    heavy_all = 0
    for i, t in enumerate(T):
        if idx_sec[i] < selector.HEAVY_SEC and t in verifiable:
            cheap_all |= 1 << i
        elif idx_sec[i] >= selector.HEAVY_SEC:
            heavy_all |= 1 << i
    # Lines some heavy test executes.
    heavy_lines = []
    for rel, d in maps["lines"].items():
        if not rel.startswith("sqlglot/"):
            continue
        for ln, sid in d.items():
            if maps["sets"][sid] & heavy_all:
                heavy_lines.append((rel, ln))
    print(f"{len(heavy_lines)} lines executed by heavy tests", flush=True)
    all_files = sorted({rel for rel, _ in heavy_lines})
    an = analysis.Analyzer(maps, REPO)
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    stats = {"mutants": 0, "dominance": 0, "cheap_failed": 0, "heavy_checked": 0, "violations": 0}
    tries = 0
    while stats["mutants"] < n and tries < n * 200:
        tries += 1
        if MODE == "import":
            rel = rng.choice(all_files)
            ln = rng.randrange(1, len(open(os.path.join(REPO, rel), encoding="utf-8").read().split(chr(10))) + 1)
        else:
            rel, ln = rng.choice(heavy_lines)
        sub = random.Random(rng.random())
        # Reuse the generic mutator but pinned to this file/line.
        lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        m = mutants.mutate_line(sub, rel, lines, ln - 1)
        if m is None:
            continue
        label, patch, new_text = m
        bits, notes, static = an.analyze(patchlib.parse(patch))
        if MODE == "runtime" and (an.trace["other"] or not an.trace["runtime"]):
            continue  # only statements inside functions
        if MODE == "import" and (not an.trace.get("import") or an.trace.get("nonsite")):
            continue  # only class- or module-level changes
        stats["mutants"] += 1
        dom = selector.dominated_heavy(an, bits, idx_sec, cheap_all) if not static else 0
        rec = {"file": rel, "line": ln, "label": label, "dom": [T[i].split("::")[-1] for i in analysis.iter_bits(dom)]}
        if dom:
            stats["dominance"] += 1
            cheap = [T[i] for i in analysis.iter_bits(bits & cheap_all)]
            root = f"/tmp/dom{stats['mutants']}"
            runner.make_overlay(root, REPO, {rel: new_text})
            job = runner.Job("c", {"root": root, "nodeids": cheap, "default_timeout": 120.0, "stop_on_fail": True}, 1800)
            runner.run_jobs([job], parallel=1)
            if len(job.passed) < len(cheap):
                stats["cheap_failed"] += 1
                rec["cheap"] = "failed"
            else:
                heavy = [T[i] for i in analysis.iter_bits(dom)]
                t0 = time.monotonic()
                hjob = runner.Job("h", {"root": root, "nodeids": heavy, "default_timeout": 900.0}, 3600)
                runner.run_jobs([hjob], parallel=1)
                stats["heavy_checked"] += 1
                failed = sorted(set(heavy) - hjob.passed)
                rec["cheap"] = "passed"
                rec["heavy_failed"] = failed
                rec["heavy_secs"] = round(time.monotonic() - t0, 1)
                if failed:
                    stats["violations"] += 1
            os.system(f"rm -rf {root}")
        print(json.dumps(rec), flush=True)
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
