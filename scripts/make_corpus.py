"""Dev tool (inside the image): a labeled corpus of synthetic mutants in /data format.

Ground truth comes from running every light test (< HEAVY seconds) on each mutant;
heavy tests are not run, so misses on them cannot be measured with this corpus.
Prints one JSON line per mutant: {"id", "patch", "failing": [test ids]}.

  python /opt/scripts/make_corpus.py <n> <seed> > corpus.jsonl
"""

import json
import os
import pickle
import random
import sys

sys.path.insert(0, "/opt/br")
sys.path.insert(0, "/opt/scripts")
import mutants  # noqa: E402
import runner  # noqa: E402

REPO = "/opt/sqlglot"
HEAVY = 1.5


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    tests = maps["tests"]
    dur = dict(zip(tests, maps["dur"]))
    sec = {t["id"]: t["seconds"] for t in json.load(open("/tests.json"))["tests"]}
    light = [t for t in tests if sec.get(t, dur.get(t, 0)) < HEAVY and maps["outcomes"].get(t) == "passed"]
    files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")

    def ground_truth(root):
        chunks = [light[j::2] for j in range(2)]
        jobs = [
            runner.Job(j, {"root": root, "nodeids": c, "default_timeout": 120.0, "import_check": []}, 3600)
            for j, c in enumerate(chunks)
        ]
        runner.run_jobs(jobs, parallel=2)
        return set(light) - set().union(*(j.passed for j in jobs))

    base_fail = ground_truth(REPO)
    sys.stderr.write(f"base failing in this env: {len(base_fail)}\n")
    for k in range(n):
        rel, ln, label, patch, new_text = mutants.mutate(rng, files)
        root = f"/tmp/corpus{k}"
        runner.make_overlay(root, REPO, {rel: new_text})
        failing = sorted(ground_truth(root) - base_fail)
        os.system(f"rm -rf {root}")
        print(json.dumps({"id": f"m{seed}_{k:03d}", "file": rel, "line": ln, "label": label, "patch": patch.decode(), "failing": failing}), flush=True)
        sys.stderr.write(f"{k} {rel}:{ln} {label} failing={len(failing)}\n")


if __name__ == "__main__":
    main()
