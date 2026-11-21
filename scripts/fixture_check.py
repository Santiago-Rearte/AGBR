"""Dev tool (inside the image): recall of data-file selections on junk-insertion edits.

For files read by shared test setup (class setup or module collection), inserts a
junk token into a random line (like the sample's fixture commits), then runs every
test that reads the file at all (heavy ones included) and reports failing tests the
analysis did not select.

  python /opt/scripts/fixture_check.py <n> <seed> [file ...]
"""

import difflib
import json
import os
import pickle
import random
import sys

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import patchlib  # noqa: E402
import runner  # noqa: E402

REPO = "/opt/sqlglot"


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    T = maps["tests"]
    an = analysis.Analyzer(maps, REPO)
    files = sys.argv[3:] or sorted(f for f in maps.get("group_reads", {}) if f.endswith(".sql"))
    print(f"files: {files}", flush=True)
    rng = random.Random(seed)
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    bad_base = {t for t, o in maps["outcomes"].items() if o != "passed"}
    misses = 0
    for k in range(n):
        rel = rng.choice(files)
        lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        idx = [i for i, l in enumerate(lines) if l.strip()]
        i = rng.choice(idx)
        words = lines[i].split(" ")
        j = rng.randrange(len(words) + 1)
        new = lines[:]
        new[i] = " ".join(words[:j] + [rng.choice(["42", "xa", "NULL", "0"])] + words[j:])
        diff = difflib.unified_diff(lines, new, f"a/{rel}", f"b/{rel}", lineterm="", n=3)
        patch = f"diff --git a/{rel} b/{rel}\n" + "\n".join(diff) + "\n"
        bits, notes, _ = an.analyze(patchlib.parse(patch.encode()))
        cands = {T[x] for x in analysis.iter_bits(bits)}
        readers = sorted({T[x] for x in analysis.iter_bits(maps["reads"].get(rel, 0))} - bad_base)
        root = f"/tmp/fx{k}"
        runner.make_overlay(root, REPO, {rel: "\n".join(new).encode()})
        job = runner.Job(0, {"root": root, "nodeids": readers, "default_timeout": 900.0, "import_check": []}, 7200)
        runner.run_jobs([job], parallel=1)
        failing = set(readers) - job.passed
        missing = sorted(failing - cands)
        misses += bool(missing)
        print(json.dumps({"k": k, "file": rel, "line": i + 1, "readers": len(readers), "cands": len(cands),
                          "failing": sorted(t.split("::")[-1] for t in failing)[:6], "n_failing": len(failing),
                          "missing": missing[:6], "done": job.done}), flush=True)
        os.system(f"rm -rf {root}")
    print(f"edits={n} misses={misses}")


if __name__ == "__main__":
    main()
