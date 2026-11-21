"""Dev tool (inside the image): labeled junk-insertion edits to given data files.

Like the sample's fixture commits (a junk token inserted into a random line).  The
ground truth runs every test that reads the file (heavy ones included, in full,
without slicing), so it is exact for edits to data only those tests read.

  python /opt/scripts/make_corpus_data.py <n> <seed> <file> [file ...] > corpus.jsonl
"""

import difflib
import json
import os
import pickle
import random
import sys

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import runner  # noqa: E402

REPO = "/opt/sqlglot"
JUNK = ["42", "xa", "NULL", "0"]


def main():
    n, seed, files = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3:]
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    T = maps["tests"]
    bad_base = {t for t, o in maps["outcomes"].items() if o != "passed"}
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    for k in range(n):
        rel = files[k % len(files)]
        lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        i = rng.choice([x for x, l in enumerate(lines) if l.strip()])
        words = lines[i].split(" ")
        j = rng.randrange(len(words) + 1)
        new = lines[:]
        new[i] = " ".join(words[:j] + [rng.choice(JUNK)] + words[j:])
        diff = difflib.unified_diff(lines, new, f"a/{rel}", f"b/{rel}", lineterm="", n=3)
        patch = f"diff --git a/{rel} b/{rel}\n" + "\n".join(diff) + "\n"
        readers = sorted({T[x] for x in analysis.iter_bits(maps["reads"].get(rel, 0))} - bad_base)
        root = f"/tmp/cd{k}"
        runner.make_overlay(root, REPO, {rel: "\n".join(new).encode()})
        job = runner.Job(0, {"root": root, "nodeids": readers, "default_timeout": 900.0, "import_check": []}, 7200)
        runner.run_jobs([job], parallel=1)
        os.system(f"rm -rf {root}")
        failing = sorted(set(readers) - job.passed)
        print(json.dumps({"id": f"d{seed}_{k:03d}", "file": rel, "line": i + 1, "label": "fixture",
                          "patch": patch, "failing": failing}), flush=True)
        sys.stderr.write(f"{k} {rel}:{i + 1} failing={[t.split('::')[-1] for t in failing]} done={job.done}\n")


if __name__ == "__main__":
    main()
