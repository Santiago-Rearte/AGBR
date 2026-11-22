"""Dev tool (inside the image): are `is` -> `==` edits the analysis calls equivalent
really harmless?  Mutates covered lines, keeps those the analysis marks "equiv",
runs every test covering the line, and reports any that fail.

  python /opt/scripts/equiv_check.py <n> <seed>
"""

import difflib
import os
import pickle
import random
import re
import sys

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import patchlib  # noqa: E402
import runner  # noqa: E402

REPO = "/opt/sqlglot"


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    T = maps["tests"]
    ok = {t for t, o in maps["outcomes"].items() if o == "passed"}
    an = analysis.Analyzer(maps, REPO)
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    sites = []
    for rel, lines in maps["lines"].items():
        if not (rel.startswith("sqlglot/") and rel.endswith(".py")):
            continue
        src = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        for ln in lines:
            if ln - 1 < len(src) and re.search(r" is (?!not)", src[ln - 1]) and not src[ln - 1].strip().startswith("#"):
                sites.append((rel, ln))
    rng.shuffle(sites)
    done = bad = 0
    for rel, ln in sites:
        if done >= n:
            break
        old = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        line = old[ln - 1]
        ms = list(re.finditer(r" is (?!not)", line))
        m = rng.choice(ms)
        new = old[:]
        new[ln - 1] = line[: m.start()] + " == " + line[m.end():]
        diff = difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm="", n=3)
        patch = f"diff --git a/{rel} b/{rel}\n" + "\n".join(diff) + "\n"
        fps = patchlib.parse(patch.encode())
        _bits, notes, _ = an.analyze(fps)
        if not any(x.endswith(":equiv") for x in notes):
            continue
        sid = maps["lines"][rel][ln]
        readers = sorted(T[i] for i in analysis.iter_bits(maps["sets"][sid]) if T[i] in ok)
        root = f"/tmp/eq{done}"
        runner.make_overlay(root, REPO, {rel: "\n".join(new).encode()})
        job = runner.Job(done, {"root": root, "nodeids": readers, "default_timeout": 900.0, "import_check": []}, 7200)
        runner.run_jobs([job], parallel=1)
        os.system(f"rm -rf {root}")
        failing = sorted(set(readers) - job.passed)
        bad += bool(failing)
        print(f"{rel}:{ln} [{line.strip()[:70]}] tests={len(readers)} failing={[t.split('::')[-1] for t in failing][:4]}", flush=True)
        done += 1
    print(f"equiv edits={done} with failures={bad}")


if __name__ == "__main__":
    main()
