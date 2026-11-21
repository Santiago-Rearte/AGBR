"""Dev tool (inside the image): recall of the static analysis on synthetic mutants.

For each mutant: candidates from analysis.py, ground truth from running every test
except the slowest ones (listed as 'skipped heavy') through runner.py.  Prints any
failing test that is not a candidate.

  python /opt/scripts/mutants.py <n_mutants> <seed> [out.jsonl]
"""

import difflib
import json
import os
import pickle
import random
import re
import sys
import time

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import patchlib  # noqa: E402
import runner  # noqa: E402

REPO = "/opt/sqlglot"
HEAVY = 1.5  # seconds; tests slower than this are not run for ground truth

OPS = [
    (r" and ", " or "),
    (r" or ", " and "),
    (r" is not ", " is "),
    (r" is None", " is not None"),
    (r" == ", " != "),
    (r" != ", " == "),
    (r" < ", " <= "),
    (r" > ", " >= "),
    (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"),
    (r"\bnot ", ""),
    (r"\+ 1\b", "+ 2"),
    (r"- 1\b", "- 2"),
    (r'"([A-Z_]{3,})"', r'"\1X"'),
    (r"\breturn None\b", "return 0"),
    (r"\bbreak\b", "continue"),
]


def mutate_line(rng, rel, lines, i):
    """Mutates line i of rel; returns (label, patch, new_text) or None."""
    line = lines[i]
    kind = rng.random()
    if kind < 0.1:
        indent = re.match(r"\s*", line).group(0)
        if not line.strip():
            return None
        new = lines[:i] + [indent + "# note: keep in sync"] + lines[i:]
        label = "comment"
    elif kind < 0.17:
        m = re.match(r"(\s*)([A-Za-z_][\w.]*\s*=\s*.+)$", line)
        if not m or line.rstrip().endswith(("(", "[", "{", ",", "\\")):
            return None
        new = lines[:i] + [m.group(1) + "pass"] + lines[i + 1 :]
        label = "delete-assign"
    else:
        cands = [(p, r) for p, r in OPS if re.search(p, line)]
        if not cands or line.strip().startswith("#"):
            return None
        p, r = rng.choice(cands)
        ms = list(re.finditer(p, line))
        m = rng.choice(ms)
        repl = m.expand(r) if "\\" in r else r
        new_line = line[: m.start()] + repl + line[m.end() :]
        new = lines[:i] + [new_line] + lines[i + 1 :]
        label = f"op:{p}"
    old_text = "\n".join(lines)
    new_text = "\n".join(new)
    try:
        compile(new_text, rel, "exec")
    except SyntaxError:
        return None
    diff = difflib.unified_diff(
        old_text.split("\n"), new_text.split("\n"), f"a/{rel}", f"b/{rel}", lineterm="", n=3
    )
    patch = f"diff --git a/{rel} b/{rel}\n" + "\n".join(diff) + "\n"
    return label, patch.encode(), new_text.encode()


def mutate(rng, files):
    while True:
        rel = rng.choice(files)
        lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        i = rng.randrange(len(lines))
        m = mutate_line(rng, rel, lines, i)
        if m is not None:
            return (rel, i + 1) + m


def main():
    n = int(sys.argv[1])
    seed = int(sys.argv[2])
    out_path = sys.argv[3] if len(sys.argv) > 3 else None
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    tests = maps["tests"]
    dur = dict(zip(tests, maps["dur"]))
    light = [t for t in tests if dur.get(t, 0) <= HEAVY and maps["outcomes"].get(t) == "passed"]
    heavy = sorted(set(tests) - set(light))
    print(f"ground truth over {len(light)} tests; skipped heavy: {heavy}", flush=True)
    # Only mutate lines some test executes, otherwise nothing can fail.
    files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
    an = analysis.Analyzer(maps, REPO)
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

    t0 = time.monotonic()
    base_fail = ground_truth(REPO)
    print(f"base run: {len(base_fail)} failing in this environment ({time.monotonic() - t0:.0f}s): {sorted(base_fail)[:5]}", flush=True)
    summary = []
    for k in range(n):
        rel, ln, label, patch, new_text = mutate(rng, files)
        fps = patchlib.parse(patch)
        bits, notes, static = an.analyze(fps)
        cands = {tests[i] for i in analysis.iter_bits(bits)}
        root = f"/tmp/mut{k}"
        runner.make_overlay(root, REPO, {rel: new_text})
        t0 = time.monotonic()
        failing = ground_truth(root) - base_fail
        missing = sorted(failing - cands)
        rec = {
            "k": k,
            "file": rel,
            "line": ln,
            "label": label,
            "cands": len(cands),
            "failing": len(failing),
            "missing": missing[:10],
            "n_missing": len(missing),
            "notes": notes,
            "secs": round(time.monotonic() - t0, 1),
        }
        summary.append(rec)
        print(json.dumps(rec), flush=True)
        if out_path:
            with open(out_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
        os.system(f"rm -rf {root}")
    bad = [r for r in summary if r["n_missing"]]
    print(f"mutants={len(summary)} with_failures={sum(1 for r in summary if r['failing'])} recall_misses={len(bad)}")


if __name__ == "__main__":
    main()
