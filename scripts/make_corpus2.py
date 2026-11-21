"""Dev tool (inside the image): a labeled corpus that imitates the sample's commit
generator, with ground truth that includes the heavy tests.

Commits: single-token edits on lines some test executes (operators, booleans,
`is`/`==`, comparisons, small integers, argument indexes) and junk tokens inserted
into a line of an optimizer fixture.  Ground truth: every light test, plus every heavy
test (>= HEAVY seconds) the static analysis lists as a candidate (a heavy test outside
the candidates cannot be selected by any rule, so its outcome does not change costs).

  python /opt/scripts/make_corpus2.py <n> <seed> > corpus.jsonl
"""

import difflib
import json
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
HEAVY = 1.5
OPS = [
    (r" is not ", " is "),
    (r" is (?!not)", " == "),
    (r" and ", " or "),
    (r" or ", " and "),
    (r" == ", " != "),
    (r" != ", " == "),
    (r" < ", " <= "),
    (r" <= ", " < "),
    (r" > ", " >= "),
    (r" >= ", " > "),
    (r"\bTrue\b", "False"),
    (r"\bFalse\b", "True"),
    (r"\bnot ", ""),
    (r"\bif ([a-z_][\w.]*):", r"if not \1:"),
    (r"(?<![\w.])(\d)(?![\w.])", None),  # small integer +- 1
    (r"seq_get\(args, (\d)\)", None),
]
JUNK = ["42", "xa", "NULL", "0"]


def code_edit(rng, maps, files):
    while True:
        rel = rng.choice(files)
        covered = sorted((maps["lines"].get(rel) or {}).keys())
        if not covered:
            continue
        lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
        i = rng.choice(covered) - 1
        if i >= len(lines) or lines[i].strip().startswith("#"):
            continue
        line = lines[i]
        cands = [(p, r) for p, r in OPS if re.search(p, line)]
        if not cands:
            continue
        p, r = rng.choice(cands)
        m = rng.choice(list(re.finditer(p, line)))
        if r is None:
            if p.startswith("seq_get"):
                d = int(m.group(1))
                repl = m.group(0).replace(str(d), str(d + 1 if d == 0 else d - 1))
            else:
                d = int(m.group(1))
                repl = str(d + 1 if rng.random() < 0.5 or d == 0 else d - 1)
        else:
            repl = m.expand(r) if "\\" in r else r
        new = lines[:]
        new[i] = line[: m.start()] + repl + line[m.end() :]
        try:
            compile("\n".join(new), rel, "exec")
        except SyntaxError:
            continue
        return rel, i + 1, f"op:{p}", lines, new


def fixture_edit(rng, fixtures):
    rel = rng.choice(fixtures)
    lines = open(os.path.join(REPO, rel), encoding="utf-8").read().split("\n")
    i = rng.choice([k for k, l in enumerate(lines) if l.strip()])
    words = lines[i].split(" ")
    j = rng.randrange(len(words) + 1)
    new = lines[:]
    new[i] = " ".join(words[:j] + [rng.choice(JUNK)] + words[j:])
    return rel, i + 1, "fixture", lines, new


def main():
    n, seed = int(sys.argv[1]), int(sys.argv[2])
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    tests = maps["tests"]
    sec = {t["id"]: t["seconds"] for t in json.load(open("/tests.json"))["tests"]}
    ok = {t for t in tests if maps["outcomes"].get(t) == "passed"}
    light = [t for t in tests if sec.get(t, 0) < HEAVY and t in ok]
    heavy = {t for t in tests if sec.get(t, 0) >= HEAVY and t in ok}
    files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
    fixtures = sorted(
        f for f in maps["reads"] if f.startswith("tests/fixtures/optimizer/") and f.endswith(".sql")
    )
    an = analysis.Analyzer(maps, REPO)
    runner.warm_imports(["duckdb", "pandas", "numpy"])
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")

    def run(root, ids, par):
        if not ids:
            return set()
        chunks = [ids[j::par] for j in range(par)]
        jobs = [runner.Job(j, {"root": root, "nodeids": c, "default_timeout": 900.0, "import_check": []}, 7200)
                for j, c in enumerate(chunks) if c]
        runner.run_jobs(jobs, parallel=par)
        return set(ids) - set().union(*(j.passed for j in jobs))

    base_fail = run(REPO, light, 2)
    sys.stderr.write(f"base failing: {len(base_fail)}\n")
    for k in range(n):
        if rng.random() < 0.3:
            rel, ln, label, old, new = fixture_edit(rng, fixtures)
        else:
            rel, ln, label, old, new = code_edit(rng, maps, files)
        diff = difflib.unified_diff(old, new, f"a/{rel}", f"b/{rel}", lineterm="", n=3)
        patch = f"diff --git a/{rel} b/{rel}\n" + "\n".join(diff) + "\n"
        bits, _notes, _static = an.analyze(patchlib.parse(patch.encode()))
        hv = sorted(heavy & {tests[i] for i in analysis.iter_bits(bits)})
        root = f"/tmp/c2_{k}"
        runner.make_overlay(root, REPO, {rel: "\n".join(new).encode()})
        failing = (run(root, light, 2) - base_fail) | run(root, hv, 1)
        os.system(f"rm -rf {root}")
        print(json.dumps({"id": f"o{seed}_{k:03d}", "file": rel, "line": ln, "label": label,
                          "patch": patch, "failing": sorted(failing), "heavy_run": hv}), flush=True)
        sys.stderr.write(f"{k} {rel}:{ln} {label} failing={len(failing)} heavy_run={len(hv)}\n")


if __name__ == "__main__":
    main()
