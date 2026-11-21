"""Dev tool (inside the image): would an import-time state diff narrow class-level changes?

For class-level mutants: imports base and patched sqlglot (every module) in forked
children, fingerprints the non-callable attributes of every sqlglot class, and reports
which classes really changed.  Compares the users of those classes with the current
candidate set (count and packed cost).

  python /opt/scripts/state_diff_eval.py <n> <seed> <tests.json>
"""

import importlib
import json
import os
import pickle
import pkgutil
import random
import re
import sys

sys.path.insert(0, "/opt/br")
sys.path.insert(0, "/opt/scripts")
import analysis  # noqa: E402
import mutants  # noqa: E402
import patchlib  # noqa: E402
import runner  # noqa: E402
import selector  # noqa: E402

REPO = "/opt/sqlglot"
ADDR = re.compile(r"0x[0-9a-fA-F]+")


def fingerprint(root):
    r, w = os.pipe()
    pid = os.fork()
    if pid == 0:
        os.close(r)
        try:
            os.chdir(root)
            sys.path.insert(0, root)
            import sqlglot

            for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
                if info.name != "sqlglot.__main__":
                    try:
                        importlib.import_module(info.name)
                    except BaseException:
                        pass
            out = {}
            for name, mod in list(sys.modules.items()):
                if not (name == "sqlglot" or name.startswith("sqlglot.")):
                    continue
                stack = [c for c in vars(mod).values() if isinstance(c, type) and c.__module__ == name]
                seen = set()
                while stack:
                    c = stack.pop()
                    if c in seen:
                        continue
                    seen.add(c)
                    parts = []
                    for k, v in sorted(vars(c).items()):
                        if isinstance(v, type):
                            if v.__module__.startswith("sqlglot"):
                                stack.append(v)
                            parts.append((k, v.__qualname__))
                            continue
                        if callable(v) or isinstance(v, (staticmethod, classmethod, property)):
                            continue
                        try:
                            parts.append((k, ADDR.sub("0x", repr(v))))
                        except Exception:
                            parts.append((k, "?"))
                    out[f"{c.__module__}:{c.__qualname__}"] = hash(tuple(parts))
            data = pickle.dumps(out)
        except BaseException as error:  # noqa: BLE001
            data = pickle.dumps({"__error__": repr(error)})
        os.write(w, data)
        os._exit(0)
    os.close(w)
    chunks = []
    while True:
        b = os.read(r, 1 << 20)
        if not b:
            break
        chunks.append(b)
    os.waitpid(pid, 0)
    return pickle.loads(b"".join(chunks))


def main():
    n, seed, tj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    rng = random.Random(seed)
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    T = maps["tests"]
    sec = {t["id"]: t["seconds"] for t in json.load(open(tj))["tests"]}
    dur = {i: int(sec.get(t, 0) * 1000 + 0.5) for i, t in enumerate(T)}
    files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
    an = analysis.Analyzer(maps, REPO)
    base = fingerprint(REPO)
    base2 = fingerprint(REPO)
    unstable = {c for c in set(base) | set(base2) if base.get(c) != base2.get(c)}
    print(f"base fingerprint: {len(base)} classes, {len(unstable)} unstable: {sorted(unstable)[:8]}", flush=True)
    tot_now = tot_diff = 0.0
    k = 0
    tries = 0
    while k < n and tries < n * 30:
        tries += 1
        rel, ln, label, patch, new_text = mutants.mutate(rng, files)
        bits, notes, static = an.analyze(patchlib.parse(patch))
        if static or not an.trace.get("import") or an.trace.get("nonsite"):
            continue
        k += 1
        root = f"/tmp/sd{k}"
        runner.make_overlay(root, REPO, {rel: new_text})
        fp = fingerprint(root)
        os.system(f"rm -rf {root}")
        if "__error__" in fp:
            print(json.dumps({"file": rel, "line": ln, "import_error": fp["__error__"][:80]}), flush=True)
            continue
        changed = sorted(c for c in set(base) | set(fp) if base.get(c) != fp.get(c) and c not in unstable)
        users = 0
        for c in changed:
            users |= an.family_bits(c) | an.holder_bits_for_key(c) if hasattr(an, "holder_bits_for_key") else an.family_bits(c)
        users |= an.reads(rel)
        now = selector.pack(list(analysis.iter_bits(bits)), dur)[0] / 1000
        diff = selector.pack(list(analysis.iter_bits(users)), dur)[0] / 1000
        tot_now += now
        tot_diff += diff
        print(
            json.dumps(
                {
                    "file": rel,
                    "line": ln,
                    "changed": changed[:6],
                    "n_changed": len(changed),
                    "cands_now": bin(bits).count("1"),
                    "cands_diff": bin(users).count("1"),
                    "cost_now": round(now, 2),
                    "cost_diff": round(diff, 2),
                }
            ),
            flush=True,
        )
    print(json.dumps({"mutants": k, "cost_now": round(tot_now, 1), "cost_state_diff": round(tot_diff, 1)}), flush=True)


if __name__ == "__main__":
    main()
