"""Merges the raw traces of the traced pytest run into compact maps for the selector.

Usage: build_maps.py <raw_dir> <out_dir>   (cwd = sqlglot checkout)

maps.pkl holds, with every test set encoded as an int bitset over test indices:
  tests      nodeids in suite order          dur   seconds measured in the traced run
  lines      relpath -> {line: set_id}       sets  list of bitsets (deduplicated)
  reads      relpath -> bitset               dirs  relpath -> bitset
  inst       "module:qualname" -> bitset     funcs "file:qualname" -> bitset
  family     "module:qualname" -> sorted descendant keys (itself included)
  by_file    test file -> bitset             by_class "file::Class" -> bitset
  outcomes   nodeid -> base outcome in the traced run
"""

import glob
import importlib
import inspect
import os
import pickle
import pkgutil
import sys
from collections import defaultdict


def main(raw_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    repo = os.getcwd()

    contexts = defaultdict(lambda: [set(), set(), set(), set(), set()])
    durations, outcomes = {}, {}
    file_index, files = {}, []

    def fidx(rel):
        i = file_index.get(rel)
        if i is None:
            i = file_index[rel] = len(files)
            files.append(rel)
        return i

    dumps = sorted(glob.glob(os.path.join(raw_dir, "*.pkl")))
    for path in dumps:
        remap = []
        with open(path, "rb") as fh:
            while True:
                try:
                    data = pickle.load(fh)
                except EOFError:
                    break
                except Exception as error:
                    print(f"  truncated dump {path}: {error}", file=sys.stderr)
                    break
                assert data["files_base"] == len(remap), path
                for fn in data["new_files"]:
                    rel = os.path.relpath(fn, repo).replace(os.sep, "/")
                    remap.append(fidx(rel))
                for name, (lines, reads, dirs, funcs, inst) in data["ctx"].items():
                    rec = contexts[name]
                    rec[0].update(remap[k // 1_000_000] * 1_000_000 + k % 1_000_000 for k in lines)
                    rec[1].update(reads)
                    rec[2].update(dirs)
                    rec[3].update(funcs)
                    rec[4].update(inst)
                durations.update(data.get("durations", {}))
                outcomes.update(data.get("outcomes", {}))

    nodeids = set(durations) | set(outcomes)
    for name in contexts:
        if "|" in name and not name.startswith("collect|"):
            nodeids.add(name.rsplit("|", 1)[0])
    tests = sorted(nodeids)
    index = {t: i for i, t in enumerate(tests)}
    print(f"dumps={len(dumps)} contexts={len(contexts)} tests={len(tests)}", file=sys.stderr)

    def tfile(t):
        return t.split("::", 1)[0]

    def tclass(t):
        return t.rsplit("::", 1)[0]

    by_file, by_class = defaultdict(int), defaultdict(int)
    for t, i in index.items():
        by_file[tfile(t)] |= 1 << i
        by_class[tclass(t)] |= 1 << i

    # Group-level contexts: class setup/teardown phases and module collection.
    group_ctx = defaultdict(lambda: [set(), set(), set(), set(), set()])
    per_test = defaultdict(lambda: [set(), set(), set(), set(), set()])
    for name, rec in contexts.items():
        if name.startswith("collect|"):
            target = ("file", name[len("collect|") :])
            dest = group_ctx[target]
        elif "|" in name:
            nodeid, phase = name.rsplit("|", 1)
            if nodeid not in index:
                continue
            dest = per_test[nodeid]
            for k in range(5):
                dest[k] |= rec[k]
            if phase in ("setup", "teardown"):
                dest = group_ctx[("class", tclass(nodeid))]
            else:
                continue
        else:
            continue
        for k in range(5):
            dest[k] |= rec[k]

    for (kind, key), rec in group_ctx.items():
        members = [t for t in tests if (tfile(t) if kind == "file" else tclass(t)) == key]
        for t in members:
            dest = per_test[t]
            for k in range(5):
                dest[k] |= rec[k]

    # Inverted indexes.
    line_tests = defaultdict(list)
    reads, dirs, funcs, inst = (defaultdict(list) for _ in range(4))
    for t, rec in per_test.items():
        i = index[t]
        for key in rec[0]:
            line_tests[key].append(i)
        for r in rec[1]:
            reads[r.replace(os.sep, "/")].append(i)
        for d in rec[2]:
            dirs[d.replace(os.sep, "/").rstrip("/")].append(i)
        for f in rec[3]:
            funcs[f].append(i)
        for c in rec[4]:
            inst[c].append(i)

    def bits(indices):
        value = 0
        for i in indices:
            value |= 1 << i
        return value

    set_ids, sets = {}, []
    lines = defaultdict(dict)
    for key, indices in line_tests.items():
        tup = tuple(sorted(indices))
        sid = set_ids.get(tup)
        if sid is None:
            sid = set_ids[tup] = len(sets)
            sets.append(bits(tup))
        lines[files[key // 1_000_000]][key % 1_000_000] = sid

    maps = {
        "tests": tests,
        "dur": [durations.get(t, 0.0) for t in tests],
        "lines": dict(lines),
        "sets": sets,
        "reads": {k: bits(v) for k, v in reads.items()},
        "dirs": {k: bits(v) for k, v in dirs.items()},
        "funcs": {k: bits(v) for k, v in funcs.items()},
        "inst": {k: bits(v) for k, v in inst.items()},
        "by_file": dict(by_file),
        "by_class": dict(by_class),
        "outcomes": outcomes,
        "family": class_families(),
    }
    maps["near"] = coverage_neighbours(tests, index, per_test, files, durations)
    maps["import_lines"] = import_lines(os.path.join(raw_dir, "import.trace"), repo)
    maps["mod_files"] = module_files(repo)
    mod_classes = defaultdict(list)
    for key in maps["family"]:
        mod_classes[key.split(":", 1)[0]].append(key)
    maps["mod_classes"] = dict(mod_classes)
    maps["eager"] = eager_modules()
    # Modules with lazy attributes (PEP 562 __getattr__): the names they export, so the
    # run-time import check can exercise every one of them (lazy imports are cached
    # per process, so coverage only credits the first test that triggers them).
    maps["lazy_all"] = {
        name: list(getattr(module, "__all__", ()))
        for name, module in list(sys.modules.items())
        if (name == "sqlglot" or name.startswith("sqlglot.")) and "__getattr__" in vars(module)
    }
    print(f"  lazy modules: {sorted(maps['lazy_all'])}", file=sys.stderr)
    with open(os.path.join(out_dir, "maps.pkl"), "wb") as fh:
        pickle.dump(maps, fh, protocol=pickle.HIGHEST_PROTOCOL)
    bad = {t: o for t, o in outcomes.items() if o != "passed"}
    print(
        f"files_with_lines={len(lines)} line_keys={len(line_tests)} unique_sets={len(sets)} "
        f"reads={len(reads)} inst={len(inst)} not_passed={len(bad)}",
        file=sys.stderr,
    )
    for t, o in sorted(bad.items())[:20]:
        print(f"  base {o}: {t}", file=sys.stderr)


def coverage_neighbours(tests, index, per_test, files, durations, slow=0.8, keep=60):
    """For every slow test, the fast tests that run most of the same sqlglot lines.

    Used at run time as probes: a change that breaks a slow test is expected to
    break some of its closest fast neighbours as well."""
    sqlglot_files = {i for i, f in enumerate(files) if f.startswith("sqlglot/")}

    def lines_of(t):
        rec = per_test.get(t)
        if rec is None:
            return set()
        return {k for k in rec[0] if k // 1_000_000 in sqlglot_files}

    slow_tests = [t for t in tests if durations.get(t, 0.0) >= slow]
    fast_tests = [t for t in tests if durations.get(t, 0.0) < slow]
    fast_lines = {t: lines_of(t) for t in fast_tests}
    near = {}
    for h in slow_tests:
        hl = lines_of(h)
        if not hl:
            continue
        scored = []
        for c, cl in fast_lines.items():
            if not cl:
                continue
            inter = len(hl & cl)
            if inter:
                scored.append((-inter, durations.get(c, 0.0), c))
        scored.sort()
        near[index[h]] = [index[c] for _, _, c in scored[:keep]]
    print(f"  neighbours for {len(near)} slow tests", file=sys.stderr)
    return near


def class_families():
    """Maps every sqlglot class to the keys of itself and all its subclasses."""
    import sqlglot

    import tests

    for pkg, prefix in ((sqlglot, "sqlglot."), (tests, "tests.")):
        for info in pkgutil.walk_packages(pkg.__path__, prefix=prefix):
            if info.name in ("sqlglot.__main__",) or info.name.startswith("tests.sqlglot"):
                continue
            try:
                importlib.import_module(info.name)
            except BaseException as error:
                print(f"  import {info.name} failed: {error!r}", file=sys.stderr)

    def key(cls):
        return f"{cls.__module__}:{cls.__qualname__}"

    classes = set()
    for name, module in list(sys.modules.items()):
        if not (name in ("sqlglot", "tests") or name.startswith(("sqlglot.", "tests."))):
            continue
        for _, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__.startswith(("sqlglot", "tests")):
                classes.add(obj)
                stack = [obj]
                while stack:
                    c = stack.pop()
                    for attr in vars(c).values():
                        if (
                            inspect.isclass(attr)
                            and attr not in classes
                            and attr.__module__.startswith(("sqlglot", "tests"))
                        ):
                            classes.add(attr)
                            stack.append(attr)

    family = {}
    for cls in classes:
        seen, stack = set(), [cls]
        while stack:
            c = stack.pop()
            try:
                subs = type.__subclasses__(c)
            except TypeError:
                subs = []
            for s in subs:
                if s not in seen:
                    seen.add(s)
                    stack.append(s)
        seen.add(cls)
        family[key(cls)] = sorted(key(c) for c in seen)
    return family


def import_lines(path, repo):
    """file -> {line: modules whose import executed that line}."""
    out = defaultdict(lambda: defaultdict(set))
    if not os.path.exists(path):
        print("  no import trace", file=sys.stderr)
        return {}
    remap = []
    with open(path, "rb") as fh:
        while True:
            try:
                data = pickle.load(fh)
            except EOFError:
                break
            for fn in data["new_files"]:
                remap.append(os.path.relpath(fn, repo).replace(os.sep, "/"))
            for name, rec in data["ctx"].items():
                if not name.startswith("import|"):
                    continue
                mod = name[len("import|") :]
                if mod == "<root>":
                    continue
                for k in rec[0]:
                    out[remap[k // 1_000_000]][k % 1_000_000].add(mod)
    result = {f: {ln: tuple(sorted(ms)) for ln, ms in d.items()} for f, d in out.items()}
    print(f"  import trace: {len(result)} files, {sum(len(d) for d in result.values())} lines", file=sys.stderr)
    return result


def module_files(repo):
    out = {}
    for top in ("sqlglot", "tests", "sqlglotc"):
        for root, _dirs, names in os.walk(os.path.join(repo, top)):
            for name in names:
                if name.endswith(".py"):
                    rel = os.path.relpath(os.path.join(root, name), repo).replace(os.sep, "/")
                    parts = rel[:-3].split("/")
                    if parts[-1] == "__init__":
                        parts = parts[:-1]
                    out[".".join(parts)] = rel
    return out


def eager_modules():
    import subprocess

    code = "import sqlglot, sys; print(' '.join(m for m in sys.modules if m.startswith('sqlglot')))"
    res = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    return res.stdout.split()


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
