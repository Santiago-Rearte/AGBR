"""Blast Radius selector: one JSON line per commit on stdout, logs on stderr.

1. Static analysis (analysis.py) bounds the tests each patch could affect.
2. Cheap candidates are run against the patched tree (runner.py); the ones
   that pass are dropped.  Candidates that are not run stay selected.
3. The selection is packed onto 1..8 workers to minimise max load + 2 s/worker.
"""

import json
import os
import pickle
import shutil
import sys
import time
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

REPO = os.environ.get("BR_REPO", "/opt/sqlglot")
MAPS = os.environ.get("BR_MAPS", "/opt/maps")
DATA = os.environ.get("BR_DATA", "/data")
WORK = os.environ.get("BR_WORK", "/tmp/br")
WORKER_MS = 2000
MAX_WORKERS = 8

# Cost model for deciding which candidates to verify (seconds).
RUN_OVERHEAD = float(os.environ.get("BR_RUN_OVERHEAD", "0.3"))
FILE_OVERHEAD = float(os.environ.get("BR_FILE_OVERHEAD", "0.12"))
POOL_OVERHEAD = float(os.environ.get("BR_POOL_OVERHEAD", "0.3"))
CPU_FACTOR = float(os.environ.get("BR_CPU_FACTOR", "1.15"))
FAIL_PRIOR = float(os.environ.get("BR_FAIL_PRIOR", "0.5"))
VERIFY = os.environ.get("BR_VERIFY", "1") == "1"
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def pack(nums, dur_ms):
    """Longest-first greedy for k = 1..8 workers; returns (cost_ms, workers)."""
    nums = sorted(set(nums), key=lambda n: (-dur_ms[n], n))
    if not nums:
        return 0, []
    total = sum(dur_ms[n] for n in nums)
    best = None
    for k in range(1, min(MAX_WORKERS, len(nums)) + 1):
        loads = [0] * k
        workers = [[] for _ in range(k)]
        for n in nums:
            i = loads.index(min(loads))
            loads[i] += dur_ms[n]
            workers[i].append(n)
        cost = max(loads) + WORKER_MS * k
        if best is None or cost < best[0]:
            best = (cost, [w for w in workers if w])
        if max(dur_ms[nums[0]], total / (k + 1)) + WORKER_MS * (k + 1) > best[0]:
            break
    return best


def cpu_seconds():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


class Plan:
    def __init__(self, commit):
        self.commit = commit
        self.static = set()  # selected without running
        self.run = set()  # to verify
        self.extra = set()  # added after verification (collection/import failures)
        self.files = {}  # rel -> new content, for the overlay
        self.deleted = set()
        self.modules = []  # patched sqlglot modules to import-check
        self.job = None
        self.notes = []


def main():
    t_start = time.monotonic()
    with open(os.path.join(DATA, "tests.json"), "rb") as fh:
        tests_json = json.loads(fh.read().decode("utf-8"))
    tests = tests_json["tests"]
    skipped = set(tests_json.get("skipped") or [])
    dur_ms = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
    sec = {t["number"]: t["seconds"] for t in tests}
    all_nums = sorted(t["number"] for t in tests if t["id"] not in skipped)
    commits = sorted(
        name[: -len(".patch")] for name in os.listdir(os.path.join(DATA, "commits")) if name.endswith(".patch")
    )

    results = {}
    try:
        results = select(commits, tests, skipped, dur_ms, sec, all_nums)
    except BaseException:  # noqa: BLE001 - the output must stay valid
        log("selector failed, falling back to all tests:\n" + traceback.format_exc())

    lines = []
    for c in commits:
        nums = results.get(c)
        if nums is None:
            nums = all_nums
        _, workers = pack(nums, dur_ms)
        lines.append(json.dumps({"commit": c, "workers": workers}, separators=(",", ":")))
    out = ("\n".join(lines) + "\n") if lines else ""
    sys.stdout.write(out)
    sys.stdout.flush()
    log(f"done in {time.monotonic() - t_start:.1f}s wall, {cpu_seconds():.2f}s cpu (self+children)")
    try:
        with open("/sys/fs/cgroup/cpu.stat") as fh:
            log("cgroup " + fh.read().split("\n", 1)[0])
    except OSError:
        pass


def select(commits, tests, skipped, dur_ms, sec, all_nums):
    import analysis
    import patchlib
    import runner

    with open(os.path.join(MAPS, "maps.pkl"), "rb") as fh:
        maps = pickle.load(fh)
    verifiable = set()
    try:
        with open(os.path.join(MAPS, "verifiable.json")) as fh:
            verifiable = set(json.load(fh)["passed"])
    except OSError:
        log("no verifiable.json: nothing will be verified")

    id2num = {t["id"]: t["number"] for t in tests if t["id"] not in skipped}
    num2id = {n: i for i, n in id2num.items()}
    idx2num = [id2num.get(t, -1) for t in maps["tests"]]
    known = {n for n in idx2num if n >= 0}
    unknown = sorted(set(id2num.values()) - known)
    if unknown:
        log(f"{len(unknown)} tests have no trace data; they are added to every code change")
    pool_bits = 0
    for name, b in maps["funcs"].items():
        if "ProcessPoolExecutor" in name or "Popen" in name:
            pool_bits |= b
    pool_nums = {idx2num[i] for i in analysis.iter_bits(pool_bits) if idx2num[i] >= 0}
    file_of = {t["number"]: t["file"] for t in tests}

    an = analysis.Analyzer(maps, REPO)
    plans = []
    for c in commits:
        plan = Plan(c)
        plans.append(plan)
        t0 = time.monotonic()
        try:
            with open(os.path.join(DATA, "commits", c + ".patch"), "rb") as fh:
                fps = patchlib.parse(fh.read())
            if not fps:
                plan.notes.append("empty")
                continue
            bits, notes, static = an.analyze(fps)
            plan.notes.extend(notes)
            cands = {idx2num[i] for i in analysis.iter_bits(bits) if idx2num[i] >= 0}
            if unknown and any(fp.path.endswith(".py") for fp in fps):
                static = True if len(unknown) > 50 else static
                plan.static |= set(unknown)
            prepare_overlay(plan, fps, an, patchlib)
            if plan.files is None:
                static = True
            decide(plan, cands, static, verifiable, num2id, sec, pool_nums, file_of)
        except Exception:
            log(f"{c}: analysis crashed, selecting everything\n{traceback.format_exc()}")
            plan.static = set(all_nums)
            plan.run = set()
        log(
            f"{c}: cand={len(plan.static) + len(plan.run)} run={len(plan.run)} static={len(plan.static)}"
            f" ({time.monotonic() - t0:.2f}s) {' '.join(plan.notes)[:200]}"
        )

    jobs = [p for p in plans if p.run]
    if jobs and VERIFY:
        verify(jobs, runner, an, idx2num, id2num, num2id, sec)

    results = {}
    for p in plans:
        nums = set(p.static) | set(p.extra)
        if p.run:
            passed = p.job.passed if p.job is not None else set()
            for n in p.run:
                if num2id[n] not in passed:
                    nums.add(n)
        results[p.commit] = sorted(n for n in nums if n in dur_ms)
        cost, workers = pack(results[p.commit], dur_ms)
        log(f"{p.commit}: selected={len(results[p.commit])} cost={cost / 1000:.3f}s workers={len(workers)}")
    return results


def prepare_overlay(plan, fps, an, patchlib):
    files, deleted, modules = {}, set(), []
    for fp in fps:
        if fp.binary:
            plan.files = None
            return
        if fp.mode_only:
            continue
        if fp.old_path and (fp.deleted or (fp.new_path and fp.new_path != fp.old_path)):
            deleted.add(fp.old_path)
        if fp.new_path:
            base = an.source(fp.old_path) if fp.old_path else b""
            files[fp.new_path] = patchlib.apply(base, fp)
            mod = analysis_modname(fp.new_path)
            if mod and mod.startswith("sqlglot"):
                modules.append(mod)
    plan.files, plan.deleted, plan.modules = files, deleted, modules


def analysis_modname(rel):
    import analysis

    return analysis.modname(rel)


def decide(plan, cands, static, verifiable, num2id, sec, pool_nums, file_of):
    """Splits candidates into verified-by-running and selected-without-running."""
    from_fixed = set(plan.static)
    if static or not VERIFY:
        plan.static = from_fixed | cands
        return
    runnable = sorted((n for n in cands if num2id.get(n) in verifiable), key=lambda n: (sec[n], n))
    never = cands - set(runnable)
    dur_ms = {n: int(sec[n] * 1000 + 0.5) for n in cands | from_fixed}
    if os.environ.get("BR_FORCE_VERIFY") == "1":  # dev: check verification fidelity
        plan.run = set(runnable)
        plan.static = from_fixed | never
        return

    best_cost = pack(cands | from_fixed, dur_ms)[0] / 1000.0
    best_j = 0
    run_sum = 0.0
    files = set()
    uses_pool = False
    for j, n in enumerate(runnable, start=1):
        run_sum += sec[n]
        files.add(file_of[n])
        uses_pool = uses_pool or n in pool_nums
        rest = never | set(runnable[j:]) | from_fixed
        cost = RUN_OVERHEAD + FILE_OVERHEAD * len(files) + CPU_FACTOR * run_sum
        cost += POOL_OVERHEAD if uses_pool else 0.0
        if rest:
            cost += pack(rest, dur_ms)[0] / 1000.0
        else:
            cost += FAIL_PRIOR * (WORKER_MS / 1000.0 + 0.05)
        if cost < best_cost:
            best_cost, best_j = cost, j
    plan.run = set(runnable[:best_j])
    plan.static = from_fixed | (cands - plan.run)
    plan.notes.append(f"est={best_cost:.2f}s")


def verify(plans, runner, an, idx2num, id2num, num2id, sec):
    os.environ.setdefault("PYTHON_CPU_COUNT", "2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("SKIP_INTEGRATION", "0")
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    need = {}
    for p in plans:
        for n in p.run:
            f = num2id[n].split("::", 1)[0]
            need[f] = need.get(f, 0) + 1
    extra = third_party_imports(an, [f for f, k in need.items()], plans)
    t0 = cpu_seconds()
    runner.warm_imports(extra)
    log(f"warm imports {sorted(extra)}: {cpu_seconds() - t0:.2f}s cpu")

    jobs = []
    for p in plans:
        root = os.path.join(WORK, p.commit)
        nodeids = sorted((num2id[n] for n in p.run), key=lambda i: i)
        timeouts = {num2id[n]: max(3.0, 6.0 * sec[n] + 2.0) for n in p.run}
        budget = 20.0 + sum(timeouts.values())
        spec = {"root": root, "nodeids": nodeids, "timeouts": timeouts, "import_check": p.modules}
        job = runner.Job(p.commit, spec, budget)
        job.plan = p
        p.job = job
        jobs.append(job)

    def setup(job):
        p = job.plan
        root = job.spec["root"]
        shutil.rmtree(root, ignore_errors=True)
        runner.make_overlay(root, REPO, p.files, p.deleted)

    def cleanup(job):
        p = job.plan
        shutil.rmtree(job.spec["root"], ignore_errors=True)
        log(
            f"{p.commit}: verified {len(job.passed)}/{len(p.run)} pass, cpu={job.cpu:.2f}s wall={job.wall:.2f}s"
            f" done={job.done} crash={job.crashed}"
        )
        for mod in job.import_failed:
            rel = mod.replace(".", "/")
            rel = rel + ".py" if os.path.exists(os.path.join(REPO, rel + ".py")) else rel + "/__init__.py"
            bits = an.broken_module(rel)
            p.extra |= {idx2num[i] for i in __import__("analysis").iter_bits(bits) if idx2num[i] >= 0}
        for nodeid in job.collect_failed:
            f = nodeid.split("::", 1)[0]
            p.extra |= {n for i, n in id2num.items() if i.split("::", 1)[0] == f}

    runner.run_jobs(jobs, parallel=PARALLEL, setup=setup, cleanup=cleanup)


def third_party_imports(an, test_files, plans):
    """Third-party modules imported at the top of the test files about to run."""
    import ast

    std = set(getattr(sys, "stdlib_module_names", ()))
    mods = set()
    for f in test_files:
        try:
            tree = an.tree(f)
        except Exception:
            continue
        for node in tree.body:
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                names = [node.module.split(".")[0]]
            for m in names:
                if m not in std and m not in ("sqlglot", "tests"):
                    mods.add(m)
    return sorted(mods)


if __name__ == "__main__":
    main()
