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
DATA_FAIL_PRIOR = float(os.environ.get("BR_DATA_FAIL_PRIOR", "0.95"))
VERIFY = os.environ.get("BR_VERIFY", "1") == "1"
PARALLEL = int(os.environ.get("BR_PARALLEL", "2"))
CALIB = {}
LAZY_ALL = {}
HEAVY_SEC = float(os.environ.get("BR_HEAVY_SEC", "1.5"))
MIN_CHEAP_COVER = int(os.environ.get("BR_MIN_CHEAP_COVER", "3"))
DOMINANCE = os.environ.get("BR_DOMINANCE", "1") == "1"
DOMINANCE_IMPORT = os.environ.get("BR_DOMINANCE_IMPORT", "0") == "1"
MIN_CHEAP_IMPORT = int(os.environ.get("BR_MIN_CHEAP_IMPORT", "10"))
PROBES = os.environ.get("BR_PROBES", "1") == "1"
DIVERSE_PROBES = int(os.environ.get("BR_DIVERSE_PROBES", "15"))
PROBES_PER_HEAVY = int(os.environ.get("BR_PROBES_PER_HEAVY", "6"))
MIN_PROBES = int(os.environ.get("BR_MIN_PROBES", "3"))
UNIQUE_MAX = float(os.environ.get("BR_UNIQUE_MAX", "0.005"))
# High-variance rule: the static remainder of a class/module-level change is kept
# only if one of a few diverse cheap probes fails.
PROBE_DROP = os.environ.get("BR_PROBE_DROP", "0") == "1"
# Probe gating of heavy tests also for changes inside functions.
PROBES_RUNTIME = os.environ.get("BR_PROBES_RUNTIME", "0") == "1"
PROBE_DROP_RUNTIME = os.environ.get("BR_PROBE_DROP_RUNTIME", "0") == "1"
# A gated heavy test that meets the strict criteria (STRICT_UNIQUE, STRICT_LINE_RATE)
# is selected only when one of its own coverage neighbours fails, not whenever any
# verified test fails.
PER_HEAVY = os.environ.get("BR_PER_HEAVY", "0") == "1"
STRICT_UNIQUE = 0.005
STRICT_LINE_RATE = 100.0
PROBE_DROP_N = int(os.environ.get("BR_PROBE_DROP_N", "25"))
# Changes inside functions: probe-drop only when the static selection would cost at
# least this many simulated seconds (big selections, where the gain is large).
PROBE_DROP_RT_MIN = float(os.environ.get("BR_PROBE_DROP_RT_MIN", "0"))
PROBE_DROP_RT_N = int(os.environ.get("BR_PROBE_DROP_RT_N", "40"))
MIN_LINE_RATE = float(os.environ.get("BR_MIN_LINE_RATE", "100"))  # sqlglot lines per traced second


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
        self.fail_prior = FAIL_PRIOR
        self.dominated = set()
        self.conditional = set()
        self.cond_near = {}  # gated heavy test -> its own neighbours being run
        self.stop_on_fail = False
        self.check_imports = False
        self.keyref = []  # key-level refinements, each with its unrefined fallback


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
    LAZY_ALL.update(maps.get("lazy_all") or {})
    verifiable = set()
    try:
        with open(os.path.join(MAPS, "verifiable.json")) as fh:
            calib = json.load(fh)
        verifiable = set(calib["passed"])
        CALIB.update(calib)
        log(f"calibration: job overhead {CALIB.get('job_overhead', 0):.3f}s")
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
    num2idx = {n: i for i, n in enumerate(idx2num) if n >= 0}
    _NEAR.update(near=maps.get("near") or {}, idx2num=idx2num, num2idx=num2idx)
    probe_eligible(maps)
    idx_sec = [sec.get(n, 0.0) if n >= 0 else 0.0 for n in idx2num]
    cheap_all = 0
    for i, n in enumerate(idx2num):
        if n >= 0 and idx_sec[i] < HEAVY_SEC and maps["tests"][i] in verifiable:
            cheap_all |= 1 << i
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
            plan.keyref = list(an.trace.get("keyref") or [])
            if plan.keyref:
                plan.notes.append("keys(" + ",".join(f"{k['attr']}:{len(k['keys'])}" for k in plan.keyref) + ")")
            cands = {idx2num[i] for i in analysis.iter_bits(bits) if idx2num[i] >= 0}
            if not static and DOMINANCE:
                dom = dominated_heavy(an, bits, idx_sec, cheap_all)
                plan.dominated = {idx2num[i] for i in analysis.iter_bits(dom) if idx2num[i] >= 0}
            if unknown and any(fp.path.endswith(".py") for fp in fps):
                static = True if len(unknown) > 50 else static
                plan.static |= set(unknown)
            prepare_overlay(plan, fps, an, patchlib)
            if plan.files is None:
                static = True
            # A change to data a test reads almost always changes what it observes.
            data_change = any(not fp.path.endswith(".py") for fp in fps)
            plan.fail_prior = DATA_FAIL_PRIOR if data_change else FAIL_PRIOR
            decide(plan, cands, static, verifiable, num2id, sec, pool_nums, file_of)
            runtime_only = bool(an.trace.get("runtime")) and not an.trace.get("other")
            if PROBES and VERIFY and not static and not an.trace.get("nonsite") and (
                an.trace.get("import") or (PROBES_RUNTIME and runtime_only)
            ):
                add_probes(plan, cands, maps, idx2num, num2idx, num2id, verifiable, sec)
                if PROBE_DROP and not plan.keyref and (an.trace.get("import") or PROBE_DROP_RUNTIME):
                    # (key-level selections are exact: one specific test may be the
                    # only one to fail, and no handful of probes would see it)
                    focus = focus_tests(an, maps, idx2num) if not an.trace.get("import") else {}
                    if an.trace.get("import"):
                        probe_drop(plan, maps, num2idx, num2id, verifiable, sec)
                    elif pack(plan.static, dur_ms)[0] / 1000.0 >= PROBE_DROP_RT_MIN:
                        probe_drop(plan, maps, num2idx, num2id, verifiable, sec, focus, limit=PROBE_DROP_RT_N)
            # Changed code that runs at import time can break the import of every
            # user; such patches always get an import check, even when nothing is run.
            if plan.files is not None and (an.trace.get("import") or an.trace.get("import_mods")):
                plan.check_imports = True
                plan.modules = sorted(set(plan.modules) | {m for m in an.trace.get("import_mods", ()) if m.startswith("sqlglot")})
        except Exception:
            log(f"{c}: analysis crashed, selecting everything\n{traceback.format_exc()}")
            plan.static = set(all_nums)
            plan.run = set()
        log(
            f"{c}: cand={len(plan.static) + len(plan.run)} run={len(plan.run)} static={len(plan.static)}"
            f" ({time.monotonic() - t0:.2f}s) {' '.join(plan.notes)[:200]}"
        )

    jobs = [p for p in plans if p.run or (p.check_imports and p.modules)]
    if jobs and VERIFY:
        verify(jobs, runner, an, idx2num, id2num, num2id, sec)

    results = {}
    for p in plans:
        nums = set(p.static) | set(p.extra)
        if p.run:
            passed = p.job.passed if p.job is not None else set()
            live = False
            for n in p.run:
                if num2id[n] not in passed:
                    nums.add(n)
                    live = True
            job = p.job
            if job is None or not job.done or job.crashed or job.collect_failed or job.import_failed:
                live = True
            if p.conditional and live:
                ok = PER_HEAVY and job is not None and job.done and not job.crashed
                ok = ok and not job.collect_failed and not job.import_failed
                for h in p.conditional:
                    nb = p.cond_near.get(h) if ok else None
                    if not nb or not is_strict(h) or any(num2id[x] not in passed for x in nb):
                        nums.add(h)
        elif p.conditional:
            nums |= p.conditional
        if p.job is not None and not p.run and not p.job.done:
            nums |= set(all_nums)  # the import check itself did not finish
        if p.keyref and (p.job is None or p.job.state_ok is not True):
            # The import-time state check did not prove the edit confined to the
            # edited dict entries: use the unrefined selection.
            fb = 0
            for info in p.keyref:
                fb |= info["fallback"]
            nums |= {idx2num[i] for i in analysis.iter_bits(fb) if idx2num[i] >= 0}
            msg = p.job.state_msg if p.job is not None else "no check"
            log(f"{p.commit}: key refinement not confirmed ({msg[:160]}), unrefined selection")
        results[p.commit] = sorted(n for n in nums if n in dur_ms)
        cost, workers = pack(results[p.commit], dur_ms)
        log(f"{p.commit}: selected={len(results[p.commit])} cost={cost / 1000:.3f}s workers={len(workers)}")
    return results


def dominated_heavy(an, cand_bits, idx_sec, cheap_all):
    """Heavy candidates that cannot fail unless a cheap candidate fails too.

    Only for patches whose every change is a statement inside a function.  A heavy
    test is dominated when, for each change, at least MIN_CHEAP_COVER cheap tests run
    the changed statement and every line of the enclosing function the heavy test
    runs is also run by some cheap test (no path only the heavy test exercises)."""
    trace = an.trace
    if trace.get("nonsite") or not (trace["runtime"] or trace.get("import")):
        return 0
    heavy = 0
    for i in __import__("analysis").iter_bits(cand_bits):
        if idx_sec[i] >= HEAVY_SEC:
            heavy |= 1 << i
    if not heavy:
        return 0
    if trace.get("import"):
        # Class- or module-level change: no enclosing function to compare paths in.
        # Measured-risk rule: enough cheap tests use the changed code for one of them
        # to fail whenever a heavy one does.
        if not DOMINANCE_IMPORT or bin(cand_bits & cheap_all).count("1") < MIN_CHEAP_IMPORT:
            return 0
    skippable = heavy
    for rel, units, (flo, fhi) in trace["runtime"]:
        covering = an.units_bits(rel, units) & cheap_all & cand_bits
        if bin(covering).count("1") < MIN_CHEAP_COVER:
            return 0
        d = an.lines.get(rel) or {}
        for ln in range(flo, fhi + 1):
            sid = d.get(ln)
            if sid is None:
                continue
            b = an.sets[sid]
            if not (b & cheap_all):
                skippable &= ~b
    return skippable


_ELIGIBLE = None
_STRICT = 0


def probe_eligible(maps):
    """Heavy tests that may be gated by probes.

    Content-independent ones (no fixture reads, never run the parser, e.g. a generator
    memory check) always qualify.  Content-driven ones qualify only when nearly every
    sqlglot line they run is also run by some other test (UNIQUE_MAX): a heavy test
    with its own private code paths (e.g. one that types every SQL function) can fail
    on a changed table entry that no cheap test exercises."""
    global _ELIGIBLE, _STRICT
    if _ELIGIBLE is None and os.environ.get("BR_PROBE_ALL") == "1":  # dev: measure risk
        _ELIGIBLE = (1 << len(maps["tests"])) - 1
    if _ELIGIBLE is None:
        n = len(maps["tests"])
        content = 0
        for rel, b in maps["reads"].items():
            if rel.startswith("tests/fixtures/"):
                content |= b
        for sid in set((maps["lines"].get("sqlglot/parser.py") or {}).values()):
            content |= maps["sets"][sid]
        eligible = ((1 << n) - 1) & ~content
        strict = eligible
        if UNIQUE_MAX > 0:
            slow = [i for i, d in enumerate(maps["dur"]) if d >= HEAVY_SEC and content >> i & 1]
            uses = {}
            for rel, lines in maps["lines"].items():
                if rel.startswith("sqlglot/"):
                    for sid in lines.values():
                        uses[sid] = uses.get(sid, 0) + 1
            sets = maps["sets"]
            for h in slow:
                bit = 1 << h
                total = sum(c for sid, c in uses.items() if sets[sid] & bit)
                own = sum(c for sid, c in uses.items() if sets[sid] == bit)
                # Scale-driven tests (few lines, long run: huge inputs, node limits) fail
                # on changes to cost or limits that small inputs never reveal.
                if total and own <= UNIQUE_MAX * total and total >= MIN_LINE_RATE * maps["dur"][h]:
                    eligible |= bit
                if total and own <= STRICT_UNIQUE * total and total >= STRICT_LINE_RATE * maps["dur"][h]:
                    strict |= bit
        _ELIGIBLE = eligible
        _STRICT = strict
    return _ELIGIBLE


FOCUS_PROBES = int(os.environ.get("BR_FOCUS_PROBES", "10"))


def focus_tests(an, maps, idx2num):
    """For changes inside functions: test number -> how many lines of the changed
    functions it runs (the tests most specific to the changed code first)."""
    counts = {}
    for rel, _units, fspan in an.trace.get("runtime") or ():
        lines = maps["lines"].get(rel) or {}
        for ln in range(fspan[0], fspan[1] + 1):
            sid = lines.get(ln)
            if sid is None:
                continue
            b = maps["sets"][sid]
            while b:
                low = b & -b
                i = low.bit_length() - 1
                b ^= low
                n = idx2num[i]
                if n >= 0:
                    counts[n] = counts.get(n, 0) + 1
    return counts


def probe_drop(plan, maps, num2idx, num2id, verifiable, sec, focus=None, limit=None):
    """Every statically selected test becomes conditional on a few diverse cheap
    probes (the cheapest verifiable test of each file): selected only when the
    verification run finds the change live."""
    if not plan.static:
        return
    pool = [n for n in plan.static if sec[n] < HEAVY_SEC and n in num2idx and maps["tests"][num2idx[n]] in verifiable]
    # Round robin over test files: the cheapest test of every file, then the second
    # cheapest of every file, ... (big files such as the optimizer's get several).
    per_file = {}
    for n in sorted(pool, key=lambda n: (sec[n], n)):
        per_file.setdefault(num2id[n].split("::", 1)[0], []).append(n)
    order = []
    depth = 0
    while len(order) < len(pool):
        layer = [ts[depth] for ts in per_file.values() if depth < len(ts)]
        if not layer:
            break
        order.extend(sorted(layer, key=lambda n: (sec[n], n)))
        depth += 1
    probes = set(order[: limit or PROBE_DROP_N])
    if focus:
        ranked = sorted((n for n in pool if n in focus), key=lambda n: (-focus[n], sec[n], n))
        probes |= set(ranked[:FOCUS_PROBES])
    if not probes:
        return
    rest = plan.static - probes
    plan.conditional |= rest
    plan.static -= rest | probes
    if not plan.run:
        plan.stop_on_fail = True
    plan.run |= probes
    plan.notes.append(f"probedrop({len(rest)}:{len(probes)})")


def add_probes(plan, cands, maps, idx2num, num2idx, num2id, verifiable, sec):
    """Measured-risk rule for class- and module-level changes.

    Each heavy candidate is selected only if one of its closest cheap coverage
    neighbours (also candidates) fails.  The neighbours are run as probes (stop at
    the first failure); every other cheap candidate keeps its usual treatment."""
    near = maps.get("near") or {}
    eligible = probe_eligible(maps)
    heavy = [
        n
        for n in plan.static
        if n in cands and sec[n] >= HEAVY_SEC and n not in plan.conditional and eligible >> num2idx.get(n, 0) & 1
    ]
    if not heavy:
        return
    probes, gated = set(), set()
    for h in heavy:
        picks = []
        for i in near.get(num2idx.get(h, -1), ()):
            n = idx2num[i]
            if n >= 0 and n in cands and n != h and sec[n] < HEAVY_SEC and maps["tests"][i] in verifiable:
                picks.append(n)
                if len(picks) >= PROBES_PER_HEAVY:
                    break
        if len(picks) >= MIN_PROBES:
            probes.update(picks)
            gated.add(h)
            plan.cond_near[h] = set(picks)
    if not gated:
        return
    # Diversity: the cheapest verifiable candidate of every test file, so a change
    # that breaks one specific dialect or component is probed by its own tests.
    cheapest = {}
    for n in cands:
        if n in gated or sec[n] >= HEAVY_SEC or maps["tests"][num2idx.get(n, 0)] not in verifiable:
            continue
        f = num2id[n].split("::", 1)[0]
        if f not in cheapest or (sec[n], n) < (sec[cheapest[f]], cheapest[f]):
            cheapest[f] = n
    probes.update(sorted(cheapest.values(), key=lambda n: (sec[n], n))[:DIVERSE_PROBES])
    plan.conditional |= gated
    plan.static -= gated | probes
    if not plan.run:
        plan.stop_on_fail = True
    plan.run |= probes
    plan.notes.append(f"probes({len(gated)}:{len(probes)})")


_NEAR = {"num2idx": {}}


def is_strict(n):
    """Heavy test (number) meeting the strict probe criteria."""
    i = _NEAR["num2idx"].get(n)
    return i is not None and (_STRICT >> i) & 1 == 1


def neighbours(h, pool):
    """The closest cheap coverage neighbours of heavy test h (numbers) within pool."""
    near, idx2num, num2idx = _NEAR.get("near") or {}, _NEAR.get("idx2num"), _NEAR.get("num2idx")
    if idx2num is None:
        return set()
    out = []
    for i in near.get(num2idx.get(h, -1), ()):
        x = idx2num[i]
        if x in pool:
            out.append(x)
            if len(out) >= PROBES_PER_HEAVY:
                break
    return set(out) if len(out) >= MIN_PROBES else set()


def order_for_run(nodeids, early_exit):
    """Test order inside a verification job.  Tests stay grouped by class (so class
    fixtures run once); with early exit the cheapest classes, and the cheapest tests
    in them, go first so a live change is caught after as little CPU as possible."""
    if not early_exit:
        return sorted(nodeids)
    tcpu = CALIB.get("test_cpu", {})
    scpu = CALIB.get("setup_cpu", {})
    groups = {}
    for nid in nodeids:
        groups.setdefault(nid.rsplit("::", 1)[0], []).append(nid)
    out = []
    for cls in sorted(groups, key=lambda c: (scpu.get(c, 0.0) + min(tcpu.get(n, 0.0) for n in groups[c]), c)):
        out.extend(sorted(groups[cls], key=lambda n: (tcpu.get(n, 0.0), n)))
    return out


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

    test_cpu = CALIB.get("test_cpu", {})
    collect_cpu = CALIB.get("collect_cpu", {})
    setup_cpu = CALIB.get("setup_cpu", {})
    overhead = CALIB.get("job_overhead", RUN_OVERHEAD)

    def run_cost(n):
        nid = num2id[n]
        return test_cpu.get(nid, CPU_FACTOR * sec[n] + (POOL_OVERHEAD if n in pool_nums else 0.0))

    runnable.sort(key=lambda n: (run_cost(n), n))
    best_cost = pack(cands | from_fixed, dur_ms)[0] / 1000.0
    best_j = 0
    best_stop = False
    run_sum = overhead
    files = set()
    classes = set()
    for j, n in enumerate(runnable, start=1):
        nid = num2id[n]
        run_sum += run_cost(n)
        f = file_of[n]
        if f not in files:
            files.add(f)
            run_sum += collect_cpu.get(f, FILE_OVERHEAD)
        c = nid.rsplit("::", 1)[0]
        if c not in classes:
            classes.add(c)
            run_sum += setup_cpu.get(c, 0.0)
        rest = never | set(runnable[j:]) | from_fixed
        q = plan.fail_prior
        p_rest = pack(rest, dur_ms)[0] / 1000.0
        p_all = pack(rest | set(runnable[:j]), dur_ms)[0] / 1000.0
        # Full run: on a live change about half of the run tests fail and are selected.
        if j == 1:
            live_full = p_all
        elif p_rest == 0.0:
            # At least one failing test is selected: one worker plus the cheapest test.
            live_full = max(WORKER_MS / 1000.0 + sec[runnable[0]], 0.5 * p_all)
        else:
            live_full = p_rest + 0.5 * (p_all - p_rest)
        full = run_sum + (1.0 - q) * p_rest + q * live_full
        # Stop at the first failure: a live change costs little CPU, but everything
        # run is selected (assume half the run happened before the failure).
        frac = 1.0 if j == 1 else 0.5
        stop = (1.0 - q) * (run_sum + p_rest) + q * (overhead + frac * (run_sum - overhead) + p_all)
        for cost, mode in ((full, False), (stop, True)):
            if cost < best_cost:
                best_cost, best_j, best_stop = cost, j, mode
    plan.run = set(runnable[:best_j])
    plan.static = from_fixed | (cands - plan.run)
    plan.stop_on_fail = best_stop and bool(plan.run)
    if plan.stop_on_fail:
        plan.notes.append("stop-on-fail")

    # Dominance: heavy tests whose code every cheap covering test also runs are only
    # selected when one of those cheap tests fails.
    dom = plan.dominated & cands
    if dom:
        run_all = [n for n in runnable if n not in dom and sec[n] < HEAVY_SEC]
        if run_all:
            rest = (cands - set(run_all) - dom) | from_fixed
            cost = overhead
            files, classes = set(), set()
            for n in run_all:
                nid = num2id[n]
                cost += run_cost(n)
                if file_of[n] not in files:
                    files.add(file_of[n])
                    cost += collect_cpu.get(file_of[n], FILE_OVERHEAD)
                c = nid.rsplit("::", 1)[0]
                if c not in classes:
                    classes.add(c)
                    cost += setup_cpu.get(c, 0.0)
            q = plan.fail_prior
            # Live change: the run stops at the first failure (assume a quarter of it ran)
            # and everything is selected.  Otherwise the whole cheap set ran and passed.
            cost = (1.0 - q) * cost + q * (overhead + 0.25 * (cost - overhead))
            cost += q * pack(cands | from_fixed, dur_ms)[0] / 1000.0
            cost += (1.0 - q) * pack(rest, dur_ms)[0] / 1000.0
            if cost < best_cost:
                best_cost = cost
                plan.run = set(run_all)
                plan.conditional = dom
                plan.stop_on_fail = True
                plan.cond_near = {h: neighbours(h, set(run_all)) for h in dom}
                plan.static = rest
                plan.notes.append(f"dominance({len(dom)})")
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
        nodeids = order_for_run([num2id[n] for n in p.run], p.stop_on_fail or bool(p.conditional))
        if PER_HEAVY and p.cond_near:
            # The gated heavy tests' own neighbours first: once they pass, a later
            # failure (which stops the run) no longer drags those heavy tests in.
            first = {num2id[x] for nb in p.cond_near.values() for x in nb}
            nodeids = [x for x in nodeids if x in first] + [x for x in nodeids if x not in first]
        tcpu = CALIB.get("test_cpu", {})
        timeouts = {
            num2id[n]: max(3.0, 6.0 * sec[n] + 2.0, 4.0 * tcpu.get(num2id[n], 0.0) + 2.0) for n in p.run
        }
        budget = 20.0 + sum(timeouts.values()) + (25.0 if p.modules else 0.0)
        spec = {
            "root": root,
            "nodeids": nodeids,
            "timeouts": timeouts,
            "import_check": p.modules,
            "lazy_all": LAZY_ALL,
            # Once a cheap test fails the dominated heavy tests are selected anyway.
            "stop_on_fail": bool(p.conditional) or p.stop_on_fail,
        }
        if p.keyref:
            allowed = {}
            for info in p.keyref:
                for k, v in info["allowed"].items():
                    allowed.setdefault(k, set()).update(v)
            spec["state_check"] = {
                "modules": sorted({m for info in p.keyref for m in info["modules"]}),
                "allowed": {k: sorted(v) for k, v in allowed.items()},
            }
            runner.STATE_BASE = an.keydata()
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
        if job.spec.get("import_check") and (not job.imports_ok or job.import_hang or not job.all_ok):
            # The import check itself crashed, hung or was killed: assume the patched
            # code breaks every importer.
            log(f"{p.commit}: import check did not complete, selecting everything")
            p.extra |= set(id2num.values())
        if job.import_failed:
            # A module that imports cleanly on the base no longer does: every test
            # file importing it (even transitively, at collection) fails.
            log(f"{p.commit}: import failures {sorted(job.import_failed)[:3]}, selecting everything")
            p.extra |= set(id2num.values())
        for mod in job.import_failed:
            rel = mod.replace(".", "/")
            rel = rel + ".py" if os.path.exists(os.path.join(REPO, rel + ".py")) else rel + "/__init__.py"
            bits = an.broken_module(rel)
            p.extra |= {idx2num[i] for i in __import__("analysis").iter_bits(bits) if idx2num[i] >= 0}
        for nodeid in job.collect_failed:
            f = nodeid.split("::", 1)[0]
            p.extra |= {n for i, n in id2num.items() if i.split("::", 1)[0] == f}

    # Children are forked from this process: drop the analysis caches (ASTs of big
    # modules, the source corpus) and freeze what is left, so the children's garbage
    # collector does not touch, and copy, the parent's heap.
    import gc

    an.release()
    gc.collect()
    gc.freeze()
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
