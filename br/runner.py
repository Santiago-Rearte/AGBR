"""Runs candidate tests against a patched tree in forked children.

A test counts as verified only when pytest reported its call phase as passed and
no report for it (setup, call, teardown, subtests) failed or was skipped.  Every
other candidate stays selected, so a crash, a timeout or a collection error can
only make a selection bigger.
"""

import importlib
import json
import os
import selectors
import shutil
import signal
import sys
import time

DEVNULL = os.open(os.devnull, os.O_RDWR)


def log(*args):
    print(*args, file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Overlay trees: symlinks to the base checkout, real files only where patched.


def make_overlay(dst, src, files, deleted=()):
    paths = set(files) | set(deleted)
    dirs = {""}
    for rel in paths:
        parts = rel.split("/")
        for k in range(1, len(parts)):
            dirs.add("/".join(parts[:k]))
    for d in sorted(dirs, key=lambda x: (x.count("/"), x)):
        target = os.path.join(dst, d) if d else dst
        os.makedirs(target, exist_ok=True)
        srcdir = os.path.join(src, d) if d else src
        try:
            names = os.listdir(srcdir)
        except OSError:
            names = []
        for name in names:
            rel = f"{d}/{name}" if d else name
            if rel in dirs or rel in paths:
                continue
            os.symlink(os.path.join(src, rel), os.path.join(dst, rel))
    for rel, data in files.items():
        with open(os.path.join(dst, rel), "wb") as fh:
            fh.write(data)


# ---------------------------------------------------------------------------
# Child side.


class BRTimeout(BaseException):
    pass


def _alarm(signum, frame):
    raise BRTimeout("test exceeded its time budget")


class Reporter:
    def __init__(self, out, timeouts, default_timeout):
        self.out = out
        self.timeouts = timeouts
        self.default_timeout = default_timeout
        self.bad = set()
        self.ok = set()
        self.t_test = 0.0
        self.t_collect = {}

    def pytest_collectstart(self, collector):
        self.t_collect[collector.nodeid] = _cpu()

    def pytest_collectreport(self, report):
        if report.failed:
            self.out.write(f"COLLECTFAIL {report.nodeid}\n")
        t = self.t_collect.pop(report.nodeid, None)
        if t is not None and report.nodeid.endswith(".py"):
            self.out.write(f"COLL {_cpu() - t:.4f} {report.nodeid}\n")

    def pytest_runtest_protocol(self, item, nextitem):
        signal.signal(signal.SIGALRM, _alarm)
        signal.setitimer(signal.ITIMER_REAL, self.timeouts.get(item.nodeid, self.default_timeout))
        self.t_test = _cpu()

    def pytest_runtest_logreport(self, report):
        if report.failed or report.skipped:
            self.bad.add(report.nodeid)
        elif report.when == "call" and report.passed:
            self.ok.add(report.nodeid)

    def pytest_runtest_logfinish(self, nodeid, location):
        signal.setitimer(signal.ITIMER_REAL, 0)
        self.out.write(f"CPU {_cpu() - self.t_test:.4f} {nodeid}\n")
        if nodeid in self.ok and nodeid not in self.bad:
            self.out.write(f"PASS {nodeid}\n")


def _cpu():
    t = os.times()
    return t.user + t.system + t.children_user + t.children_system


def install_inline_pools():
    """Process pools run tasks in this process, with arguments and results still
    round-tripped through pickle as a spawned worker would see them."""
    import concurrent.futures as cf
    import concurrent.futures.process as cfp
    import pickle

    class InlineProcessPoolExecutor(cf.Executor):
        def __init__(self, max_workers=None, mp_context=None, initializer=None, initargs=(), **_kw):
            self._initializer = initializer
            self._initargs = initargs
            self._ready = False

        def submit(self, fn, /, *args, **kwargs):
            future = cf.Future()
            try:
                if not self._ready:
                    self._ready = True
                    if self._initializer is not None:
                        self._initializer(*pickle.loads(pickle.dumps(self._initargs)))
                fn2, args2, kwargs2 = pickle.loads(pickle.dumps((fn, args, kwargs)))
                result = fn2(*args2, **kwargs2)
                future.set_result(pickle.loads(pickle.dumps(result)))
            except BRTimeout:
                raise
            except BaseException as error:  # noqa: BLE001
                future.set_exception(error)
            return future

        def map(self, fn, *iterables, timeout=None, chunksize=1):
            return [self.submit(fn, *args).result() for args in zip(*iterables)]

        def shutdown(self, wait=True, *, cancel_futures=False):
            pass

    cfp.ProcessPoolExecutor = InlineProcessPoolExecutor
    cf.ProcessPoolExecutor = InlineProcessPoolExecutor


def run_unittest(nodeids, out, timeouts, default_timeout, stop_on_fail=False, keep=()):
    """Runs unittest-style tests (all of sqlglot's) without pytest's start-up and
    collection cost.  Class and module fixtures are handled by unittest.TestSuite."""
    import unittest

    by_id = {}
    tests = []
    files = {}
    for nid in nodeids:
        f, _, rest = nid.partition("::")
        files.setdefault(f, []).append((rest, nid))
    for f, items in files.items():
        modname = f[:-3].replace("/", ".")
        t0 = _cpu()
        try:
            module = importlib.import_module(modname)
        except BaseException as error:  # noqa: BLE001
            out.write(f"COLLECTFAIL {f} {type(error).__name__}\n")
            continue
        out.write(f"COLL {_cpu() - t0:.4f} {f}\n")
        for rest, nid in items:
            cls_name, _, meth = rest.partition("::")
            cls = getattr(module, cls_name, None)
            if not (isinstance(cls, type) and issubclass(cls, unittest.TestCase) and hasattr(cls, meth)):
                out.write(f"NOTUNIT {nid}\n")
                continue
            test = cls(meth)
            by_id[test.id()] = nid
            tests.append(test)

    class Result(unittest.TestResult):
        def __init__(self):
            super().__init__()
            self.bad = set()
            self.ok = set()
            self.t0 = 0.0
            self.t_end = _cpu()
            self.cls = None
            self.ran = set()
            self.stop_on_fail = stop_on_fail

        def _nid(self, test):
            return by_id.get(test.id()) if hasattr(test, "id") else None

        def startTest(self, test):
            super().startTest(test)
            nid = self._nid(test)
            now = _cpu()
            cls = nid.rsplit("::", 1)[0] if nid else None
            if cls is not None and cls != self.cls:
                # Time since the previous test went to class/module set-up of this one.
                out.write(f"SETUP {now - self.t_end:.4f} {cls}\n")
                self.cls = cls
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, timeouts.get(nid, default_timeout))
            self.t0 = _cpu()

        def stopTest(self, test):
            signal.setitimer(signal.ITIMER_REAL, 0)
            super().stopTest(test)
            self.t_end = _cpu()
            nid = self._nid(test)
            if nid is None:
                return
            out.write(f"CPU {self.t_end - self.t0:.4f} {nid}\n")
            self.ran.add(nid)
            if nid in self.ok and nid not in self.bad:
                out.write(f"PASS {nid}\n")
            elif self.stop_on_fail:
                out.write("STOPPED\n")
                out.flush()
                if not keep:
                    os._exit(0)
                # Some tests must still run (the neighbours deciding whether gated
                # heavy tests are needed): stop this pass, run them next.
                self.shouldStop = True

        def _bad(self, test):
            nid = self._nid(test)
            if nid is not None:
                self.bad.add(nid)

        def addError(self, test, err):
            super().addError(test, err)
            self._bad(test)

        def addFailure(self, test, err):
            super().addFailure(test, err)
            self._bad(test)

        def addSkip(self, test, reason):
            super().addSkip(test, reason)
            self._bad(test)

        def addExpectedFailure(self, test, err):
            super().addExpectedFailure(test, err)
            self._bad(test)

        def addUnexpectedSuccess(self, test):
            super().addUnexpectedSuccess(test)
            self._bad(test)

        def addSubTest(self, test, subtest, err):
            super().addSubTest(test, subtest, err)
            if err is not None:
                self._bad(test)

        def addSuccess(self, test):
            super().addSuccess(test)
            nid = self._nid(test)
            if nid is not None:
                self.ok.add(nid)

    first = Result()
    unittest.TestSuite(tests).run(first)
    # (When nothing passed before the first failure the change breaks everything it
    # touches: the gated heavy tests are selected anyway, so skip the second pass.)
    if first.shouldStop and keep and (first.ok - first.bad):
        # keep: groups of tests (the neighbours of one gated heavy test); a group is
        # decided as soon as one of its tests fails.
        groups = [set(g) for g in keep]

        def failed(nid):
            return nid in first.ran and nid not in (first.ok - first.bad)

        open_groups = [g for g in groups if not any(failed(n) for n in g)]
        wanted = set().union(*open_groups) if open_groups else set()
        rest = [t for t in tests if by_id.get(t.id()) in wanted and by_id.get(t.id()) not in first.ran]
        if rest:

            class Second(Result):
                def stopTest(self, test):
                    super().stopTest(test)
                    nid = self._nid(test)
                    if nid is not None and not (nid in self.ok and nid not in self.bad):
                        for g in list(open_groups):
                            if nid in g:
                                open_groups.remove(g)
                        if not open_groups:
                            self.shouldStop = True

            second = Second()
            second.stop_on_fail = False
            second.ran = first.ran
            unittest.TestSuite(rest).run(second)


def check_lazy_attributes(out, base_all, only):
    """Resolves every lazy attribute (PEP 562) of loaded sqlglot modules: the names
    they export now and the ones they exported on the base commit."""
    for name, module in list(sys.modules.items()):
        if name not in only or module is None:
            continue
        if "__getattr__" not in vars(module) and name not in base_all:
            continue
        names = set(getattr(module, "__all__", ()) or ()) | set(base_all.get(name, ()))
        for attr in sorted(names):
            try:
                getattr(module, attr)
            except BaseException as error:  # noqa: BLE001
                out.write(f"IMPORTFAIL {name} {type(error).__name__}:{attr}\n")
                if isinstance(error, BRTimeout):
                    out.write("IMPORTHANG\n")
                break


STATE_BASE = None  # keys.pkl content, set by the parent before forking


def check_state(out, sc):
    """Import-time state of the patched tree vs the base: only the edited entries of
    the edited class dicts (and same-named copies) may differ.  Writes STATEOK or
    STATEDIFF."""
    try:
        for mod in sc["modules"]:
            try:
                importlib.import_module(mod)
            except BRTimeout:
                raise
            except BaseException as error:  # noqa: BLE001
                out.write(f"IMPORTFAIL {mod} {type(error).__name__}\n")
                out.write("STATEDIFF import\n")
                return
        # Every module, as for the base state: registries filled on import (dialect
        # classes) and derived tables anywhere are compared like for like.
        import pkgutil

        import sqlglot

        for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
            if info.name != "sqlglot.__main__":
                try:
                    importlib.import_module(info.name)
                except BRTimeout:
                    raise
                except BaseException:  # noqa: BLE001 - the base swallowed these too
                    pass
        import statefp

        base = STATE_BASE or {}
        loaded = {m for m in sys.modules if m == "sqlglot" or m.startswith("sqlglot.")}
        bad = statefp.diff(
            base.get("state", {}),
            statefp.state(loaded),
            {k: set(v) for k, v in sc["allowed"].items()},
            loaded,
            base.get("unstable", ()),
        )
        out.write("STATEOK\n" if not bad and base else f"STATEDIFF {len(bad)} {bad[:1]}\n")
    except BRTimeout:
        out.write("STATEDIFF timeout\nIMPORTHANG\n")
        raise
    except BaseException as error:  # noqa: BLE001
        out.write(f"STATEDIFF error {type(error).__name__}\n")


def check_all_modules(out, spec):
    """In a forked grandchild (so the tests about to run see the usual lazily loaded
    state): import every sqlglot module, reporting IMPORTFAIL for any failure, then the
    import-time state check.  Modules loaded lazily (a dialect's generator) may run
    patched code at import time, e.g. parse SQL templates, and every test importing
    them would fail at collection.  All of them import cleanly on the base."""
    out.flush()
    remaining = signal.getitimer(signal.ITIMER_REAL)[0] or spec.get("import_timeout", 20.0)
    pid = os.fork()
    if pid == 0:
        try:
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, remaining)
            import pkgutil

            import sqlglot

            for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
                if info.name == "sqlglot.__main__" or info.name in sys.modules:
                    continue
                try:
                    importlib.import_module(info.name)
                except BRTimeout:
                    raise
                except BaseException as error:  # noqa: BLE001
                    out.write(f"IMPORTFAIL {info.name} {type(error).__name__}\n")
            if spec.get("state_check") is not None:
                check_state(out, spec["state_check"])
            out.write("ALLOK\n")
        except BRTimeout:
            out.write("IMPORTHANG\n")
        except BaseException as error:  # noqa: BLE001
            out.write(f"IMPORTFAIL sqlglot {type(error).__name__}\n")
        finally:
            try:
                out.flush()
            except Exception:
                pass
            os._exit(0)
    try:
        os.waitpid(pid, 0)
    except BRTimeout:
        try:
            os.kill(pid, signal.SIGKILL)
            os.waitpid(pid, 0)
        except OSError:
            pass
        out.write("IMPORTHANG\n")
        raise


def child_main(spec, wfd):
    try:
        os.setpgid(0, 0)
    except OSError:
        pass
    os.dup2(DEVNULL, 0)
    os.dup2(DEVNULL, 1)
    out = os.fdopen(wfd, "w", buffering=1)
    try:
        root = spec["root"]
        os.chdir(root)
        sys.path.insert(0, root)
        if spec.get("inline_pools", True):
            install_inline_pools()
        if spec.get("import_check"):
            # A patched import can also hang (an infinite loop in code run at import
            # time): bound it, and confirm completion so the parent can tell.
            signal.signal(signal.SIGALRM, _alarm)
            signal.setitimer(signal.ITIMER_REAL, spec.get("import_timeout", 20.0))
            try:
                for mod in spec["import_check"]:
                    try:
                        importlib.import_module(mod)
                    except BaseException as error:  # noqa: BLE001
                        out.write(f"IMPORTFAIL {mod} {type(error).__name__}\n")
                        if isinstance(error, BRTimeout):
                            out.write("IMPORTHANG\n")
                            break
                check_lazy_attributes(out, spec.get("lazy_all", {}), set(spec["import_check"]))
                if spec.get("import_all", True) or spec.get("state_check") is not None:
                    check_all_modules(out, spec)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            out.write("IMPORTSOK\n")
            out.flush()
        if spec.get("mode", "unittest") == "unittest":
            run_unittest(
                spec["nodeids"],
                out,
                spec.get("timeouts", {}),
                spec.get("default_timeout", 30.0),
                spec.get("stop_on_fail", False),
                [list(g) for g in (spec.get("keep") or ())],
            )
            out.write("DONE\n")
            out.flush()
            os._exit(0)
        import pytest

        reporter = Reporter(out, spec.get("timeouts", {}), spec.get("default_timeout", 30.0))
        args = list(spec["nodeids"]) + [
            "-q",
            "-p",
            "no:cacheprovider",
            "--rootdir",
            root,
            "-o",
            "addopts=",
            "--assert=plain",
        ]
        pytest.main(args, plugins=[reporter])
        out.write("DONE\n")
        out.flush()
    except BaseException as error:  # noqa: BLE001
        try:
            out.write(f"CRASH {type(error).__name__}\n")
            out.flush()
        except Exception:
            pass
    os._exit(0)


# ---------------------------------------------------------------------------
# Parent side.


def warm_imports(extra_modules=(), with_pytest=False):
    """Imports, once before any fork, what the children need: unittest (sqlglot reads
    `"unittest" in sys.modules` at import time, so it must come first) and the
    third-party modules of the test files about to run.  pytest and its plugins
    (0.3 s of CPU) only for the pytest mode: the unittest runner never needs them."""
    os.environ.setdefault("PYTEST_DISABLE_PLUGIN_AUTOLOAD", "1")
    import unittest  # noqa: F401

    if with_pytest or os.environ.get("BR_WARM_PYTEST") == "1":
        import pytest  # noqa: F401
        from _pytest.config import default_plugins

        for name in default_plugins:
            try:
                importlib.import_module(f"_pytest.{name}")
            except Exception:
                pass
    for mod in extra_modules:
        try:
            importlib.import_module(mod)
        except Exception:
            pass


class Job:
    def __init__(self, key, spec, budget):
        self.key = key
        self.spec = spec
        self.budget = budget
        self.passed = set()
        self.collect_failed = set()
        self.import_failed = set()
        self.done = False
        self.crashed = None
        self.stopped = False
        self.imports_ok = False
        self.import_hang = False
        self.state_ok = None
        self.all_ok = False
        self.state_msg = ""
        self.pid = None
        self.fd = None
        self.buf = b""
        self.deadline = None
        self.cpu = 0.0
        self.wall = 0.0
        self.test_cpu = {}
        self.collect_cpu = {}
        self.setup_cpu = {}


def _consume(job, data):
    job.buf += data
    *lines, job.buf = job.buf.split(b"\n")
    for raw in lines:
        line = raw.decode("utf-8", "replace")
        tag, _, rest = line.partition(" ")
        if tag == "PASS":
            job.passed.add(rest)
        elif tag == "COLLECTFAIL":
            job.collect_failed.add(rest)
        elif tag == "IMPORTFAIL":
            job.import_failed.add(rest.split(" ", 1)[0])
        elif tag == "CPU":
            secs, _, nodeid = rest.partition(" ")
            job.test_cpu[nodeid] = float(secs)
        elif tag == "COLL":
            secs, _, nodeid = rest.partition(" ")
            job.collect_cpu[nodeid.split(" ", 1)[0]] = float(secs)
        elif tag == "SETUP":
            secs, _, cls = rest.partition(" ")
            job.setup_cpu[cls] = float(secs)
        elif tag == "IMPORTHANG":
            job.import_hang = True
        elif tag == "IMPORTSOK":
            job.imports_ok = True
        elif tag == "ALLOK":
            job.all_ok = True
        elif tag == "STATEOK":
            job.state_ok = job.state_ok is None
        elif tag == "STATEDIFF":
            job.state_ok = False
            job.state_msg = rest
        elif tag == "STOPPED":
            job.stopped = True
        elif tag == "DONE":
            job.done = True
        elif tag == "CRASH":
            job.crashed = rest


def run_jobs(jobs, parallel=2, setup=None, cleanup=None):
    """Runs every job in a forked child, at most `parallel` at a time."""
    pending = list(jobs)
    active = {}
    sel = selectors.DefaultSelector()

    def start(job):
        if setup:
            setup(job)
        r, w = os.pipe()
        sys.stdout.flush()
        sys.stderr.flush()
        pid = os.fork()
        if pid == 0:
            os.close(r)
            try:
                sel.close()
            except Exception:
                pass
            child_main(job.spec, w)
        os.close(w)
        job.pid, job.fd = pid, r
        job.deadline = time.monotonic() + job.budget
        job.wall = time.monotonic()
        os.set_blocking(r, False)
        sel.register(r, selectors.EVENT_READ, job)
        active[pid] = job

    def finish(job, kill=False):
        if kill:
            try:
                os.killpg(job.pid, signal.SIGKILL)
            except OSError:
                try:
                    os.kill(job.pid, signal.SIGKILL)
                except OSError:
                    pass
        try:
            sel.unregister(job.fd)
        except Exception:
            pass
        try:
            while True:
                data = os.read(job.fd, 65536)
                if not data:
                    break
                _consume(job, data)
        except OSError:
            pass
        os.close(job.fd)
        _, status, usage = os.wait4(job.pid, 0)
        job.cpu = usage.ru_utime + usage.ru_stime
        job.wall = time.monotonic() - job.wall
        # Reap anything the child left behind in its process group.
        try:
            os.killpg(job.pid, signal.SIGKILL)
        except OSError:
            pass
        del active[job.pid]
        if cleanup:
            cleanup(job)

    while pending or active:
        while pending and len(active) < parallel:
            start(pending.pop(0))
        timeout = max(0.05, min(j.deadline for j in active.values()) - time.monotonic())
        for key, _ in sel.select(timeout):
            job = key.data
            try:
                data = os.read(job.fd, 65536)
            except BlockingIOError:
                continue
            except OSError:
                data = b""
            if data:
                _consume(job, data)
            else:
                finish(job)
        now = time.monotonic()
        for job in list(active.values()):
            if now > job.deadline:
                log(f"  job {job.key} over budget, killed")
                finish(job, kill=True)
    sel.close()
    return jobs


# ---------------------------------------------------------------------------
# Build-time calibration: which tests pass under this runner on the base commit.


def calibrate(repo, tests_list, out_path, parallel=4, chunks=24):
    warm_imports(["duckdb", "pandas", "numpy"])
    by_file = {}
    for t in tests_list:
        by_file.setdefault(t.split("::", 1)[0], []).append(t)
    groups = [[] for _ in range(chunks)]
    loads = [0] * chunks
    for f, ts in sorted(by_file.items(), key=lambda kv: -len(kv[1])):
        i = loads.index(min(loads))
        groups[i].extend(ts)
        loads[i] += len(ts)
    jobs = [
        Job(i, {"root": repo, "nodeids": g, "default_timeout": 600.0}, budget=900.0)
        for i, g in enumerate(groups)
        if g
    ]
    t0 = time.monotonic()
    run_jobs(jobs, parallel=parallel)
    passed = set()
    for job in jobs:
        passed |= job.passed
        if job.collect_failed or job.crashed or not job.done:
            log(f"  calibration job {job.key}: done={job.done} crash={job.crashed} collect={sorted(job.collect_failed)[:3]}")
    test_cpu, collect_cpu, setup_cpu = {}, {}, {}
    for job in jobs:
        test_cpu.update(job.test_cpu)
        collect_cpu.update(job.collect_cpu)
        setup_cpu.update(job.setup_cpu)
    # Fixed cost of one verification job: fork, pytest start-up, an empty session.
    first = sorted(tests_list)[0]
    probe = [Job(f"probe{i}", {"root": repo, "nodeids": [first], "default_timeout": 60.0}, 120.0) for i in range(3)]
    run_jobs(probe, parallel=1)
    overheads = sorted(
        j.cpu - sum(j.test_cpu.values()) - sum(j.collect_cpu.values()) - sum(j.setup_cpu.values()) for j in probe
    )
    failed = sorted(set(tests_list) - passed)
    log(f"calibration: {len(passed)} pass, {len(failed)} not verifiable, {time.monotonic() - t0:.1f}s")
    for t in failed[:30]:
        log(f"  not verifiable: {t}")
    with open(out_path, "w") as fh:
        json.dump(
            {"passed": sorted(passed), "test_cpu": test_cpu, "collect_cpu": collect_cpu, "setup_cpu": setup_cpu, "job_overhead": overheads[1]},
            fh,
        )
    log(f"job overhead {overheads}, collect total {sum(collect_cpu.values()):.2f}s, test cpu total {sum(test_cpu.values()):.2f}s")


if __name__ == "__main__":
    if sys.argv[1] == "--calibrate":
        import pickle

        with open(sys.argv[3], "rb") as fh:
            maps = pickle.load(fh)
        calibrate(sys.argv[2], maps["tests"], sys.argv[4], parallel=int(sys.argv[5]) if len(sys.argv) > 5 else 4)
