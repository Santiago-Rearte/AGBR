"""pytest plugin: attributes traced execution to test phases and test-module collection."""

import os
import time

import pytest

import brcore

_ACTIVE = False
DURATIONS = {}


def _set(ctx):
    os.environ["BR_CTX"] = ctx
    brcore.set_ctx(ctx)


def pytest_configure(config):
    global _ACTIVE
    is_controller = (
        getattr(config.option, "numprocesses", None) not in (None, 0)
        and not hasattr(config, "workerinput")
    )
    if is_controller:
        brcore.CHILD_MODE = False
        if brcore.STARTED:
            brcore.M.set_events(brcore.TOOL, 0)
        return
    _ACTIVE = True
    brcore.CHILD_MODE = False
    brcore.install_import_hooks()
    brcore.start("preimport")
    # Lines that run on `import sqlglot` belong to no test in particular.
    import sqlglot  # noqa: F401

    brcore.set_ctx("misc")

    import concurrent.futures.process as cfp

    original_submit = cfp.ProcessPoolExecutor.submit

    def submit(self, fn, /, *args, **kwargs):
        return original_submit(self, brcore.call_in_ctx, brcore.CTX, fn, args, kwargs)

    cfp.ProcessPoolExecutor.submit = submit


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    if _ACTIVE and isinstance(collector, pytest.Module):
        _set("collect|" + collector.nodeid)
        yield
        _set("misc")
    else:
        yield


def _phase(item, phase):
    _set(item.nodeid + "|" + phase)


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item):
    if _ACTIVE:
        _phase(item, "setup")
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_call(item):
    if _ACTIVE:
        _phase(item, "call")
    t0 = time.perf_counter()
    yield
    DURATIONS[item.nodeid] = time.perf_counter() - t0
    OUTCOMES.setdefault(item.nodeid, "passed")


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(item, nextitem):
    if _ACTIVE:
        _phase(item, "teardown")
    yield
    if _ACTIVE:
        _set("misc")
        _flush()


_FH = None


def _flush(extra=None):
    global _FH
    if _FH is None:
        _FH = brcore._orig_open(os.path.join(os.environ["BR_OUT"], f"w-{os.getpid()}.pkl"), "ab")
    brcore.flush(_FH, keep_current=False, extra=extra)


OUTCOMES = {}


def pytest_runtest_logreport(report):
    prev = OUTCOMES.get(report.nodeid, "passed")
    if report.failed:
        OUTCOMES[report.nodeid] = "failed"
    elif report.skipped and prev != "failed":
        OUTCOMES[report.nodeid] = "skipped"
    else:
        OUTCOMES.setdefault(report.nodeid, prev)


def pytest_sessionfinish(session):
    if not _ACTIVE:
        return
    brcore.M.set_events(brcore.TOOL, 0)
    _flush({"durations": DURATIONS, "outcomes": OUTCOMES})
    _FH.close()
