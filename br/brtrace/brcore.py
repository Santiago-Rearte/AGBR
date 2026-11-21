"""Per-context execution tracer built on sys.monitoring (Python 3.12+).

Used only at build time.  For every context (a pytest phase of one test, the
collection of one test module, or a task run inside a spawned worker) it records:

  L  executed lines of files under the sqlglot checkout   (fid * 1_000_000 + line)
  R  files under the checkout that were opened or read as source (relative paths)
  D  directories under the checkout that were listed      (relative paths)
  F  selected stdlib functions that ran (subprocess, doctest, inspect, ...)
  I  sqlglot classes that were instantiated               ("module:qualname")

Each line fires once per context (the callback returns DISABLE and the next
context calls restart_events), so the overhead stays small.
"""

import atexit
import builtins
import io
import os
import pickle
import sys

M = sys.monitoring
TOOL = 4
DISABLE = M.DISABLE
EV = M.events

REPO = os.path.realpath(os.environ.get("BR_REPO", "/opt/sqlglot")) + os.sep
NREPO = len(REPO)

FILES = []
FID = {}
CODE_FID = {}
CODE_KIND = {}

DATA = {}
CTX = None
L = R = D = F = I = None
CHILD_MODE = False
STARTED = False

_INIT_NAMES = {"__init__", "__new__", "__post_init__"}
_INTERESTING = {
    "linecache.py": {"getlines", "updatecache"},
    "inspect.py": {
        "getsource",
        "getsourcelines",
        "findsource",
        "getsourcefile",
        "getdoc",
        "getcomments",
        "getmembers",
    },
    "tokenize.py": {"open"},
    "doctest.py": None,
    "subprocess.py": {"Popen.__init__"},
    "process.py": {"ProcessPoolExecutor.submit", "ProcessPoolExecutor.__init__"},
    "runpy.py": {"run_path", "run_module"},
    "pkgutil.py": {"walk_packages", "iter_modules"},
    "pydoc.py": None,
}
_RECORD_ARGS = {"getlines", "updatecache"}


def _fid(fn):
    i = FID.get(fn)
    if i is None:
        i = FID[fn] = len(FILES)
        FILES.append(fn)
    return i


def _classify(code):
    fn = code.co_filename
    kind = 0
    if fn.startswith(REPO):
        CODE_FID[code] = _fid(fn)
        # In the import trace only functions count: module and class bodies are
        # analysed from the AST, and their lines would hide the lambdas they define.
        if not IMPORT_CTX or code.co_flags & 0x1:  # CO_OPTIMIZED
            try:
                M.set_local_events(TOOL, code, EV.LINE)
            except ValueError:
                pass
        kind = 2 if code.co_name in _INIT_NAMES else 1
    else:
        base = fn.rsplit("/", 1)[-1]
        names = _INTERESTING.get(base, False)
        if names is not False and (names is None or code.co_qualname in names):
            kind = 4 if code.co_name in _RECORD_ARGS else 3
    CODE_KIND[code] = kind
    return kind


def on_start(code, offset):
    kind = CODE_KIND.get(code)
    if kind is None:
        kind = _classify(code)
    if kind <= 1:
        return DISABLE
    if kind == 2:
        try:
            obj = sys._getframe(1).f_locals[code.co_varnames[0]]
            cls = obj if code.co_name == "__new__" else type(obj)
            I.add(cls)
        except Exception:
            pass
        return None
    base = code.co_filename.rsplit("/", 1)[-1]
    F.add(base + ":" + code.co_qualname)
    if kind == 4:
        try:
            fn = sys._getframe(1).f_locals.get("filename")
            if isinstance(fn, str):
                fn = os.path.abspath(fn)
                if fn.startswith(REPO):
                    R.add(fn[NREPO:])
        except Exception:
            pass
        return None
    return DISABLE


def on_line(code, line):
    L.add(CODE_FID[code] * 1_000_000 + line)
    return DISABLE


def _new():
    return [set(), set(), set(), set(), set()]


def set_ctx(name):
    global L, R, D, F, I, CTX
    rec = DATA.get(name)
    if rec is None:
        rec = DATA[name] = _new()
    L, R, D, F, I = rec
    CTX = name
    if STARTED:
        M.restart_events()


def _rel(p):
    try:
        p = os.path.abspath(os.fsdecode(p))
    except Exception:
        return None
    if p.startswith(REPO):
        return p[NREPO:]
    return None


_orig_open = builtins.open
_orig_listdir = os.listdir
_orig_scandir = os.scandir


def _br_open(file, *args, **kwargs):
    if not isinstance(file, int):
        r = _rel(file)
        if r is not None and R is not None:
            R.add(r)
    return _orig_open(file, *args, **kwargs)


def _br_listdir(path="."):
    r = _rel(path)
    if r is not None and D is not None:
        D.add(r)
    return _orig_listdir(path)


def _br_scandir(path="."):
    if not isinstance(path, int):
        r = _rel(path)
        if r is not None and D is not None:
            D.add(r)
    return _orig_scandir(path)


def start(ctx):
    global STARTED
    if STARTED:
        set_ctx(ctx)
        return
    builtins.open = _br_open
    io.open = _br_open
    os.listdir = _br_listdir
    os.scandir = _br_scandir
    M.use_tool_id(TOOL, "brtrace")
    M.register_callback(TOOL, EV.PY_START, on_start)
    M.register_callback(TOOL, EV.LINE, on_line)
    set_ctx(ctx)
    M.set_events(TOOL, EV.PY_START)
    STARTED = True


_FLUSHED_FILES = 0


def _names(classes):
    inst = set()
    for cls in classes:
        mod = getattr(cls, "__module__", "") or ""
        if mod.startswith("sqlglot") or mod.startswith("tests"):
            inst.add(mod + ":" + getattr(cls, "__qualname__", cls.__name__))
    return inst


def flush(fh, keep_current=True, extra=None):
    """Appends one record with every finished context to fh and forgets them."""
    global _FLUSHED_FILES
    out = {}
    for name in list(DATA):
        if keep_current and name == CTX:
            continue
        l, r, d, f, i = DATA.pop(name)
        if l or r or d or f or i:
            out[name] = (l, r, d, f, _names(i))
    record = {"files_base": _FLUSHED_FILES, "new_files": FILES[_FLUSHED_FILES:], "ctx": out}
    if extra:
        record.update(extra)
    _FLUSHED_FILES = len(FILES)
    pickle.dump(record, fh, protocol=pickle.HIGHEST_PROTOCOL)
    fh.flush()


def dump(path):
    M.set_events(TOOL, 0)
    with _orig_open(path, "ab") as fh:
        flush(fh, keep_current=False)


def call_in_ctx(ctx, fn, args, kwargs):
    """Wraps tasks submitted to process pools so the worker attributes lines to the test."""
    if STARTED and ctx and ctx != CTX:
        set_ctx(ctx)
    return fn(*args, **kwargs)


def start_child():
    """Called from sitecustomize in spawned interpreters."""
    global CHILD_MODE
    out = os.environ.get("BR_OUT")
    if not out:
        return
    CHILD_MODE = True
    install_import_hooks()
    start(os.environ.get("BR_CTX") or "misc")

    def _dump():
        if CHILD_MODE:
            try:
                dump(os.path.join(out, f"c-{os.getpid()}.pkl"))
            except Exception as error:  # pragma: no cover
                sys.stderr.write(f"brtrace child dump failed: {error}\n")

    atexit.register(_dump)


# ---------------------------------------------------------------------------
# Import hooks: per-module import contexts and memoization caches disabled.


class NoCacheDict(dict):
    """A dict that never remembers anything, so memoized code runs for every test."""

    def get(self, key, default=None):
        return default

    def __contains__(self, key):
        return False

    def __getitem__(self, key):
        raise KeyError(key)

    def __setitem__(self, key, value):
        pass

    def setdefault(self, key, default=None):
        return default


IMPORT_CTX = False


def _defeat_caches(module):
    name = getattr(module, "__name__", "")
    if not (name == "sqlglot" or name.startswith("sqlglot.")):
        return
    for attr, value in list(vars(module).items()):
        if type(value) is dict and "cache" in attr.lower():
            setattr(module, attr, NoCacheDict())


def install_import_hooks(defeat_caches=True):
    from importlib.machinery import SourceFileLoader, SourcelessFileLoader

    for loader in (SourceFileLoader, SourcelessFileLoader):
        original = loader.exec_module

        def exec_module(self, module, _original=original):
            prev = CTX
            name = getattr(module, "__name__", "")
            switch = IMPORT_CTX and STARTED and (name == "sqlglot" or name.startswith("sqlglot."))
            if switch:
                set_ctx("import|" + name)
            try:
                _original(self, module)
            finally:
                if switch:
                    set_ctx(prev)
                if defeat_caches:
                    _defeat_caches(module)

        loader.exec_module = exec_module
    if defeat_caches:
        for name, module in list(sys.modules.items()):
            _defeat_caches(module)


def import_trace_main(out_path):
    """Imports every sqlglot module, attributing lines to the module being executed."""
    global IMPORT_CTX
    import importlib
    import pkgutil

    IMPORT_CTX = True
    install_import_hooks(defeat_caches=False)
    start("import|<root>")
    import sqlglot

    for info in pkgutil.walk_packages(sqlglot.__path__, prefix="sqlglot."):
        if info.name == "sqlglot.__main__":
            continue
        try:
            importlib.import_module(info.name)
        except Exception as error:
            sys.stderr.write(f"import trace: {info.name} failed: {error}\n")
    dump(out_path)


if __name__ == "__main__":
    import_trace_main(sys.argv[1])
