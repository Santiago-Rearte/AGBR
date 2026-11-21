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
L = R = D = F = I = K = None
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
    return [set(), set(), set(), set(), set(), set()]


def set_ctx(name):
    global L, R, D, F, I, K, CTX
    rec = DATA.get(name)
    if rec is None:
        rec = DATA[name] = _new()
    L, R, D, F, I, K = rec
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
        l, r, d, f, i, k = DATA.pop(name)
        if l or r or d or f or i or k:
            out[name] = (l, r, d, f, _names(i), {(lab, key_canon(x)) for lab, x in k})
    record = {"files_base": _FLUSHED_FILES, "new_files": FILES[_FLUSHED_FILES:], "ctx": out}
    if TRACKED_NEW:
        record["tracked"] = dict(TRACKED_NEW)
        TRACKED_NEW.clear()
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
# Key reads of large class-level dicts (KEYWORDS, FUNCTIONS, TRANSFORMS...).
#
# After a sqlglot module is imported, each such dict that only class namespaces
# refer to is replaced by a TrackDict copy.  Lookups record (label, key); anything
# that can see the whole content (iteration, views, len, copies, merges, equality)
# records (label, ALL).  Dicts aliased elsewhere are left alone and reported as
# untracked, so the analysis never relies on them.

ALL = "\x00*"
MIN_TRACK = 16
MAX_SMALL_NAME = int(os.environ.get("BR_MAX_SMALL_NAME", "50"))
_NAME_COUNTS = None


def _dict_name_counts():
    """How many class bodies in the sqlglot sources assign a dict display to each name."""
    global _NAME_COUNTS
    if _NAME_COUNTS is None:
        import ast

        counts = {}
        for root, _dirs, names in os.walk(os.path.join(REPO, "sqlglot")):
            for fn in names:
                if not fn.endswith(".py"):
                    continue
                try:
                    with _orig_open(os.path.join(root, fn), "rb") as fh:
                        tree = ast.parse(fh.read())
                except Exception:
                    continue
                for node in ast.walk(tree):
                    if not isinstance(node, ast.ClassDef):
                        continue
                    for s in node.body:
                        target = s.targets[0] if isinstance(s, ast.Assign) and len(s.targets) == 1 else getattr(s, "target", None)
                        if isinstance(target, ast.Name) and isinstance(getattr(s, "value", None), ast.Dict):
                            counts[target.id] = counts.get(target.id, 0) + 1
        _NAME_COUNTS = counts
    return _NAME_COUNTS
TRACK_KEYS = os.environ.get("BR_TRACK_KEYS", "1") == "1"
TRACKED = {}  # id(original dict) -> (original, TrackDict); originals kept alive
TRACKED_NEW = {}  # "module:Qual.attr" -> label (None: untracked), not yet flushed


def key_canon(k):
    import enum

    if k is ALL:
        return "*"
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, type):
        return "c:" + k.__name__
    if k is None or isinstance(k, (bool, int, float)):
        return "r:" + repr(k)
    return "o:" + type(k).__name__


def _kread(d, key):
    if K is not None:
        K.add((d._br_label, key))


def _kall(d):
    if K is not None:
        K.add((d._br_label, ALL))


class TrackDict(dict):
    __slots__ = ("_br_label",)

    def __getitem__(self, key):
        if K is not None:
            K.add((self._br_label, key))
        return dict.__getitem__(self, key)

    def get(self, key, default=None):
        if K is not None:
            K.add((self._br_label, key))
        return dict.get(self, key, default)

    def __contains__(self, key):
        if K is not None:
            K.add((self._br_label, key))
        return dict.__contains__(self, key)

    def setdefault(self, key, default=None):
        if K is not None:
            K.add((self._br_label, key))
        return dict.setdefault(self, key, default)

    def pop(self, key, *default):
        if K is not None:
            K.add((self._br_label, key))
        return dict.pop(self, key, *default)

    def __setitem__(self, key, value):
        if K is not None:
            K.add((self._br_label, key))
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        if K is not None:
            K.add((self._br_label, key))
        dict.__delitem__(self, key)

    def __iter__(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__iter__(self)

    def __reversed__(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__reversed__(self)

    def __len__(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__len__(self)

    def keys(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.keys(self)

    def values(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.values(self)

    def items(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.items(self)

    def copy(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict(dict.items(self))

    def __or__(self, other):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__or__(dict(dict.items(self)), other)

    def __ror__(self, other):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__or__(dict(other), dict(dict.items(self)))

    def __ior__(self, other):
        if K is not None:
            K.add((self._br_label, ALL))
        dict.update(self, other)
        return self

    def update(self, *args, **kwargs):
        if K is not None:
            K.add((self._br_label, ALL))
        dict.update(self, *args, **kwargs)

    def popitem(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.popitem(self)

    def clear(self):
        if K is not None:
            K.add((self._br_label, ALL))
        dict.clear(self)

    def __eq__(self, other):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__eq__(self, other)

    def __ne__(self, other):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__ne__(self, other)

    __hash__ = None

    def __repr__(self):
        if K is not None:
            K.add((self._br_label, ALL))
        return dict.__repr__(self)

    def __reduce_ex__(self, protocol):
        if K is not None:
            K.add((self._br_label, ALL))
        return (dict, (dict(dict.items(self)),))

    def __sizeof__(self):
        return dict.__sizeof__(self)


class TrackList(list):
    """A test-class data attribute (e.g. fixture rows loaded by setUpClass): any read
    records (label, ALL) for the running test."""

    __slots__ = ("_br_label",)

    def _r(self):
        if K is not None:
            K.add((self._br_label, ALL))

    def __iter__(self):
        self._r()
        return list.__iter__(self)

    def __reversed__(self):
        self._r()
        return list.__reversed__(self)

    def __getitem__(self, i):
        self._r()
        return list.__getitem__(self, i)

    def __len__(self):
        self._r()
        return list.__len__(self)

    def __contains__(self, x):
        self._r()
        return list.__contains__(self, x)

    def __eq__(self, other):
        self._r()
        return list.__eq__(self, other)

    def __ne__(self, other):
        self._r()
        return list.__ne__(self, other)

    __hash__ = None

    def __repr__(self):
        self._r()
        return list.__repr__(self)

    def __add__(self, other):
        self._r()
        return list.__add__(self, other)

    def __mul__(self, n):
        self._r()
        return list.__mul__(self, n)

    def copy(self):
        self._r()
        return list(list.__iter__(self))

    def index(self, *a):
        self._r()
        return list.index(self, *a)

    def count(self, x):
        self._r()
        return list.count(self, x)

    def __reduce_ex__(self, protocol):
        self._r()
        return (list, (list(list.__iter__(self)),))


def track_test_class(cls):
    """Wraps the list and dict attributes of a test class (after its setup ran), so
    the tests that actually use data loaded by setUpClass/setUp are known."""
    if cls is None or getattr(cls, "_br_tracked", False):
        return
    try:
        cls._br_tracked = True
    except Exception:
        return
    for name, val in list(vars(cls).items()):
        if name.startswith("__") or name == "_br_tracked":
            continue
        label = f"{cls.__module__}:{cls.__qualname__}.{name}"
        try:
            if type(val) is list:
                tv = TrackList(val)
            elif type(val) is dict:
                tv = TrackDict(val)
            else:
                continue
            tv._br_label = label
            setattr(cls, name, tv)
            TRACKED_NEW[label] = label
        except Exception:
            TRACKED_NEW[label] = None


def _module_classes(module, name):
    out, stack = [], [v for v in vars(module).values() if isinstance(v, type) and v.__module__ == name]
    seen = set()
    while stack:
        c = stack.pop()
        if id(c) in seen:
            continue
        seen.add(id(c))
        out.append(c)
        prefix = c.__qualname__ + "."
        for v in list(vars(c).values()):
            if isinstance(v, type) and v.__module__ == name and v.__qualname__.startswith(prefix):
                stack.append(v)
    return out


def track_module_dicts(module):
    import gc

    name = getattr(module, "__name__", "")
    if not (name == "sqlglot" or name.startswith("sqlglot.")):
        return
    try:
        classes = _module_classes(module, name)
    except Exception:
        return
    import enum

    cands = []
    for c in classes:
        if issubclass(c, enum.Enum):
            continue
        for attr, val in list(vars(c).items()):
            if attr.startswith("_"):
                continue
            if type(val) is dict:
                # Small dicts too, unless the name is a per-class schema defined on a
                # great many classes (expression arg_types): those are read whole on
                # every instantiation, so tracking them costs a lot and narrows nothing.
                if len(val) >= MIN_TRACK or _dict_name_counts().get(attr, 0) <= MAX_SMALL_NAME:
                    cands.append((c, attr, val))
            elif type(val) is TrackDict:
                TRACKED_NEW[f"{name}:{c.__qualname__}.{attr}"] = val._br_label
    if not cands:
        return
    namespaces = set()
    for c in classes:
        for x in gc.get_referents(c):
            if type(x) is dict:
                namespaces.add(id(x))
    targets = {id(v): v for _, _, v in cands}
    tl = list(targets.values())
    mine = {id(cands), id(targets), id(tl), id(sys._getframe())} | {id(t) for t in cands}
    aliased = set()
    module_dict = vars(module)
    for r in gc.get_referrers(*tl):
        if id(r) in namespaces or id(r) in mine:
            continue
        if type(r) is tuple and len(r) == len(tl) and all(a is b for a, b in zip(r, tl)):
            continue  # the argument tuple of this very call
        if r is module_dict:
            continue  # module-level aliases are re-pointed below
        for x in gc.get_referents(r):
            if id(x) in targets:
                aliased.add(id(x))
    for c, attr, val in cands:
        key = f"{name}:{c.__qualname__}.{attr}"
        if id(val) in aliased:
            TRACKED_NEW[key] = None
            continue
        hit = TRACKED.get(id(val))
        td = hit[1] if hit is not None and hit[0] is val else None
        if td is None:
            td = TrackDict(val)
            td._br_label = key
            TRACKED[id(val)] = (val, td)
            for g, gv in list(module_dict.items()):
                if gv is val:
                    module_dict[g] = td
        try:
            setattr(c, attr, td)
        except Exception:
            TRACKED_NEW[key] = None
            continue
        TRACKED_NEW[key] = td._br_label


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
            global K
            prev = CTX
            name = getattr(module, "__name__", "")
            ours = name == "sqlglot" or name.startswith("sqlglot.")
            switch = IMPORT_CTX and STARTED and ours
            if switch:
                set_ctx("import|" + name)
            # Dict reads while a sqlglot module executes build import-time state
            # (merged copies, tries), which the run-time state check compares; they
            # are not reads by the test that happened to trigger the import.
            saved_k = K
            if ours:
                K = None
            try:
                _original(self, module)
            finally:
                if ours and K is None:
                    K = saved_k
                if switch:
                    set_ctx(prev)
                if defeat_caches:
                    _defeat_caches(module)
                if TRACK_KEYS:
                    track_module_dicts(module)

        loader.exec_module = exec_module
    if defeat_caches:
        for name, module in list(sys.modules.items()):
            _defeat_caches(module)
    if TRACK_KEYS:
        for name, module in list(sys.modules.items()):
            track_module_dicts(module)


def import_trace_main(out_path):
    """Imports every sqlglot module, attributing lines to the module being executed."""
    global IMPORT_CTX, TRACK_KEYS
    import importlib
    import pkgutil

    IMPORT_CTX = True
    TRACK_KEYS = False
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
