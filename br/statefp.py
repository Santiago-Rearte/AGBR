"""Stable fingerprints of sqlglot's import-time state (class attributes and module globals).

Built once from the base checkout (build_maps) and recomputed in the run-time import
check of a patched tree, so a class-level change can be proven to alter nothing but
the entries it edits: no derived table (tries, inverse mappings, merged dicts under
other names) may differ.

  state(modules) -> {"module:Qual.attr" or "module:name": value}
  value: int hash, or {key canon: int hash} for dicts (entry-wise, so edited entries
  can be excluded from the comparison).
"""

import enum
import functools
import hashlib
import re
import sys
import types

_SKIP_TYPES = (types.ModuleType,)


def key_canon(k):
    """Same canonical key form as the build-time key tracer (brtrace/brcore.py)."""
    if isinstance(k, enum.Enum):
        return f"e:{type(k).__name__}.{k.name}"
    if isinstance(k, str):
        return "s:" + k
    if isinstance(k, type):
        return "c:" + k.__name__
    if k is None or isinstance(k, (bool, int, float)):
        return "r:" + repr(k)
    return "o:" + type(k).__name__


def _canon(x, out, stack, depth):
    t = type(x)
    if x is None or t in (bool, int, float, complex, str, bytes):
        out.append(t.__name__[0] + repr(x))
        return
    if isinstance(x, enum.Enum):
        out.append(f"E{type(x).__module__}.{type(x).__qualname__}.{x.name}")
        return
    if isinstance(x, type):
        out.append(f"T{x.__module__}:{x.__qualname__}")
        return
    if isinstance(x, (types.FunctionType, types.BuiltinFunctionType, types.MethodType)):
        f = getattr(x, "__func__", x)
        out.append(f"F{getattr(f, '__module__', '')}:{getattr(f, '__qualname__', '')}")
        defaults = getattr(f, "__defaults__", None)
        if defaults and depth < 8:
            _canon(defaults, out, stack, depth + 1)
        return
    if isinstance(x, re.Pattern):
        out.append("P" + repr(x.pattern) + str(x.flags))
        return
    if isinstance(x, _SKIP_TYPES):
        out.append("M" + getattr(x, "__name__", ""))
        return
    if id(x) in stack:
        out.append("@")
        return
    if depth > 80:
        out.append("~")
        return
    stack.add(id(x))
    try:
        if isinstance(x, dict):
            items = []
            for k, v in dict.items(x):
                kb, vb = [], []
                _canon(k, kb, stack, depth + 1)
                _canon(v, vb, stack, depth + 1)
                items.append("".join(kb) + "=" + "".join(vb))
            items.sort()
            out.append("{" + ",".join(items) + "}")
        elif isinstance(x, (list, tuple)) and x and all(isinstance(v, type) for v in x):
            # Often tuple(set_of_classes): the order follows id-based hashes and
            # changes from process to process.
            out.append("[c" + ",".join(sorted(f"{v.__module__}:{v.__qualname__}" for v in x)) + "]")
        elif isinstance(x, (list, tuple)):
            out.append("[" if isinstance(x, list) else "(")
            for v in x:
                _canon(v, out, stack, depth + 1)
                out.append(",")
            out.append("]")
        elif isinstance(x, (set, frozenset)):
            parts = []
            for v in x:
                b = []
                _canon(v, b, stack, depth + 1)
                parts.append("".join(b))
            parts.sort()
            out.append("<" + ",".join(parts) + ">")
        elif isinstance(x, (staticmethod, classmethod)):
            _canon(x.__func__, out, stack, depth + 1)
        elif isinstance(x, property):
            for f in (x.fget, x.fset, x.fdel):
                _canon(f, out, stack, depth + 1)
        elif isinstance(x, types.MappingProxyType):
            _canon(dict(x), out, stack, depth + 1)
        elif isinstance(x, functools.partial):
            out.append("p")
            _canon((x.func, x.args, x.keywords), out, stack, depth + 1)
        elif not (t.__module__ or "").startswith("sqlglot"):
            # Foreign objects (loggers, locks, typing constructs...) carry state that
            # depends on the process, not on sqlglot's sources.
            out.append(f"X{t.__module__}.{t.__qualname__}")
        else:
            out.append(f"O{t.__module__}.{t.__qualname__}")
            state = getattr(x, "__dict__", None)
            if state is None and hasattr(t, "__slots__"):
                state = {s: getattr(x, s, None) for s in getattr(t, "__slots__", ()) if isinstance(s, str)}
            if isinstance(state, dict) and depth < 12:
                _canon(state, out, stack, depth + 1)
    finally:
        stack.discard(id(x))


def fp(x):
    out = []
    _canon(x, out, set(), 0)
    return int.from_bytes(hashlib.blake2b("".join(out).encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


def _value(v):
    if type(v) is dict or (isinstance(v, dict) and type(v).__name__ == "TrackDict"):
        entries = {}
        for k, x in dict.items(v):
            kc = key_canon(k)
            h = fp((k, x))
            entries[kc] = entries.get(kc, 0) ^ h
        return entries
    return fp(v)


def _classes(module, name):
    out, stack, seen = [], [v for v in vars(module).values() if isinstance(v, type) and v.__module__ == name], set()
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


def state(names=None):
    """Fingerprints of the given (default: every loaded) sqlglot modules."""
    res = {}
    for name, module in list(sys.modules.items()):
        if not (name == "sqlglot" or name.startswith("sqlglot.")) or module is None:
            continue
        if names is not None and name not in names:
            continue
        for g, v in list(vars(module).items()):
            if g.startswith("__") or isinstance(v, (type, types.ModuleType, types.FunctionType)):
                continue
            try:
                res[f"{name}:{g}"] = _value(v)
            except Exception:
                res[f"{name}:{g}"] = -1
        for c in _classes(module, name):
            for a, v in list(vars(c).items()):
                if a.startswith("__") or isinstance(v, types.FunctionType):
                    continue
                try:
                    res[f"{name}:{c.__qualname__}.{a}"] = _value(v)
                except Exception:
                    res[f"{name}:{c.__qualname__}.{a}"] = -1
    return res


def diff(base, new, allowed, modules, ignore=()):
    """Keys of new (restricted to modules) whose state differs from base.

    allowed: {"module:Qual.attr": set of key canons} entries that may differ.
    ignore: keys whose base value is not reproducible (hash-seed dependent)."""
    bad = []
    ignore = set(ignore)
    for k, v in new.items():
        if k.split(":", 1)[0] not in modules or k in ignore:
            continue
        b = base.get(k)
        if b == v:
            continue
        ok = allowed.get(k)
        if ok is not None and isinstance(v, dict) and isinstance(b, dict):
            if all(v.get(x) == b.get(x) for x in set(v) | set(b) if x not in ok):
                continue
        bad.append(k)
    for k in base:
        if k.split(":", 1)[0] in modules and k not in new and k not in ignore:
            bad.append(k)
    return bad
