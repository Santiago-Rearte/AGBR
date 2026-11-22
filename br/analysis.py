"""Maps a patch to the set of tests whose outcome it could change.

Every result is an int bitset over the traced test indices.  The rules only use
maps built from the pinned commit (per-test line coverage, file reads, directory
listings, instantiated classes, import-time lines) plus the ASTs of the base
and patched sources.  When a rule cannot bound the effect it widens the set.
"""

import ast
import os
import re
import sys

import patchlib

FUNC = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)
DEFS = (ast.FunctionDef, ast.AsyncFunctionDef)
STMTLIKE = (ast.stmt, ast.excepthandler, ast.match_case)
BLOCK_FIELDS = ("body", "orelse", "finalbody", "handlers", "cases")
PYTEST_CONFIG = {"pyproject.toml", "setup.cfg", "tox.ini", "pytest.ini"}
KEY_REFINE = os.environ.get("BR_KEY_REFINE", "1") == "1"
SETUP_REFINE = os.environ.get("BR_SETUP_REFINE", "1") == "1"
DATA_SITES = os.environ.get("BR_DATA_SITES", "1") == "1"  # see analyze()
EQUIV_IS = os.environ.get("BR_EQUIV_IS", "1") == "1"  # `is` -> `==` against a singleton
KEYS_PATH = os.path.join(os.environ.get("BR_MAPS", "/opt/maps"), "keys.pkl")


def log(*args):
    print(*args, file=sys.stderr, flush=True)


def iter_bits(x):
    while x:
        low = x & -x
        yield low.bit_length() - 1
        x ^= low


def _lineno(n):
    if isinstance(n, ast.match_case):
        return n.pattern.lineno
    lo = getattr(n, "lineno", None)
    for d in getattr(n, "decorator_list", ()) or ():
        lo = min(lo, d.lineno) if lo else d.lineno
    return lo


def _end(n):
    if isinstance(n, ast.match_case):
        return n.body[-1].end_lineno if n.body else n.pattern.end_lineno
    return getattr(n, "end_lineno", None) or _lineno(n)


def span(n):
    return (_lineno(n), _end(n))


def header_span(n):
    lo = _lineno(n)
    if lo is None:
        return None
    first = None
    for f in BLOCK_FIELDS:
        lst = getattr(n, f, None)
        if isinstance(lst, list):
            for el in lst:
                ln = _lineno(el)
                if ln is not None and (first is None or ln < first):
                    first = ln
    if first is None:
        return (lo, _end(n))
    return (lo, max(lo, first - 1))


def ast_equal(a, b):
    """Structural equality of two ASTs, ignoring positions (like comparing ast.dump)."""
    stack = [(a, b)]
    while stack:
        x, y = stack.pop()
        if type(x) is not type(y):
            return False
        if isinstance(x, ast.AST):
            for f in x._fields:
                stack.append((getattr(x, f, None), getattr(y, f, None)))
        elif isinstance(x, list):
            if len(x) != len(y):
                return False
            stack.extend(zip(x, y))
        elif x != y:
            return False
    return True


def ast_diff(a, b):
    """Sites where the two trees differ. A site is the list of steps from the root:
    (node, field, index) where index is an int, None, or ("ins", position)."""
    sites = []

    def same(x, y):
        if isinstance(x, ast.AST) and isinstance(y, ast.AST):
            return ast_equal(x, y)
        return type(x) is type(y) and x == y

    def rec(x, y, path):
        if type(x) is not type(y):
            sites.append(path)
            return
        for f in x._fields:
            vx, vy = getattr(x, f, None), getattr(y, f, None)
            if isinstance(vx, list) and isinstance(vy, list):
                rec_list(x, f, vx, vy, path)
            elif isinstance(vx, ast.AST) and isinstance(vy, ast.AST):
                rec(vx, vy, path + [(x, f, None)])
            elif not same(vx, vy):
                sites.append(path + [(x, f, None)])

    def rec_list(x, f, lx, ly, path):
        if len(lx) == len(ly):
            for i, (ex, ey) in enumerate(zip(lx, ly)):
                if isinstance(ex, ast.AST) and isinstance(ey, ast.AST):
                    rec(ex, ey, path + [(x, f, i)])
                elif not same(ex, ey):
                    sites.append(path + [(x, f, i)])
            return
        p = 0
        while p < len(lx) and p < len(ly) and same(lx[p], ly[p]):
            p += 1
        s = 0
        while s < len(lx) - p and s < len(ly) - p and same(lx[-1 - s], ly[-1 - s]):
            s += 1
        mid_x = lx[p : len(lx) - s]
        mid_y = ly[p : len(ly) - s]
        k = min(len(mid_x), len(mid_y))
        for j in range(k):
            ex, ey = mid_x[j], mid_y[j]
            if isinstance(ex, ast.AST) and isinstance(ey, ast.AST):
                rec(ex, ey, path + [(x, f, p + j)])
            else:
                sites.append(path + [(x, f, p + j)])
        for j in range(k, len(mid_x)):
            sites.append(path + [(x, f, p + j)])
        if len(mid_y) > k:
            sites.append(path + [(x, f, ("ins", p + k))])

    rec(a, b, [])
    return sites


def bound_names(stmt):
    """Names a module- or class-level statement binds (not descending into functions)."""
    names = set()

    def targets(t):
        for n in ast.walk(t):
            if isinstance(n, ast.Name):
                names.add(n.id)
            elif isinstance(n, ast.Starred) and isinstance(n.value, ast.Name):
                names.add(n.value.id)

    def visit(s):
        if isinstance(s, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.add(s.name)
            return
        if isinstance(s, ast.Assign):
            for t in s.targets:
                targets(t)
        elif isinstance(s, (ast.AnnAssign, ast.AugAssign)):
            targets(s.target)
        elif isinstance(s, (ast.Import, ast.ImportFrom)):
            for a in s.names:
                names.add((a.asname or a.name).split(".")[0])
        elif isinstance(s, (ast.For, ast.AsyncFor)):
            targets(s.target)
        elif isinstance(s, (ast.With, ast.AsyncWith)):
            for it in s.items:
                if it.optional_vars is not None:
                    targets(it.optional_vars)
        elif isinstance(s, ast.TypeAlias) if hasattr(ast, "TypeAlias") else False:
            targets(s.name)
        for f in BLOCK_FIELDS:
            for child in getattr(s, f, None) or []:
                if isinstance(child, ast.stmt):
                    visit(child)
                elif isinstance(child, (ast.excepthandler, ast.match_case)):
                    if isinstance(child, ast.ExceptHandler) and child.name:
                        names.add(child.name)
                    for c in child.body:
                        visit(c)
        if isinstance(s, ast.stmt):
            for n in ast.walk(s):
                if isinstance(n, ast.NamedExpr) and isinstance(n.target, ast.Name):
                    names.add(n.target.id)

    visit(stmt)
    return names


def modname(rel):
    if not rel.endswith(".py"):
        return None
    parts = rel[:-3].split("/")
    if parts[-1] == "__init__":
        parts = parts[:-1]
    return ".".join(parts)


class Analyzer:
    def __init__(self, maps, repo):
        self.m = maps
        self.repo = repo
        self.sets = maps["sets"]
        self.lines = maps["lines"]
        self.ntests = len(maps["tests"])
        self.all_bits = (1 << self.ntests) - 1
        self._ast = {}
        self._src = {}
        self._file_bits = {}
        self._corpus = None
        self._family_bits = {}
        self._memo = {}
        self.import_lines = maps.get("import_lines", {})
        self.mod_classes = maps.get("mod_classes", {})
        self.eager = set(maps.get("eager", ()))
        self.trace = {"runtime": [], "other": False, "import": False, "nonsite": False, "import_mods": set(), "keyref": []}
        self.keys = None
        self._new_tree = None

    def release(self):
        """Forgets cached ASTs and sources (they are rebuilt on demand)."""
        self._ast.clear()
        self._src.clear()
        self._corpus = None
        self._memo.clear()

    # -- primitive lookups --------------------------------------------------

    def source(self, rel):
        s = self._src.get(rel)
        if s is None:
            try:
                with open(os.path.join(self.repo, rel), "rb") as fh:
                    s = fh.read()
            except OSError:
                s = b""
            self._src[rel] = s
        return s

    def tree(self, rel):
        t = self._ast.get(rel)
        if t is None:
            try:
                t = ast.parse(self.source(rel))
            except (SyntaxError, ValueError):
                t = ast.Module(body=[], type_ignores=[])
            self._ast[rel] = t
        return t

    def lines_bits(self, rel, lo, hi):
        d = self.lines.get(rel)
        if not d or lo is None:
            return 0
        bits = 0
        sets = self.sets
        if hi - lo > 400:
            for ln, sid in d.items():
                if lo <= ln <= hi:
                    bits |= sets[sid]
        else:
            for ln in range(lo, hi + 1):
                sid = d.get(ln)
                if sid is not None:
                    bits |= sets[sid]
        return bits

    def file_bits(self, rel):
        b = self._file_bits.get(rel)
        if b is None:
            b = 0
            for sid in set((self.lines.get(rel) or {}).values()):
                b |= self.sets[sid]
            self._file_bits[rel] = b
        return b

    def reads(self, rel):
        bits = self.m["reads"].get(rel, 0)
        if rel.startswith("tests/") and rel.endswith(".py") and "/fixtures/" not in rel:
            # pytest reads every test module's own source while collecting it (assert
            # rewriting), which would make each test a "reader" of its own file.  Its
            # own tests are covered by line coverage of the changed statement instead.
            bits &= ~self.m["by_file"].get(rel, 0)
        groups = (self.m.get("group_reads") or {}).get(rel)
        if groups and SETUP_REFINE and not rel.endswith(".py") and "own_reads" in self.m:
            # Read by shared setup (class setup, module collection): only the tests
            # that use what the setup loaded from it.
            refined = self.m["own_reads"].get(rel, 0)
            for g in groups:
                kind, _, key = g.partition(":")
                users = self.setup_data_users(key, rel) if kind == "class" else None
                if users is None:
                    users = self.m["by_class" if kind == "class" else "by_file"].get(key, 0)
                refined |= users
            bits &= refined
        return bits

    def dirs(self, rel_dir):
        return self.m["dirs"].get(rel_dir.rstrip("/"), 0)

    def family_bits(self, key):
        b = self._family_bits.get(key)
        if b is None:
            b = 0
            inst = self.m["inst"]
            for k in self.m["family"].get(key, [key]):
                b |= inst.get(k, 0)
                b |= self._test_class_bits(k)
            self._family_bits[key] = b
        return b

    def _test_class_bits(self, key):
        mod, _, qual = key.partition(":")
        if not mod.startswith("tests"):
            return 0
        rel = mod.replace(".", "/") + ".py"
        return self.m["by_class"].get(f"{rel}::{qual}", 0)

    def tests_under(self, rel_dir):
        prefix = rel_dir.rstrip("/") + "/" if rel_dir else ""
        bits = 0
        for f, b in self.m["by_file"].items():
            if f.startswith(prefix):
                bits |= b
        return bits

    def corpus(self):
        if self._corpus is None:
            files = {}
            for top in ("sqlglot", "tests", "sqlglotc"):
                base = os.path.join(self.repo, top)
                for root, _dirs, names in os.walk(base):
                    for name in names:
                        if name.endswith(".py"):
                            p = os.path.join(root, name)
                            rel = os.path.relpath(p, self.repo).replace(os.sep, "/")
                            try:
                                with open(p, "rb") as fh:
                                    files[rel] = fh.read().decode("utf-8", "replace")
                            except OSError:
                                pass
            self._corpus = files
        return self._corpus

    # -- semantic units ---------------------------------------------------

    def module_wide(self, rel):
        key = ("module_wide", rel)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = 0
        bits = self.file_bits(rel) | self.reads(rel) | self.m["by_file"].get(rel, 0)
        mod = modname(rel)
        for ckey in self.mod_classes.get(mod, ()):
            bits |= self.family_bits(ckey)
        if mod and mod.startswith("tests"):
            bits |= self.importers_bits(mod, rel)
        self._memo[key] = bits
        return bits

    def importers_bits(self, mod, rel):
        """Tests in files that import the given (test-side) module."""
        last = mod.rsplit(".", 1)[-1]
        pat = re.compile(r"(?:from\s+" + re.escape(mod) + r"\s+import|import\s+" + re.escape(mod) + r"\b|from\s+"
                         + re.escape(mod.rsplit(".", 1)[0]) + r"\s+import[^\n]*\b" + re.escape(last) + r"\b)")
        bits = 0
        for f, text in self.corpus().items():
            if f != rel and pat.search(text):
                bits |= self.m["by_file"].get(f, 0) | self.module_wide(f)
        return bits

    def import_time_bits(self, rel, units):
        il = self.import_lines.get(rel)
        if not il:
            return 0
        mods = set()
        for lo, hi in units:
            if lo is None:
                continue
            for ln in range(lo, hi + 1):
                ms = il.get(ln)
                if ms:
                    mods.update(ms)
        # These modules run the changed code while being imported: the run-time
        # import check must import them, since an exception there breaks every user.
        self.trace.setdefault("import_mods", set()).update(mods)
        bits = 0
        for mod in mods:
            mrel = self.m["mod_files"].get(mod)
            if mrel:
                bits |= self.module_wide(mrel)
        return bits

    def units_bits(self, rel, units):
        bits = 0
        for lo, hi in units:
            if lo is not None:
                bits |= self.lines_bits(rel, lo, hi)
        return bits | self.import_time_bits(rel, units)

    def block_units(self, n, f, i):
        lst = getattr(n, f, None)
        if not isinstance(lst, list):
            return [span(n)]
        if isinstance(i, tuple):
            pos = i[1]
            sibs = []
            if pos > 0 and pos - 1 < len(lst):
                sibs.append(lst[pos - 1])
            if pos < len(lst):
                sibs.append(lst[pos])
            units = [span(s) for s in sibs]
            return units or [header_span(n) or span(n)]
        if isinstance(i, int) and i < len(lst):
            el = lst[i]
            if isinstance(el, ast.AST) and getattr(el, "lineno", None) is not None or isinstance(el, ast.match_case):
                return [span(el)]
        return [span(n)]

    def site_bits(self, rel, site, depth=0):
        """Tests that could observe a change at this site of the old tree of rel."""
        jf = None
        for j, (n, f, _i) in enumerate(site):
            if isinstance(n, FUNC) and f == "body":
                jf = j
        if jf is not None:
            inner = [s for s in site[jf + 1 :] if isinstance(s[0], STMTLIKE)]
            if not inner:
                n, f, i = site[jf]
                units = [span(n)] if isinstance(n, ast.Lambda) else self.block_units(n, f, i)
            else:
                n, f, i = inner[-1]
                if f in BLOCK_FIELDS and isinstance(getattr(n, f, None), list):
                    units = self.block_units(n, f, i)
                else:
                    units = [header_span(n) or span(n)]
            if depth == 0:
                self.trace["runtime"].append((rel, units, span(site[jf][0])))
            return self.units_bits(rel, units) | self.reads(rel)

        if depth == 0:
            self.trace["other"] = True
            self.trace["import"] = True
        # Import-time site: module or class body, or a def/class header.
        classes = []
        hdr = None
        cls_hdr = None
        for n, f, i in site:
            if isinstance(n, ast.ClassDef):
                if f == "body":
                    classes.append(n)
                else:
                    cls_hdr = n
            elif isinstance(n, DEFS) and f != "body":
                hdr = (n, f)
        bits = 0
        if hdr is not None:
            fn, field = hdr
            if fn.body:
                bits |= self.units_bits(rel, [(_lineno(fn.body[0]), _end(fn))])
            if field not in ("name", "decorator_list", "type_params"):
                return bits | self.reads(rel)
        if cls_hdr is not None:
            return bits | self.class_level(rel, classes + [cls_hdr], None, depth, header=True)
        stmt = self._changed_stmt(site, classes)
        if classes:
            return bits | self.class_level(rel, classes, stmt, depth, site=site)
        return bits | self.module_level(rel, stmt, depth)

    def _changed_stmt(self, site, classes):
        """The class- or module-level statement that contains the site."""
        owner = classes[-1] if classes else None
        for idx, (n, f, i) in enumerate(site):
            if (owner is None and isinstance(n, ast.Module)) or n is owner:
                if f == "body":
                    lst = n.body
                    if isinstance(i, int) and i < len(lst):
                        return lst[i]
                    if isinstance(i, tuple):
                        return ("ins", n, i[1])
        return None

    def family_bits_excluding(self, key, names):
        """Users of a class family, leaving out subclasses (and their descendants)
        that redefine every one of names without mentioning them: attribute lookup on
        those classes never reaches the changed definition."""
        fam = self.m["family"].get(key, [key])
        if not names or len(fam) > 400:
            return self.family_bits(key), fam
        excluded = set()
        for sub in fam:
            if sub == key or sub in excluded:
                continue
            smod, _, squal = sub.partition(":")
            srel = self.m["mod_files"].get(smod)
            node = self._find_class(srel, squal) if srel else None
            if node is None:
                continue
            redefined = set()
            for st in node.body:
                if isinstance(st, (ast.Assign, ast.AnnAssign)) and st.value is not None:
                    bound = bound_names(st) & names
                    if not bound:
                        continue
                    mentions = any(
                        (isinstance(x, ast.Name) and x.id in names)
                        or (isinstance(x, ast.Attribute) and x.attr in names)
                        or (isinstance(x, ast.Constant) and x.value in names)
                        for x in ast.walk(st.value)
                    )
                    if not mentions:
                        redefined |= bound
            if redefined >= names:
                excluded.update(self.m["family"].get(sub, [sub]))
        kept = [k for k in fam if k not in excluded]
        inst = self.m["inst"]
        bits = 0
        for k in kept:
            bits |= inst.get(k, 0) | self._test_class_bits(k)
        return bits, kept

    def class_level(self, rel, chain, stmt, depth, header=False, site=None):
        bits = self._class_level(rel, chain, stmt, depth, header)
        if depth == 0 and site is not None and KEY_REFINE:
            try:
                refined = self.key_refine(rel, chain, stmt, site)
            except Exception as error:  # never let the refinement crash the analysis
                log(f"  key refinement error on {rel}: {error!r}")
                refined = None
            if refined is not None:
                rbits, info = refined
                info["fallback"] = bits
                self.trace["keyref"].append(info)
                return rbits | self.reads(rel)
        return bits

    def _class_level(self, rel, chain, stmt, depth, header=False):
        cls = chain[-1]
        mod = modname(rel) or ""
        qual = ".".join(c.name for c in chain)
        key = f"{mod}:{qual}"
        attr_names = set()
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            attr_names = {n for n in bound_names(stmt) if n != "_"}
        fam_bits, fam = self.family_bits_excluding(key, attr_names)
        users = fam_bits | self.func_lines_bits(rel, cls)
        # A class is often used through the classes that hold it (a dialect's
        # tokenizer_class, parser_class...) without ever being instantiated.
        users |= self.holder_bits(rel, chain)
        # Subclasses defined elsewhere may use the attribute without being instantiated.
        if len(fam) <= 40:
            for sub in fam:
                smod, _, squal = sub.partition(":")
                srel = self.m["mod_files"].get(smod)
                if not srel:
                    continue
                node = self._find_class(srel, squal)
                if node is not None:
                    users |= self.func_lines_bits(srel, node)
        names = set()
        bits = self.reads(rel)
        if isinstance(stmt, ast.AST):
            names = bound_names(stmt)
            bits |= self.func_lines_bits(rel, stmt)
            if isinstance(stmt, (ast.Assign, ast.AnnAssign, ast.AugAssign)) and names:
                bounded = self.attr_readers(names, rel, stmt)
                if bounded is not None:
                    readers, copiers = bounded
                    users = (users | copiers) & readers
        elif isinstance(stmt, tuple):
            _, owner, pos = stmt
            for s in owner.body[max(0, pos - 1) : pos + 1]:
                names |= bound_names(s)
        bits |= users
        if depth < 2:
            pats = [re.escape(cls.name) + r"\." + re.escape(n) + r"\b" for n in names]
            if header:
                pats.append(r"\b" + re.escape(cls.name) + r"\b")
            if pats:
                bits |= self.refs_bits(re.compile("|".join(pats)), rel, depth + 1, skip_span=span(cls),
                    literal=[cls.name] if header else [f"{cls.name}.{n}" for n in names],
                )
        return bits

    def holder_bits(self, rel, chain):
        """Users of the classes that contain or refer to the innermost class of chain:
        enclosing classes, and classes whose body names it (in this module or in
        modules importing it).  Applied transitively, a few levels deep."""
        key = ("holders", rel, ".".join(c.name for c in chain))
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = 0
        mod = modname(rel) or ""
        bits = 0
        holders = set()
        for k in range(1, len(chain)):
            holders.add((rel, ".".join(c.name for c in chain[:k])))
        name = chain[-1].name
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        files = [rel] + [f for f in self.importer_files(rel) if f != rel] if len(chain) == 1 else [rel]
        for f in files:
            text = self.corpus().get(f)
            if text is None or name not in text or not pat.search(text):
                continue
            for ln in {text.count("\n", 0, m.start()) + 1 for m in pat.finditer(text)}:
                path = self.locate(f, ln)
                if not path or any(isinstance(n, FUNC) and fl == "body" for n, fl, _i in path):
                    continue
                classes = [n for n, fl, _i in path if isinstance(n, ast.ClassDef) and fl == "body"]
                if not classes:
                    continue
                if f == rel and [c.name for c in classes] == [c.name for c in chain]:
                    continue  # the class's own body
                # Only aliases (`Parser = SnowflakeParser`, `tokenizer_class = Tokenizer`)
                # make the holder hand the class itself out.  Mentions inside dicts or
                # calls dispatch on instances, which the class family already covers.
                stmt = next((n for n, fl, _i in path if fl == "__ref__"), None)
                if not (
                    isinstance(stmt, (ast.Assign, ast.AnnAssign))
                    and isinstance(stmt.value, (ast.Name, ast.Attribute))
                    and (stmt.value.id if isinstance(stmt.value, ast.Name) else stmt.value.attr) == name
                ):
                    continue
                holders.add((f, ".".join(c.name for c in classes)))
        for f, qual in holders:
            hkey = f"{modname(f)}:{qual}"
            bits |= self.family_bits(hkey)
            node = self._find_class(f, qual)
            if node is not None:
                bits |= self.func_lines_bits(f, node)
                hchain = []
                body = self.tree(f).body
                for part in qual.split("."):
                    c = next((s for s in body if isinstance(s, ast.ClassDef) and s.name == part), None)
                    if c is None:
                        break
                    hchain.append(c)
                    body = c.body
                if hchain and len(holders) < 50:
                    bits |= self.holder_bits(f, hchain)
        self._memo[key] = bits
        return bits

    def attr_readers(self, names, origin_rel, stmt):
        """(readers, copiers) for a class attribute, or None when it cannot be bounded.

        readers: tests that execute code mentioning one of names (plus, when that code
        also runs while some module is imported, every test depending on that module).
        copiers: users of other classes whose body copies the attribute under the same
        name (`X = {**Base.X, ...}`); run-time reads of the name cover those copies.
        None when a mention feeds import-time state under another name."""
        key = ("readers", tuple(sorted(names)), origin_rel, _lineno(stmt))
        if key in self._memo:
            return self._memo[key]
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        lo0, hi0 = span(stmt)
        readers = 0
        copiers = 0
        hits = 0
        result = None
        literals = sorted(names)
        for rel, text in self.corpus().items():
            if not any(n in text for n in literals) or not pat.search(text):
                continue
            lines_hit = sorted({text.count("\n", 0, m.start()) + 1 for m in pat.finditer(text)})
            hits += len(lines_hit)
            if hits > 3000:
                break
            seen_stmts = set()
            ok = True
            for ln in lines_hit:
                if rel == origin_rel and lo0 <= ln <= hi0:
                    continue
                path = self.locate(rel, ln)
                if not path:
                    ok = False
                    break
                node = next(n for n, f, _i in path if f == "__ref__")
                lam = path[-1][0] if isinstance(path[-1][0], ast.Lambda) else None
                if id(lam or node) in seen_stmts:
                    continue
                seen_stmts.add(id(lam or node))
                in_func = any(isinstance(n, FUNC) and f == "body" for n, f, _i in path)
                if lam is not None or in_func:
                    hs = span(lam) if lam is not None else (header_span(node) or span(node))
                    readers |= self.units_bits(rel, [hs])
                    continue
                if isinstance(node, (ast.Assign, ast.AnnAssign)):
                    value = node.value
                    mentioned = value is not None and any(
                        (isinstance(x, ast.Name) and x.id in names)
                        or (isinstance(x, ast.Attribute) and x.attr in names)
                        or (isinstance(x, ast.Constant) and x.value in names)
                        for x in ast.walk(value)
                    )
                    if not mentioned:
                        continue  # another class defining the same attribute
                    classes = [n for n, f, _i in path if isinstance(n, ast.ClassDef) and f == "body"]
                    if classes and bound_names(node) <= set(names):
                        # A subclass copying the attribute under the same name.
                        qual = ".".join(c.name for c in classes)
                        copiers |= self.family_bits(f"{modname(rel)}:{qual}")
                        copiers |= self.func_lines_bits(rel, classes[-1])
                        continue
                ok = False
                break
            if not ok:
                break
        else:
            result = (readers, copiers)
        self._memo[key] = result
        return result

    # -- key-level refinement of class dict entries ------------------------------

    def keydata(self):
        if self.keys is None:
            import pickle

            try:
                with open(KEYS_PATH, "rb") as fh:
                    self.keys = pickle.load(fh)
            except Exception:
                self.keys = {}
        return self.keys

    @staticmethod
    def static_key(k):
        """Canonical forms (as brcore.key_canon writes them) a dict key expression
        may take at run time, or None when unknown."""
        if isinstance(k, ast.Constant):
            v = k.value
            if isinstance(v, str):
                return frozenset({"s:" + v})
            if v is None or isinstance(v, (bool, int, float)):
                return frozenset({"r:" + repr(v)})
            return None
        if isinstance(k, ast.Attribute):
            out = {"c:" + k.attr}
            base = k.value
            bname = base.attr if isinstance(base, ast.Attribute) else base.id if isinstance(base, ast.Name) else None
            if bname:
                out.add(f"e:{bname}.{k.attr}")
            return frozenset(out)
        return None

    def _display(self, d):
        spreads, final = [], {}
        for k, v in zip(d.keys, d.values):
            if k is None:
                spreads.append(v)
                continue
            kc = self.static_key(k)
            if kc is None:
                # A key known only at run time (a name): fine while its entry is
                # untouched; an edit to it gives up (see key_refine).
                kc = frozenset({"?" + ast.dump(k)})
            final[kc] = v
        return spreads, final

    def _follow_new(self, site, pos):
        cur = self._new_tree
        for _n, f, i in site[:pos]:
            nxt = getattr(cur, f, None)
            if isinstance(i, int):
                if not isinstance(nxt, list) or i >= len(nxt):
                    return None
                nxt = nxt[i]
            elif i is not None:
                return None
            cur = nxt
            if cur is None:
                return None
        return cur

    def mentioning_modules(self, name):
        pat = re.compile(r"\b" + re.escape(name) + r"\b")
        out = set()
        for rel, text in self.corpus().items():
            if rel.startswith("sqlglot/") and pat.search(text):
                mod = modname(rel)
                if mod:
                    out.add(mod)
        return out

    def key_refine(self, rel, chain, stmt, site):
        """Tests that read an edited key of a class-level dict display.

        Applies when the change lies inside the entries of `ATTR = {...}`: the
        readers are the tests that looked those keys up (or saw the whole dict) in
        any tracked dict of that name.  Returns (bits, info) or None; info tells the
        run-time import check which state may differ, and the selector falls back to
        the unrefined set when anything else changed."""
        if self._new_tree is None or not isinstance(stmt, (ast.Assign, ast.AnnAssign)):
            return None
        if not isinstance(stmt.value, ast.Dict):
            return None
        names = bound_names(stmt)
        if len(names) != 1:
            return None
        attr = next(iter(names))
        pos = next((j for j, (n, _f, _i) in enumerate(site) if n is stmt), None)
        if pos is None or site[pos][1] != "value":
            return None
        new_stmt = self._follow_new(site, pos)
        if type(new_stmt) is not type(stmt) or not isinstance(new_stmt.value, ast.Dict):
            return None
        if bound_names(new_stmt) != names:
            return None
        if isinstance(stmt, ast.AnnAssign) and not ast_equal(stmt.annotation, new_stmt.annotation):
            return None
        old_d, new_d = self._display(stmt.value), self._display(new_stmt.value)
        if old_d is None or new_d is None:
            return None
        if len(old_d[0]) != len(new_d[0]) or not all(ast_equal(a, b) for a, b in zip(old_d[0], new_d[0])):
            return None
        changed = set()
        for kc in set(old_d[1]) | set(new_d[1]):
            a, b = old_d[1].get(kc), new_d[1].get(kc)
            if a is None or b is None or not ast_equal(a, b):
                changed |= kc
        if any(x.startswith("?") for x in changed):
            return None
        kd = self.keydata()
        base = kd.get("state") or {}
        tracked = kd.get("tracked") or {}
        if not base or not tracked:
            return None
        mod = modname(rel) or ""
        qual = ".".join(c.name for c in chain)
        own = f"{mod}:{qual}.{attr}"
        if not tracked.get(own) or not isinstance(base.get(own), dict):
            return None
        fam = set(self.m.get("family", {}).get(f"{mod}:{qual}", [f"{mod}:{qual}"]))
        suffix = "." + attr
        same_name = [k for k, v in base.items() if k.endswith(suffix) and isinstance(v, dict) and "." in k.split(":", 1)[1]]
        own_base = base[own]
        owners = []
        for k in same_name:
            # Dicts that can hold the edited entries: the class's own, its subclasses'
            # (inherited or merged copies), and any other dict whose entry for an
            # edited key is identical to this one's (a copy made elsewhere).
            # (A subclass defining its own unrelated dict is not one: if it changed
            # anyway, the run-time state check sees it and falls back.)
            if k == own or (k[: -len(suffix)] in fam and set(own_base) <= set(base[k])) or any(
                x in own_base and base[k].get(x) == own_base[x] for x in changed
            ):
                if not tracked.get(k):
                    return None  # a copy that may hold an edited key is not tracked
                owners.append(k)
        reads, sets = kd["reads"], kd["sets"]
        bits = 0
        for label in {tracked[k] for k in owners if tracked.get(k)}:
            r = reads.get(label) or {}
            for kc in changed | {"*"}:
                sid = r.get(kc)
                if sid is not None:
                    bits |= sets[sid]
        # Lambdas and functions inside the edited entries.
        for k, v in zip(stmt.value.keys, stmt.value.values):
            kc = self.static_key(k) if k is not None else None
            if kc and kc & changed:
                bits |= self.func_lines_bits(rel, v)
        modules = {k.split(":", 1)[0] for k in owners} | {c.split(":", 1)[0] for c in fam}
        modules |= self.mentioning_modules(attr)
        info = {
            "allowed": {k: sorted(changed) for k in owners},
            "modules": sorted(m for m in modules if m),
            "attr": attr,
            "keys": sorted(changed),
        }
        return bits, info


    # -- data read by shared test setup ------------------------------------------

    def setup_data_users(self, class_key, rel):
        """Tests of a test class that use the data its setup methods loaded from rel
        (None when unsure; see setup_data_attrs)."""
        attrs = self.setup_data_attrs(class_key, rel)
        if not attrs:
            return None
        f, _, qual = class_key.partition("::")
        kd = self.keydata()
        tracked, reads, sets = kd.get("tracked") or {}, kd.get("reads") or {}, kd.get("sets") or []
        mod = modname(f) or ""
        cqual = qual.replace("::", ".")
        bits = 0
        for a in attrs:
            label = f"{mod}:{cqual}.{a}"
            if not tracked.get(label):
                return None
            for sid in (reads.get(label) or {}).values():
                bits |= sets[sid]
        return bits

    def setup_data_attrs(self, class_key, rel):
        """The class attributes a test class's setup methods load from rel.

        Data flow inside the class's setup/teardown methods: a statement naming the
        file (a string constant rel ends with) taints what it assigns (or the objects
        it calls methods on); a statement reading a tainted name or class attribute
        taints its own targets.  The users are the tests that read a tainted class
        attribute at build time (brcore.track_test_class).  None when unsure: no
        statement names the file, or a tainted class attribute was not tracked."""
        f, _, qual = class_key.partition("::")
        try:
            tree = self.tree(f)
        except Exception:
            return None
        body, cls = tree.body, None
        for part in qual.split("::"):
            cls = next((s for s in body if isinstance(s, ast.ClassDef) and s.name == part), None)
            if cls is None:
                return None
            body = cls.body
        methods = [s for s in cls.body if isinstance(s, DEFS) and s.name.lower().startswith(("setup", "teardown"))]
        stmts = []
        for m in methods:
            for n in ast.walk(m):
                if isinstance(n, (ast.Assign, ast.AnnAssign, ast.AugAssign, ast.Expr, ast.For, ast.With, ast.Return)):
                    stmts.append(n)

        def syms(node):
            out = set()
            if node is None:
                return out
            for x in ast.walk(node):
                if isinstance(x, ast.Attribute) and isinstance(x.value, ast.Name) and x.value.id in ("cls", "self"):
                    out.add(("attr", x.attr))
                elif isinstance(x, ast.Name) and x.id not in ("cls", "self"):
                    out.add(("name", x.id))
            return out

        def base_syms(t):
            # The symbol a target writes into: x, cls.x, x[...], cls.x[...], x.y = ...
            while isinstance(t, (ast.Subscript, ast.Starred)) or (
                isinstance(t, ast.Attribute) and not (isinstance(t.value, ast.Name) and t.value.id in ("cls", "self"))
            ):
                t = t.value
            if isinstance(t, (ast.Tuple, ast.List)):
                return set().union(*(base_syms(e) for e in t.elts)) if t.elts else set()
            return syms(t) if isinstance(t, (ast.Name, ast.Attribute)) else set()

        def flow(s):
            """(targets, sources) of a statement."""
            if isinstance(s, ast.Assign):
                return set().union(*(base_syms(t) for t in s.targets)), syms(s.value)
            if isinstance(s, ast.AnnAssign):
                return base_syms(s.target), syms(s.value)
            if isinstance(s, ast.AugAssign):
                return base_syms(s.target), syms(s.value) | base_syms(s.target)
            if isinstance(s, ast.For):
                return base_syms(s.target), syms(s.iter)
            if isinstance(s, ast.With):
                tg = set()
                for it in s.items:
                    tg |= base_syms(it.optional_vars) if it.optional_vars is not None else set()
                return tg, set().union(*(syms(it.context_expr) for it in s.items))
            if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call):
                # obj.method(args): the receiver may store what the arguments carry.
                func = s.value.func
                tg = base_syms(func.value) if isinstance(func, ast.Attribute) else set()
                src = set().union(*(syms(a) for a in s.value.args), *(syms(k.value) for k in s.value.keywords))
                return tg, src
            return set(), syms(s)

        def names_file(node):
            for x in ast.walk(node):
                if isinstance(x, ast.Constant) and isinstance(x.value, str) and len(x.value) >= 6:
                    v = x.value.lstrip("./")
                    if ("/" in v or "." in v) and rel.endswith(v):
                        return True
            return False

        tainted = set()
        for s in stmts:
            if names_file(s):
                tg, _src = flow(s)
                if not tg:
                    return None  # the file feeds something we cannot follow
                tainted |= tg
        if not tainted:
            return None
        changed = True
        while changed:
            changed = False
            for s in stmts:
                tg, src = flow(s)
                if src & tainted and not tg:
                    if isinstance(s, (ast.Expr, ast.Return)):
                        return None  # handed to a call or returned: not followed
                if src & tainted and not tg <= tainted:
                    tainted |= tg
                    changed = True
        attrs = sorted(a for kind, a in tainted if kind == "attr")
        return attrs or None

    def func_lines_bits(self, rel, node):
        """Tests that ran any function or lambda defined inside node.

        Module and class bodies run at import time, in whichever process imports the
        module first (often a spawned helper), so their own lines say nothing."""
        bits = 0
        for n in ast.walk(node):
            if isinstance(n, DEFS) and n.body:
                bits |= self.units_bits(rel, [(_lineno(n.body[0]), _end(n))])
            elif isinstance(n, ast.Lambda):
                bits |= self.units_bits(rel, [span(n.body)])
        return bits

    def _find_class(self, rel, qual):
        body = self.tree(rel).body
        node = None
        for part in qual.split("."):
            node = next((s for s in body if isinstance(s, ast.ClassDef) and s.name == part), None)
            if node is None:
                return None
            body = node.body
        return node

    def module_level(self, rel, stmt, depth):
        if stmt is None:
            return self.module_wide(rel)
        if isinstance(stmt, tuple):
            _, owner, pos = stmt
            bits = self.reads(rel)
            for s in owner.body[max(0, pos - 1) : pos + 1]:
                bits |= self.func_lines_bits(rel, s)
            # A new module-level statement can do anything at import time.
            return bits | self.module_wide(rel)
        names = {n for n in bound_names(stmt) if n != "_"}
        bits = self.func_lines_bits(rel, stmt) | self.reads(rel)
        if not names or isinstance(stmt, (ast.If, ast.Try, ast.With, ast.For, ast.While, ast.Expr)):
            bits |= self.module_wide(rel)
        if not names or depth >= 3:
            return bits
        pat = re.compile(r"\b(?:" + "|".join(re.escape(n) for n in sorted(names)) + r")\b")
        bits |= self.refs_in_file(rel, pat, depth + 1, skip_span=span(stmt))
        if depth > 0:
            return bits
        if isinstance(stmt, (ast.Import, ast.ImportFrom)):
            files = self.importer_files(rel)
        else:
            files = [f for f, text in self.corpus().items() if f != rel and pat.search(text)]
        if len(files) > 80:
            log(f"  {rel}: {sorted(names)[:5]} referenced in {len(files)} files, widening")
            return bits | self.all_bits
        for f in files:
            bits |= self.refs_in_file(f, pat, depth + 1)
        return bits

    def importer_files(self, rel):
        mod = modname(rel) or ""
        key = ("importers", mod)
        if key in self._memo:
            return self._memo[key]
        parent, _, last = mod.rpartition(".")
        pats = [r"from\s+" + re.escape(mod) + r"\s+import", r"import\s+" + re.escape(mod) + r"\b"]
        if parent:
            pats.append(r"from\s+" + re.escape(parent) + r"\s+import[^\n]*\b" + re.escape(last) + r"\b")
        pat = re.compile("|".join(pats))
        files = [
            f
            for f, text in self.corpus().items()
            if f != rel and (mod in text or (parent and parent in text and last in text)) and pat.search(text)
        ]
        self._memo[key] = files
        return files

    def refs_bits(self, pattern, origin_rel, depth, skip_span=None, literal=None):
        """Tests that execute code matching pattern anywhere in the repo.

        literal: substrings one of which every match contains (a cheap prefilter)."""
        key = ("refs", pattern.pattern, origin_rel, depth)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = 0
        bits = 0
        for rel, text in self.corpus().items():
            if literal is not None and not any(x in text for x in literal):
                continue
            if pattern.search(text):
                bits |= self.refs_in_file(rel, pattern, depth, skip_span if rel == origin_rel else None)
        self._memo[key] = bits
        return bits

    def refs_in_file(self, rel, pattern, depth, skip_span=None):
        text = self.corpus().get(rel)
        if text is None:
            text = self.source(rel).decode("utf-8", "replace")
        hit_lines = {text.count("\n", 0, m.start()) + 1 for m in pattern.finditer(text)}
        bits = 0
        for ln in sorted(hit_lines):
            if skip_span and skip_span[0] <= ln <= skip_span[1]:
                continue
            bits |= self.line_ref_bits(rel, ln, depth)
        return bits

    def line_ref_bits(self, rel, ln, depth):
        path = self.locate(rel, ln)
        if not path:
            return self.lines_bits(rel, ln, ln)
        return self.site_bits(rel, path, depth)

    def locate(self, rel, ln):
        """Steps from the module to the innermost statement-like node containing line ln."""
        tree = self.tree(rel)
        path = []
        node = tree
        while True:
            nxt = None
            for f in BLOCK_FIELDS:
                lst = getattr(node, f, None)
                if not isinstance(lst, list):
                    continue
                for i, el in enumerate(lst):
                    lo, hi = _lineno(el), _end(el)
                    if lo is not None and lo <= ln <= hi:
                        nxt = (f, i, el)
                        break
                if nxt:
                    break
            if nxt is None:
                if not path:
                    return path
                # A lambda covering this line executes when called, not with its statement.
                lam = None
                for sub in ast.walk(node):
                    if isinstance(sub, ast.Lambda) and sub.lineno <= ln <= sub.end_lineno:
                        if lam is None or sub.lineno >= lam.lineno:
                            lam = sub
                if lam is not None:
                    return path + [(node, "__ref__", None), (lam, "body", None)]
                return path + [(node, "__ref__", None)]
            f, i, el = nxt
            path.append((node, f, i))
            node = el

    # -- files --------------------------------------------------------------

    def analyze(self, fps):
        """Returns (bits, notes, static) for a whole patch.

        self.trace then says whether every change was a statement inside a function
        ("runtime" sites with their line units and enclosing function span)."""
        self.trace = {"runtime": [], "other": False, "import": False, "nonsite": False, "import_mods": set(), "keyref": []}
        bits = 0
        notes = []
        static = False
        for fp in fps:
            try:
                b, note, st = self.analyze_file(fp)
            except Exception as error:  # never let analysis crash: widen instead
                log(f"  analysis error on {fp.path}: {error!r}")
                b, note, st = self.all_bits, "error", True
            bits |= b
            static |= st
            self.trace.setdefault("per_file", []).append((fp.path, b, note))
            notes.append(f"{fp.path}:{note}")
            if note == "data" and DATA_SITES:
                # Data files only add their readers (kept selected by the selector);
                # they do not stop the code part's sites from being analysed as such.
                self.trace["other"] = True
                self.trace["data"] = True
            elif not (note.startswith("sites=") or note in ("noop", "mode", "equiv")):
                self.trace["other"] = True
                self.trace["nonsite"] = True
        return bits, notes, static

    def analyze_file(self, fp):
        path = fp.path
        base = os.path.basename(path)
        if fp.mode_only:
            return 0, "mode", False
        if fp.old_path and fp.new_path and fp.old_path != fp.new_path:
            # Rename: the old file disappears and a new one appears.
            gone = patchlib.FilePatch()
            gone.old_path, gone.new_path = fp.old_path, None
            new = patchlib.FilePatch()
            new.old_path, new.new_path = None, fp.new_path
            b1, n1, s1 = self.analyze_file(gone)
            b2, n2, s2 = self.analyze_file(new)
            return b1 | b2, f"rename({n1},{n2})", s1 or s2
        if base == "conftest.py":
            d = os.path.dirname(path)
            return self.tests_under(d) | self.all_bits * (d == ""), "conftest", False
        if base in PYTEST_CONFIG and "/" not in path:
            if self._touches_pytest(fp) or base == "pytest.ini":
                return self.all_bits, "pytest-config", True
            return self.reads(path), "config", False
        if fp.binary or not path.endswith(".py"):
            bits = self.reads(path) | self.reads(fp.old_path or "")
            if fp.added or fp.deleted or fp.binary:
                bits |= self.dirs(os.path.dirname(path))
            return bits, "data", False
        if fp.deleted:
            return self.broken_module(fp.old_path) | self.dirs(os.path.dirname(fp.old_path)), "deleted", True
        if fp.added:
            bits = self.dirs(os.path.dirname(path))
            if base == "__init__.py" or base.startswith("test_"):
                bits |= self.tests_under(os.path.dirname(path))
            return bits, "added", False
        old = self.source(path)
        try:
            new = patchlib.apply(old, fp)
        except patchlib.ApplyError as error:
            log(f"  apply failed on {path}: {error}")
            return self.broken_module(path), "apply-failed", True
        try:
            new_tree = ast.parse(new)
            compile(new_tree, path, "exec")
        except (SyntaxError, ValueError):
            return self.broken_module(path), "syntax", True
        old_tree = self.tree(path)
        if ast_equal(old_tree, new_tree):
            return self.reads(path), "noop", False
        if EQUIV_IS and self._is_eq_equivalent(old_tree, new_tree, old, new):
            return self.reads(path), "equiv", False
        sites = ast_diff(old_tree, new_tree)
        bits = 0
        self._new_tree = new_tree
        try:
            for site in sites:
                bits |= self.site_bits(path, site)
        finally:
            self._new_tree = None
        return bits, f"sites={len(sites)}", False

    def _is_eq_equivalent(self, old_tree, new_tree, old_src=None, new_src=None):
        """The only change turns `a is b` into `a == b` where one side is a singleton:
        None/True/False or a member of an Enum class defined in sqlglot (equality of
        such objects is identity)."""
        line = None
        if old_src is not None and new_src is not None:
            # Cheap text check first: one changed line, one ` is ` became ` == `.
            a, b = old_src.split(b"\n"), new_src.split(b"\n")
            if len(a) != len(b):
                return False
            diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
            if len(diff) != 1:
                return False
            x, y = a[diff[0]], b[diff[0]]
            if not any(
                x[: m.start()] + b" == " + x[m.end():] == y for m in re.finditer(rb" is (?!not )", x)
            ):
                return False
            line = diff[0] + 1
        for node in ast.walk(new_tree):
            if not isinstance(node, ast.Compare):
                continue
            if line is not None and not (node.lineno <= line <= getattr(node, "end_lineno", node.lineno)):
                continue
            for k, op in enumerate(node.ops):
                if not isinstance(op, ast.Eq):
                    continue
                node.ops[k] = ast.Is()
                try:
                    same = ast_equal(old_tree, new_tree)
                finally:
                    node.ops[k] = op
                if same:
                    left = node.left if k == 0 else node.comparators[k - 1]
                    return self._singleton(left) or self._singleton(node.comparators[k])
        return False

    def _singleton(self, node):
        if isinstance(node, ast.Constant):
            return node.value is None or node.value is True or node.value is False
        if isinstance(node, ast.Attribute):
            owner = node.value
            name = owner.id if isinstance(owner, ast.Name) else owner.attr if isinstance(owner, ast.Attribute) else None
            return name is not None and name in self._enum_classes()
        return False

    def _enum_classes(self):
        key = ("enums",)
        if key not in self._memo:
            bases = {}
            pat = re.compile(r"^\s*class\s+(\w+)\s*\(([^)]*)\)\s*:", re.M)
            for rel, text in self.corpus().items():
                if rel.startswith("sqlglot/") and "class " in text:
                    for m in pat.finditer(text):
                        bases.setdefault(m.group(1), set()).update(
                            b.strip().rsplit(".", 1)[-1] for b in m.group(2).split(",") if b.strip()
                        )
            enums = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}
            changed = True
            while changed:
                changed = False
                for name, bs in bases.items():
                    if name not in enums and bs & enums:
                        enums.add(name)
                        changed = True
            # (a class name defined twice, once as an enum and once not, is ambiguous)
            self._memo[key] = {n for n in enums if n in bases and bases[n] & enums}
        return self._memo[key]

    def broken_module(self, rel):
        mod = modname(rel)
        bits = self.module_wide(rel)
        if mod in self.eager or (mod and mod.split(".")[0] == "sqlglot" and mod.count(".") == 0):
            bits |= self.all_bits
        last = (mod or "").rsplit(".", 1)[-1]
        if mod and last:
            pat = re.compile(r"\b" + re.escape(mod) + r"\b|import[^\n]*\b" + re.escape(last) + r"\b")
            bits |= self.refs_bits(pat, rel, 2)
        return bits

    def _touches_pytest(self, fp):
        for _os, _ol, _ns, _nl, body in fp.hunks:
            for _tag, text, _ in body:
                if b"pytest" in text or b"[tool" in text or b"addopts" in text:
                    return True
        return False
