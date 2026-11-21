"""Dev tool (inside the image): where does the selection cost concentrate?

Analysis only (no tests run): for many synthetic mutants, the static selection cost,
split by the kind of site and by how much heavy tests contribute.

  python /opt/scripts/cost_profile.py <n> <seed> <tests.json>
"""

import collections
import json
import pickle
import random
import sys

sys.path.insert(0, "/opt/br")
sys.path.insert(0, "/opt/scripts")
import analysis  # noqa: E402
import mutants  # noqa: E402
import patchlib  # noqa: E402
import selector  # noqa: E402

n, seed, tj = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
T = maps["tests"]
sec = {t["id"]: t["seconds"] for t in json.load(open(tj))["tests"]}
dur = {i: int(sec.get(t, 0) * 1000 + 0.5) for i, t in enumerate(T)}
files = [f for f in maps["lines"] if f.startswith("sqlglot/") and f.endswith(".py")]
an = analysis.Analyzer(maps, "/opt/sqlglot")
rng = random.Random(seed)
by_kind = collections.defaultdict(list)
heavy_names = collections.Counter()
for k in range(n):
    rel, ln, label, patch, _ = mutants.mutate(rng, files)
    bits, notes, static = an.analyze(patchlib.parse(patch))
    tr = an.trace
    kind = "noop" if any("noop" in x for x in notes) else ("runtime" if tr["runtime"] and not tr["other"] else "import-time")
    idx = list(analysis.iter_bits(bits))
    cost = selector.pack(idx, dur)[0] / 1000
    light = [i for i in idx if dur[i] < 1500]
    cost_light = selector.pack(light, dur)[0] / 1000
    for i in idx:
        if dur[i] >= 1500:
            heavy_names[T[i].split("::")[-1]] += 1
    by_kind[kind].append((cost, cost_light, len(idx), rel, ln, label))
for kind, rows in sorted(by_kind.items()):
    tot = sum(r[0] for r in rows)
    print(f"{kind:12} n={len(rows):4} static_total={tot:8.1f}s mean={tot / len(rows):6.2f}s light_only_mean={sum(r[1] for r in rows) / len(rows):6.2f}s")
    for r in sorted(rows, reverse=True)[:6]:
        print(f"    {r[0]:6.2f}s cands={r[2]:5} {r[3]}:{r[4]} {r[5][:20]}")
print("heavy tests by frequency:", heavy_names.most_common(10))
