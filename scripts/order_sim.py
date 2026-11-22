"""Dev tool (inside the image): CPU spent before the first failing test, under
different run orders, on labeled live commits (analysis only, nothing is run).

  python /opt/scripts/order_sim.py /data   (a corpus dir with commits/ and answers.json)
"""

import json
import os
import pickle
import sys

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import patchlib  # noqa: E402
import selector  # noqa: E402


def main():
    data = sys.argv[1]
    maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
    calib = json.load(open("/opt/maps/verifiable.json"))
    tcpu, scpu = calib["test_cpu"], calib["setup_cpu"]
    ok = set(calib["passed"])
    tests = json.load(open(os.path.join(data, "tests.json")))["tests"]
    num = {t["id"]: t["number"] for t in tests}
    sec = {t["id"]: t["seconds"] for t in tests}
    answers = json.load(open(os.path.join(data, "answers.json")))["commits"]
    T = maps["tests"]
    an = analysis.Analyzer(maps, "/opt/sqlglot")
    idx2num = [num.get(t, -1) for t in T]
    totals = {}
    n_live = 0
    for c, a in sorted(answers.items()):
        failing = {t["id"] for t in tests if t["number"] in set(a["failing"])}
        if not failing:
            continue
        try:
            fps = patchlib.parse(open(os.path.join(data, "commits", c + ".patch"), "rb").read())
            bits, _notes, static = an.analyze(fps)
        except Exception:
            continue
        if static:
            continue
        cands = [T[i] for i in analysis.iter_bits(bits) if T[i] in ok and sec.get(T[i], 9) < 1.0]
        if not cands or not (set(cands) & failing):
            continue
        n_live += 1
        focus = selector.focus_tests(an, maps, idx2num) if an.trace.get("runtime") else {}
        rel_files = {fp.path for fp in fps}

        def cost(t):
            return tcpu.get(t, sec.get(t, 0.1))

        def related(t):
            base = t.split("::")[0].rsplit("/", 1)[-1].replace("test_", "").replace(".py", "")
            return any(base and base in os.path.basename(r) for r in rel_files)

        def related_path(t):
            base = t.split("::")[0].rsplit("/", 1)[-1].replace("test_", "").replace(".py", "")
            return any(base and base in r for r in rel_files)

        orders = {
            "related-path": sorted(cands, key=lambda t: (not related_path(t), cost(t), t)),
            "cheapest": sorted(cands, key=lambda t: (cost(t), t)),
            "focus/cost": sorted(cands, key=lambda t: (-(focus.get(num.get(t, -1), 0) + 1) / (cost(t) + 0.01), t)),
            "related-first": sorted(cands, key=lambda t: (not related(t), cost(t), t)),
            "fewest-lines": sorted(cands, key=lambda t: (bin(an.m["by_class"].get(t.rsplit("::", 1)[0], 0)).count("1"), cost(t), t)),
        }
        for name, order in orders.items():
            spent, classes = 0.0, set()
            for t in order:
                cls = t.rsplit("::", 1)[0]
                if cls not in classes:
                    classes.add(cls)
                    spent += scpu.get(cls, 0.0)
                spent += cost(t)
                if t in failing:
                    break
            totals[name] = totals.get(name, 0.0) + spent
    print(f"{data}: live commits={n_live}", {k: round(v, 2) for k, v in totals.items()})


if __name__ == "__main__":
    main()
