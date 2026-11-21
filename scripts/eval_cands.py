"""Dev tool (runs inside the image): static-analysis recall against sample answers.

docker run ... -v sample:/sample br-dev python /opt/scripts/eval_cands.py /sample
"""

import json
import os
import pickle
import sys
import time

sys.path.insert(0, "/opt/br")
import analysis  # noqa: E402
import patchlib  # noqa: E402
import selector  # noqa: E402

sample = sys.argv[1]
only = set(sys.argv[2:])
tests = json.load(open(os.path.join(sample, "tests.json")))["tests"]
answers = json.load(open(os.path.join(sample, "answers.json")))["commits"]
maps = pickle.load(open("/opt/maps/maps.pkl", "rb"))
id2num = {t["id"]: t["number"] for t in tests}
num2id = {v: k for k, v in id2num.items()}
idx2num = [id2num.get(t, -1) for t in maps["tests"]]
dur = {t["number"]: int(t["seconds"] * 1000 + 0.5) for t in tests}
an = analysis.Analyzer(maps, "/opt/sqlglot")
total_static = total_oracle = 0
misses = []
for c in sorted(answers):
    if only and c not in only:
        continue
    t0 = time.monotonic()
    fps = patchlib.parse(open(os.path.join(sample, "commits", c + ".patch"), "rb").read())
    bits, notes, static = an.analyze(fps)
    cands = {idx2num[i] for i in analysis.iter_bits(bits) if idx2num[i] >= 0}
    failing = set(answers[c]["failing"])
    missing = failing - cands
    cs = selector.pack(cands, dur)[0] / 1000
    co = selector.pack(failing, dur)[0] / 1000
    total_static += cs
    total_oracle += co
    if missing:
        misses.append(c)
    heavy = sorted((n for n in cands if dur[n] > 1000), key=lambda n: -dur[n])
    print(
        f"{c} {answers[c]['kind']:8} cand={len(cands):4} sum={sum(dur[n] for n in cands) / 1000:7.2f}s "
        f"static={cs:6.2f} oracle={co:6.2f} fail={len(failing):3} MISSING={len(missing)} "
        f"t={time.monotonic() - t0:.2f}s {' '.join(notes)[:90]}"
    )
    if heavy:
        print("      heavy:", ", ".join(f"{num2id[n].split('::')[-1]}={dur[n] / 1000:.1f}" for n in heavy[:6]))
    for n in sorted(missing)[:5]:
        print("      missing:", num2id[n])
print(f"static total={total_static:.1f}s oracle total={total_oracle:.1f}s misses={misses}")
