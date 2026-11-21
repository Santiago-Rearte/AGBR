"""Turns make_corpus.py output into a /data-style directory plus answers.json.

  python scripts/corpus_to_data.py corpus.jsonl tests.json out_dir
"""

import json
import os
import sys

src, tests_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
os.makedirs(os.path.join(out, "commits"), exist_ok=True)
tests = json.load(open(tests_path, encoding="utf-8"))
num = {t["id"]: t["number"] for t in tests["tests"]}
answers = {}
for line in open(src, encoding="utf-8"):
    if not line.startswith("{"):
        continue
    r = json.loads(line)
    with open(os.path.join(out, "commits", r["id"] + ".patch"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(r["patch"])
    answers[r["id"]] = {"failing": sorted(num[t] for t in r["failing"] if t in num), "kind": r["label"]}
with open(os.path.join(out, "tests.json"), "w", encoding="utf-8") as fh:
    json.dump(tests, fh)
with open(os.path.join(out, "answers.json"), "w", encoding="utf-8") as fh:
    json.dump({"commits": answers}, fh)
print(f"{len(answers)} commits, {sum(1 for a in answers.values() if a['failing'])} with failures")
