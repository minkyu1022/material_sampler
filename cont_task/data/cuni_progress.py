#!/usr/bin/env python3
import json
import sys
from pathlib import Path

records = {}
malformed = 0
for path in Path(sys.argv[1]).glob("*/frames.jsonl"):
    for line in path.read_text().splitlines():
        try:
            record = json.loads(line)
            records[(path.parent.name, int(record["frame"]))] = record.get("status")
        except json.JSONDecodeError:
            malformed += 1
print(json.dumps({
    "recorded": len(records),
    "converged": sum(value == "converged" for value in records.values()),
    "bad": sum(value != "converged" for value in records.values()),
    "malformed": malformed,
}))
