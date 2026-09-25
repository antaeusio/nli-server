"""Write a --require-hashes lock with every PyPI file hash of each pinned requirement."""

import json
import re
import sys
import urllib.request

source, target = sys.argv[1], sys.argv[2]
out = []
for line in open(source):
    line = line.strip()
    if not line or line.startswith("#"):
        continue
    match = re.match(r"^([A-Za-z0-9_.-]+)==([^ ;]+)(.*)$", line)
    if not match:
        sys.exit(f"unpinned requirement: {line}")
    name, version, marker = match.groups()
    with urllib.request.urlopen(f"https://pypi.org/pypi/{name}/{version}/json") as response:
        files = json.load(response)["urls"]
    hashes = sorted({f["digests"]["sha256"] for f in files})
    if not hashes:
        sys.exit(f"no files on PyPI for {name}=={version}")
    out.append(f"{name}=={version}{marker} \\\n" + " \\\n".join(f"    --hash=sha256:{h}" for h in hashes))
open(target, "w").write("\n".join(out) + "\n")
