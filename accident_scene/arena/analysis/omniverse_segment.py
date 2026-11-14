"""Cut what Isaac runs appended to the shared Omniverse logs since a start snapshot.

~/.nvidia-omniverse/logs/*.log are JSON-line logs every Isaac launch appends to (the
telemetry side of Kit). An experiment records their sizes before its first run
(omniverse_logs_offsets_start.json); this copies the bytes appended since then into
<out>/ and writes segment.json (byte ranges, time span). A file that shrank was rotated:
the whole current file is kept and that is noted. Runs that overlap in time share a
segment; each entry has its own timestamp and each run's window is in its *.kept.json.

    python3 analysis/omniverse_segment.py results/turns/prompt_a2/isaac_logs/omniverse_logs_offsets_start.json \
        results/turns/prompt_a2/isaac_logs/omniverse_logs
"""
import glob
import json
import os
import sys
import time

start = json.load(open(sys.argv[1]))
out = sys.argv[2]
d = start["dir"]
os.makedirs(out, exist_ok=True)
manifest = {"from": start["taken"], "to": time.strftime("%F %T"), "to_epoch": time.time(), "files": {}}
for f in sorted(glob.glob(d + "/*")):
    if not os.path.isfile(f):
        continue
    name, size = os.path.basename(f), os.path.getsize(f)
    off = start["files"].get(name, 0)
    note = ""
    if size < off:
        off, note = 0, "rotated - whole current file kept"
    if size == off:
        manifest["files"][name] = {"start": off, "end": size, "bytes": 0, "note": "nothing appended"}
        continue
    with open(f, "rb") as fh:
        fh.seek(off)
        blob = fh.read(size - off)
    open(os.path.join(out, name), "wb").write(blob)
    manifest["files"][name] = {"start": off, "end": size, "bytes": len(blob), "note": note}
json.dump(manifest, open(os.path.join(out, "segment.json"), "w"), indent=1)
print(json.dumps({k: v["bytes"] for k, v in manifest["files"].items()}), manifest["from"], "->", manifest["to"])
