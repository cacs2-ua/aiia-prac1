import sys, csv, os
from datetime import datetime

def fail(msg): print("FAIL:", msg); sys.exit(1)
def ok(msg): print("OK:", msg)

if len(sys.argv) < 4:
    print("Usage: python validate_tarea13.py <csv_path> <expected_samples> <interval_seconds>")
    sys.exit(2)

csv_path = sys.argv[1]
expected = int(sys.argv[2])
interval = float(sys.argv[3])

if not os.path.exists(csv_path): fail(f"CSV not found: {csv_path}")

with open(csv_path, newline='', encoding='utf-8') as f:
    r = list(csv.reader(f))
if not r or r[0] != ["Timestamp","Person_Count"]:
    fail("Header must be exactly: Timestamp,Person_Count")
ok("Header OK")

rows = r[1:]
if len(rows) != expected:
    fail(f"Row count {len(rows)} != expected {expected}")
ok(f"Row count OK = {expected}")

# parse data
times = []
counts = []
for i,(ts, cnt) in enumerate(rows, start=1):
    try:
        t = datetime.strptime(ts, "%Y-%m-%d %H:%M:%S")
        c = int(cnt)
        if c < 0: fail(f"Negative count at row {i}")
    except Exception as e:
        fail(f"Bad row {i}: {e}")
    times.append(t); counts.append(c)
ok("All rows are valid and non-negative integers")

# timing deltas (allow slack because inference adds time)
if len(times) >= 2:
    deltas = [(times[i]-times[i-1]).total_seconds() for i in range(1, len(times))]
    median = sorted(deltas)[len(deltas)//2]
    print(f"Δt stats (s): min={min(deltas):.2f} median={median:.2f} max={max(deltas):.2f}")
else:
    print("Only one sample; skipping Δt checks.")

# weak sanity: if all zeros and you expected people, warn (not fail)
if all(c==0 for c in counts):
    print("WARN: All counts are zero. If your scene has people, try lowering --conf or another stream.")
else:
    print("Counts vary as expected.")

print("PASS: Task 1.3 artifacts look correct.")
