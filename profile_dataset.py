import csv
from collections import Counter

p = r"data\raw\AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"

rows = 0
pairs = Counter()
apps = set()
min_t = float("inf")
max_t = float("-inf")
min_d = float("inf")
max_d = float("-inf")

with open(p, newline="", encoding="utf-8") as f:
    r = csv.DictReader(f)

    for x in r:
        rows += 1

        app = x["app"]
        func = x["func"]
        t = float(x["end_timestamp"])
        d = float(x["duration"])

        apps.add(app)
        pairs[(app, func)] += 1

        min_t = min(min_t, t)
        max_t = max(max_t, t)
        min_d = min(min_d, d)
        max_d = max(max_d, d)

print("TOTAL ROWS:", rows)
print("UNIQUE APPS:", len(apps))
print("UNIQUE APP+FUNCTIONS:", len(pairs))
print("TIMESTAMP RANGE:", min_t, "to", max_t)
print("DURATION RANGE:", min_d, "to", max_d)

print("\nTOP 10 APP+FUNCTIONS:")
for (app, func), count in pairs.most_common(10):
    print(count, app[:12] + "...", func[:12] + "...")
