import json

# Load data from files
with open('nksr_results.txt') as f:
    nksr_data = [json.loads(line) for line in f]
with open('ours_results.txt') as f:
    ours_data = [json.loads(line) for line in f]

# Initialize counters
ours_better = 0
ours_worse = 0
ours_same = 0

metrics = 'chamfer-L1'

# Compare Chamfer-L1 values line by line with rounding to 5 decimal places
for nksr_v2, ours in zip(nksr_data, ours_data):
    nksr_value = round(nksr_v2[metrics], 8)
    ours_value = round(ours[metrics], 8)
    if ours_value < nksr_value:
        ours_better += 1
    elif ours_value > nksr_value:
        ours_worse += 1
    else:
        ours_same += 1

print(f"Ours {metrics} is better than NKSR v2 on {ours_better} scenes")
print(f"Ours {metrics} is worse than NKSR v2 on {ours_worse} scenes")
print(f"Ours {metrics} is the same as NKSR v2 on {ours_same} scenes")
