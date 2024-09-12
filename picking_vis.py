import json

# Load data from files
with open('nksr_scannet_results.txt') as f:
    nksr_data = [json.loads(line) for line in f]
with open('ours_scannet_results.txt') as f:
    ours_data = [json.loads(line) for line in f]

# Initialize list to store gaps
gaps = []

metrics = 'completeness'

# Compute the difference NKSR - Ours in completeness
for nksr, ours in zip(nksr_data, ours_data):
    nksr_value = nksr[metrics]
    ours_value = ours[metrics]
    gap = nksr_value - ours_value  # NKSR - Ours
    
    # Only consider positive gaps where Ours is better
    if gap > 0:
        gaps.append((ours["data_id"], gap))

# Sort the gaps in descending order (largest gap first)
gaps.sort(key=lambda x: x[1], reverse=True)

# Output the 10 largest gaps where Ours is better
top_10_gaps = gaps[:15]
for data_id, gap in top_10_gaps:
    print(f"data_id: {data_id}, completeness gap: {gap}")
