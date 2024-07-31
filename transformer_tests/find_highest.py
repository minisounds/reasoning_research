import json

# Load the JSON data
with open("../res/last_token.json", "r") as f:
    data = json.load(f)

# Extract avg_scores with their corresponding layer and coefficient
avg_scores_info = [(result["avg_score"], result["layer"], result["coefficient"]) for result in data["results"]]

# count # responses with average score 4 or greater
good_count = 0
for res in avg_scores_info: 
    if res[0] >= 4:
        good_count +=1 
print(f"Results with average score over 4.0: {good_count} (out of {len(avg_scores_info)} responses)")

# Find: average score
avg_score = sum(score[0] for score in avg_scores_info)/len(avg_scores_info)
print(f"Average score: {avg_score}")

# Find: most common layer in good responses: 
common_layers = {}
for res in avg_scores_info:
    if res[0] >= 4:
        common_layers[res[1]] = 1 + common_layers.get(res[1], 0)

buckets = [[] for i in range(max(list(common_layers.values()))+1)]
for k, v in common_layers.items():
    buckets[v].append(k)

res = []
for i in range(len(buckets)-1, 0, -1):
    for val in buckets[i]:
        res.append(val)
    if len(res) == 10: 
        break

print(f"Top 10 most common layers in 'good' responses: {res}")


# Print the top 10 avg_scores with their layer and coefficient
top_10_avg_scores = sorted(avg_scores_info, key=lambda x: x[0], reverse=True)[:10]
for avg_score, layer, coefficient in top_10_avg_scores:
    print(f"Avg Score: {avg_score}, Layer: {layer}, Coefficient: {coefficient}")
