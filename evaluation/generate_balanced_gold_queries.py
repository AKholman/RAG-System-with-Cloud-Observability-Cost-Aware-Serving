# evaluation/generate_balanced_gold_queries.py

import json
import random

with open("artifacts/metadata.json") as f:
    docs = json.load(f)

def paraphrase(title):
    rules = [
        lambda t: t.replace("announces", "announces plans for"),
        lambda t: t.replace("wins", "secures victory in"),
        lambda t: t.replace("arrested", "taken into custody"),
        lambda t: t.replace("launches", "rolls out"),
        lambda t: "Report on " + t.lower(),
    ]
    return random.choice(rules)(title)

gold = []

for doc in random.sample(docs, 50):
    gold.append({
        "query": paraphrase(doc["title"]),
        "relevant_doc_ids": [doc["doc_id"]]
    })

with open("evaluation/gold_queries_balanced.json", "w") as f:
    json.dump(gold, f, indent=2)

print("✅ Generated balanced gold queries")
