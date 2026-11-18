from itertools import combinations
from collections import Counter

# --------------------------------------------------
# Dataset
# --------------------------------------------------
transactions = [
    {"apple", "banana", "milk"},
    {"apple", "banana"},
    {"banana", "milk"},
    {"apple", "milk"},
    {"apple", "banana", "milk"}
]

min_support = 2

# Step 1: Find frequent 1-itemsets
def find_frequent_1_itemsets(transactions, min_support):
    item_counts = Counter()
    for transaction in transactions:
        for item in transaction:
            item_counts[item] += 1
            
    return {
        frozenset({item})
        for item, count in item_counts.items()
        if count >= min_support
    }

# Step 2: Generate candidate (k+1)-itemsets
def generate_candidates(prev_frequent_itemsets, k):
    items = set()
    for itemset in prev_frequent_itemsets:
        for item in itemset:
            items.add(item)
            
    items_list = sorted(list(items))
    candidates = set()
    
    for combo in combinations(items_list, k + 1):
        candidates.add(frozenset(combo))
        
    return candidates

# Step 3: Count support for an itemset
def support_count(itemset, transactions):
    count = 0
    for transaction in transactions:
        if itemset.issubset(transaction):
            count += 1
    return count

# Step 4: Filter candidates to get frequent itemsets
def filter_frequent_itemsets(candidates, transactions, min_support):
    frequent = set()
    for candidate in candidates:
        if support_count(candidate, transactions) >= min_support:
            frequent.add(candidate)
    return frequent

# --------------------------------------------------
# Execution & Verification
# --------------------------------------------------
if __name__ == "__main__":
    L1 = find_frequent_1_itemsets(transactions, min_support)
    print("Frequent 1-itemsets:", L1)
    
    C2 = generate_candidates(L1, k=1)
    L2 = filter_frequent_itemsets(C2, transactions, min_support)
    print("Frequent 2-itemsets:", L2)
    
    C3 = generate_candidates(L2, k=2)
    print("Candidate 3-itemsets:", C3)