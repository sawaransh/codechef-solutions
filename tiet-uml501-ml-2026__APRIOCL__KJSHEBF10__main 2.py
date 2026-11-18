from itertools import combinations

# --------------------------------------------------
# Dataset (use items from the problem description)
# --------------------------------------------------
L2 = {
    frozenset({"apple", "banana"}),
    frozenset({"apple", "milk"})
}

# Candidate 3-itemsets generated earlier
C3 = {
    frozenset({"apple", "banana", "milk"})
}

# Apriori Pruning Function
def prune_candidates(candidates, prev_frequent_itemsets):
    """
    Removes candidates whose any (k-1)-subset
    is NOT present in prev_frequent_itemsets
    """
    pruned = set()
    for candidate in candidates:
        k = len(candidate)
        
        # Generate all subsets of size (k-1)
        subsets = [frozenset(combo) for combo in combinations(candidate, k - 1)]
        
        # Keep candidate ONLY if all subsets are frequent
        if all(sub in prev_frequent_itemsets for sub in subsets):
            pruned.add(candidate)
            
    return pruned

# --------------------------------------------------
# Execution & Verification
# --------------------------------------------------
if __name__ == "__main__":
    print("Candidates before pruning:", C3)
    final_candidates = prune_candidates(C3, L2)
    print("Candidates after pruning:", final_candidates)
