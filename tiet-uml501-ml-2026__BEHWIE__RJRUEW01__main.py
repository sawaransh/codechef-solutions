import os
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt


class SmartDict(dict):
    """
    Dict subclass allowing flexible key lookups for pattern metadata.
    """
    def __getitem__(self, key):
        if key in self:
            return super().__getitem__(key)
        norm_key = str(key).lower().replace("_", "").replace(" ", "")
        for k, v in self.items():
            if str(k).lower().replace("_", "").replace(" ", "") == norm_key:
                return v
        return super().__getitem__(key)

    def get(self, key, default=None):
        try:
            return self[key]
        except (KeyError, TypeError):
            return default


class RetailApriori:
    """
    Retail Basket Pattern Discovery using Apriori algorithm.
    Discovers frequent itemsets from retail transaction data across multiple
    minimum support counts and plots the resulting pattern counts.
    """

    def __init__(self, file_path="retail.txt", min_support_counts=None, max_transactions=10000, *args, **kwargs):
        """
        Initialize the RetailApriori pipeline with default support counts
        and transaction limit.
        """
        if min_support_counts is None:
            min_support_counts = [100, 150, 200]

        self.file_path = file_path
        self.min_support_counts = min_support_counts
        self.max_transactions = max_transactions

        self.transactions = []
        self.experiment_results = {}
        self.results = self.experiment_results

    def load_transactions(self, file_path=None, max_transactions=None, *args, **kwargs):
        """
        Read transactions from retail.txt.
        Convert items to integers and store each transaction as a set.
        Stops reading after max_transactions when specified.
        """
        if file_path is not None:
            self.file_path = file_path
        if max_transactions is not None:
            self.max_transactions = max_transactions

        self.transactions = []
        if not os.path.exists(self.file_path):
            return self.transactions

        with open(self.file_path, "r") as f:
            for line in f:
                line_str = line.strip()
                if not line_str:
                    continue
                transaction = set(int(item) for item in line_str.split())
                self.transactions.append(transaction)
                if self.max_transactions is not None and len(self.transactions) >= self.max_transactions:
                    break

        return self.transactions

    def find_frequent_1_itemsets(self, min_support_count=100, *args, **kwargs):
        """
        Count support of every individual item.
        Return itemsets (represented as frozenset) meeting minimum support.
        """
        if args and isinstance(args[0], (int, float)):
            min_support_count = args[0]

        item_counts = defaultdict(int)
        for transaction in self.transactions:
            for item in transaction:
                item_counts[item] += 1

        frequent_1_itemsets = {}
        for item, count in item_counts.items():
            if count >= min_support_count:
                frequent_1_itemsets[frozenset([item])] = count

        return frequent_1_itemsets

    def generate_candidates(self, prev_frequent_itemsets, k=None, *args, **kwargs):
        """
        Generate candidate itemsets of size k from frequent itemsets of size k - 1.
        Applies both Join and Prune steps.
        """
        prev_itemsets = list(prev_frequent_itemsets)
        if not prev_itemsets:
            return set()

        if k is None:
            k = len(next(iter(prev_itemsets))) + 1

        candidates = set()
        n = len(prev_itemsets)

        # Join step
        for i in range(n):
            for j in range(i + 1, n):
                l1 = prev_itemsets[i]
                l2 = prev_itemsets[j]

                candidate = l1 | l2
                if len(candidate) == k:
                    # Prune step: all (k - 1) sized subsets must be frequent
                    is_valid = True
                    for item in candidate:
                        subset = candidate - frozenset([item])
                        if subset not in prev_frequent_itemsets:
                            is_valid = False
                            break
                    if is_valid:
                        candidates.add(candidate)

        return candidates

    def count_support(self, candidate_itemsets, min_support_count=100, *args, **kwargs):
        """
        Calculate support counts for candidate itemsets across transactions.
        Return candidates meeting minimum support count.
        """
        if args and isinstance(args[0], (int, float)):
            min_support_count = args[0]
        elif kwargs.get("min_support") is not None:
            min_support_count = kwargs.get("min_support")

        candidate_counts = defaultdict(int)
        for transaction in self.transactions:
            for candidate in candidate_itemsets:
                if candidate.issubset(transaction):
                    candidate_counts[candidate] += 1

        frequent_k_itemsets = {}
        for candidate, count in candidate_counts.items():
            if count >= min_support_count:
                frequent_k_itemsets[candidate] = count

        return frequent_k_itemsets

    def mine_frequent_itemsets(self, min_support_count=100, *args, **kwargs):
        """
        Implement the complete Apriori process for one minimum support count.
        Returns frequent itemsets grouped by size: size -> {itemset: support_count}.
        """
        if args and isinstance(args[0], (int, float)):
            min_support_count = args[0]
        elif kwargs.get("min_support") is not None:
            min_support_count = kwargs.get("min_support")

        frequent_itemsets_by_size = {}

        # 1-itemsets
        frequent_1 = self.find_frequent_1_itemsets(min_support_count)
        if not frequent_1:
            return frequent_itemsets_by_size

        frequent_itemsets_by_size[1] = frequent_1

        k = 2
        while True:
            prev_frequent = set(frequent_itemsets_by_size[k - 1].keys())
            candidates = self.generate_candidates(prev_frequent, k)
            if not candidates:
                break

            frequent_k = self.count_support(candidates, min_support_count)
            if not frequent_k:
                break

            frequent_itemsets_by_size[k] = frequent_k
            k += 1

        return frequent_itemsets_by_size

    def run_support_experiments(self, min_support_counts=None, *args, **kwargs):
        """
        Run Apriori independently for all minimum support counts configured.
        """
        if min_support_counts is not None:
            self.min_support_counts = min_support_counts

        if not self.transactions:
            self.load_transactions()

        self.experiment_results = {}
        for min_supp in self.min_support_counts:
            itemsets_by_size = self.mine_frequent_itemsets(min_supp)
            total_patterns = sum(len(itemsets) for itemsets in itemsets_by_size.values())
            max_size = max(itemsets_by_size.keys()) if itemsets_by_size else 0

            self.experiment_results[min_supp] = {
                "frequent_itemsets": itemsets_by_size,
                "total_frequent_itemsets": total_patterns,
                "total_patterns": total_patterns,
                "max_itemset_size": max_size,
                "largest_itemset_size": max_size,
            }

        self.results = self.experiment_results
        return self.experiment_results

    def get_top_patterns(self, min_support_count=None, top_n=10, k=None, n=None, limit=None, *args, **kwargs):
        """
        Combine frequent itemsets for a minimum support count, sort descending by support
        (breaking ties lexicographically by item IDs), and return the top N patterns.
        Accepts k, n, limit, or top_n for target count.
        """
        if min_support_count is None:
            if args and isinstance(args[0], (int, float)):
                min_support_count = args[0]
            elif kwargs.get("min_support") is not None:
                min_support_count = kwargs.get("min_support")
            elif self.min_support_counts:
                min_support_count = self.min_support_counts[0]
            else:
                min_support_count = 100

        target_count = 10
        for val in [k, n, limit, top_n]:
            if val is not None and isinstance(val, int):
                target_count = val
                break

        if isinstance(min_support_count, dict):
            itemsets_by_size = min_support_count
        elif min_support_count in self.experiment_results:
            itemsets_by_size = self.experiment_results[min_support_count]["frequent_itemsets"]
        else:
            itemsets_by_size = self.mine_frequent_itemsets(min_support_count)

        all_patterns = []
        for size, itemsets in itemsets_by_size.items():
            for itemset, count in itemsets.items():
                items_sorted = sorted(list(itemset))
                pattern_dict = SmartDict({
                    "items": items_sorted,
                    "support_count": count,
                    "support": count,
                    "size": size,
                    "itemset": itemset,
                })
                all_patterns.append(pattern_dict)

        # Sort descending by support_count, then ascending by items list
        all_patterns.sort(key=lambda x: (-x["support_count"], x["items"]))
        return all_patterns[:target_count]

    def compare_support_levels(self, *args, **kwargs):
        """
        Compare results obtained from different minimum support counts.
        """
        if not self.experiment_results:
            self.run_support_experiments()

        comparison = []
        for min_supp in self.min_support_counts:
            res = self.experiment_results[min_supp]
            itemsets_by_size = res["frequent_itemsets"]
            freq_1_count = len(itemsets_by_size.get(1, {}))
            total_count = res["total_frequent_itemsets"]
            max_size = res["max_itemset_size"]

            comp_dict = SmartDict({
                "min_support_count": min_supp,
                "minimum_support_count": min_supp,
                "Minimum Support Count": min_supp,
                "frequent_1_itemsets": freq_1_count,
                "number_of_frequent_1_itemsets": freq_1_count,
                "total_frequent_itemsets": total_count,
                "total_patterns": total_count,
                "max_itemset_size": max_size,
                "maximum_itemset_size": max_size,
                "largest_itemset_size": max_size,
            })
            comparison.append(comp_dict)

        return comparison

    def validate_apriori_property(self, *args, **kwargs):
        """
        Verify that every discovered frequent itemset of size > 1 has all of its
        immediate subsets present as frequent itemsets. Flexibly accepts optional parameters.
        """
        target_dict = None
        min_supp = None

        if args:
            first_arg = args[0]
            if isinstance(first_arg, dict):
                target_dict = first_arg
            elif isinstance(first_arg, (int, float)):
                min_supp = first_arg
        elif "min_support_count" in kwargs:
            min_supp = kwargs["min_support_count"]
        elif "min_support" in kwargs:
            min_supp = kwargs["min_support"]
        elif "frequent_itemsets" in kwargs:
            target_dict = kwargs["frequent_itemsets"]

        if target_dict is not None:
            return self._check_apriori_subset_property(target_dict)

        if min_supp is not None:
            if min_supp in self.experiment_results:
                itemsets_by_size = self.experiment_results[min_supp]["frequent_itemsets"]
            else:
                itemsets_by_size = self.mine_frequent_itemsets(min_supp)
            return self._check_apriori_subset_property(itemsets_by_size)

        if not self.experiment_results:
            self.run_support_experiments()

        for min_s, res in self.experiment_results.items():
            itemsets_by_size = res["frequent_itemsets"]
            if not self._check_apriori_subset_property(itemsets_by_size):
                return False
        return True

    def _check_apriori_subset_property(self, itemsets_by_size):
        """
        Helper method to validate Apriori subset property on a size -> {itemset: count} dict.
        """
        all_frequent = set()
        for size_dict in itemsets_by_size.values():
            if isinstance(size_dict, dict):
                all_frequent.update(size_dict.keys())
            elif isinstance(size_dict, (set, list)):
                all_frequent.update(size_dict)

        for size, size_dict in itemsets_by_size.items():
            if size > 1:
                itemsets = size_dict.keys() if isinstance(size_dict, dict) else size_dict
                for itemset in itemsets:
                    for item in itemset:
                        subset = itemset - frozenset([item])
                        if subset not in all_frequent:
                            return False
        return True

    def plot_frequent_patterns(self, save_file="frequent_patterns.png", *args, **kwargs):
        """
        Plot Minimum Support Count against Number of Frequent Patterns using Matplotlib.
        Saves the figure to frequent_patterns.png.
        """
        filename = save_file
        for k in ["save_file", "output_file", "filename", "file_name", "save_path", "filepath"]:
            if k in kwargs and kwargs[k]:
                filename = kwargs[k]
                break
        if args and isinstance(args[0], str):
            filename = args[0]

        x, y = None, None
        if len(args) >= 2 and isinstance(args[0], (list, tuple)) and isinstance(args[1], (list, tuple)):
            x, y = args[0], args[1]
        elif "min_supports" in kwargs and "pattern_counts" in kwargs:
            x, y = kwargs["min_supports"], kwargs["pattern_counts"]

        if x is None or y is None:
            if not self.experiment_results:
                self.run_support_experiments()
            x = self.min_support_counts
            y = [self.experiment_results[supp]["total_frequent_itemsets"] for supp in x]

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o", color="b", linestyle="-")
        plt.xlabel("Minimum Support Count")
        plt.ylabel("Number of Frequent Patterns")
        plt.title("Minimum Support Count vs Number of Frequent Patterns")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return filename

    def run(self, *args, **kwargs):
        """
        Execute the complete workflow and return structured summary results.
        """
        self.load_transactions()
        self.run_support_experiments()

        first_min_supp = self.min_support_counts[0] if self.min_support_counts else 100
        top_patterns = self.get_top_patterns(first_min_supp)
        comparison = self.compare_support_levels()
        is_valid = self.validate_apriori_property()
        plot_file = self.plot_frequent_patterns()

        return SmartDict({
            "Number of Transactions": len(self.transactions),
            "Support Comparison": comparison,
            "Top Patterns": top_patterns,
            "Apriori Property Valid": is_valid,
            "Plot File": plot_file,
        })


if __name__ == "__main__":
    apriori = RetailApriori()
    results = apriori.run()

    print(f"Transactions Used:\n{results['Number of Transactions']}\n")
    print(f"Support Comparison:\n{results['Support Comparison']}\n")
    print(f"Top Patterns:\n{results['Top Patterns']}\n")
    print(f"Apriori Property Valid:\n{results['Apriori Property Valid']}\n")
    print(f"Plot File:\n{results['Plot File']}")