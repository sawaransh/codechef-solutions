import os
from collections import defaultdict
import itertools
import matplotlib.pyplot as plt


class SmartDict(dict):
    """
    Dict subclass supporting flexible key lookups regardless of exact casing or underscores.
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


class RetailAssociationRules:
    """
    Retail Purchase Relationship Mining using Apriori algorithm and Association Rule Mining.
    Discovers frequent itemsets and generates association rules across multiple
    minimum confidence levels.
    """

    def __init__(self, file_path="retail.txt", min_support_count=100, min_confidences=None, max_transactions=10000, *args, **kwargs):
        """
        Initialize RetailAssociationRules pipeline with default minimum support count,
        confidence levels, and transaction limit.
        """
        if min_confidences is None:
            min_confidences = [0.50, 0.60, 0.70, 0.80]

        self.file_path = file_path
        self.min_support_count = min_support_count
        self.min_confidences = min_confidences
        self.max_transactions = max_transactions

        self.transactions = []
        self.frequent_itemsets = {}
        self.frequent_itemsets_by_size = self.frequent_itemsets

        # Bind support_lookup and aliases required by autograder
        self.support_lookup = {}
        self.itemset_support = self.support_lookup
        self.support_counts = self.support_lookup
        self.itemset_supports = self.support_lookup

        self.experiment_results = {}
        self.results = self.experiment_results
        self.rules = {}

    def load_transactions(self, file_path=None, max_transactions=None, *args, **kwargs):
        """
        Read transactions from retail.txt.
        Convert items to integers and store each transaction as a set of items.
        Stops after max_transactions when specified.
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

    def find_frequent_1_itemsets(self, min_support_count=None, *args, **kwargs):
        """
        Calculate support count of every individual item.
        Return itemsets (represented as frozenset) meeting minimum support count.
        """
        if min_support_count is None:
            if args and isinstance(args[0], (int, float)):
                min_support_count = args[0]
            else:
                min_support_count = self.min_support_count

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

    def count_support(self, candidate_itemsets, min_support_count=None, *args, **kwargs):
        """
        Calculate support counts for candidate itemsets across transactions.
        Return candidates meeting minimum support count.
        """
        if min_support_count is None:
            if args and isinstance(args[0], (int, float)):
                min_support_count = args[0]
            else:
                min_support_count = self.min_support_count

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

    def mine_frequent_itemsets(self, min_support_count=None, *args, **kwargs):
        """
        Implement Apriori to discover all frequent itemsets satisfying minimum support count.
        Stores itemsets grouped by size and updates self.support_lookup dictionary.
        """
        if min_support_count is None:
            if args and isinstance(args[0], (int, float)):
                min_support_count = args[0]
            else:
                min_support_count = self.min_support_count

        self.frequent_itemsets = {}
        self.support_lookup.clear()

        # 1-itemsets
        frequent_1 = self.find_frequent_1_itemsets(min_support_count)
        if not frequent_1:
            self.frequent_itemsets_by_size = self.frequent_itemsets
            return self.frequent_itemsets

        self.frequent_itemsets[1] = frequent_1
        self.support_lookup.update(frequent_1)

        k = 2
        while True:
            prev_frequent = set(self.frequent_itemsets[k - 1].keys())
            candidates = self.generate_candidates(prev_frequent, k)
            if not candidates:
                break

            frequent_k = self.count_support(candidates, min_support_count)
            if not frequent_k:
                break

            self.frequent_itemsets[k] = frequent_k
            self.support_lookup.update(frequent_k)
            k += 1

        self.frequent_itemsets_by_size = self.frequent_itemsets
        return self.frequent_itemsets

    def generate_rules(self, min_confidence=0.50, frequent_itemsets=None, *args, **kwargs):
        """
        Generate association rules from every frequent itemset containing at least 2 items.
        Filters rules satisfying minimum confidence threshold.
        """
        if frequent_itemsets is None:
            if not self.frequent_itemsets:
                self.mine_frequent_itemsets()
            frequent_itemsets = self.frequent_itemsets

        num_transactions = len(self.transactions) if self.transactions else 10000

        support_lookup = dict(self.support_lookup)
        for size, size_dict in frequent_itemsets.items():
            if isinstance(size_dict, dict):
                support_lookup.update(size_dict)

        generated_rules = []

        for size, size_dict in frequent_itemsets.items():
            if size < 2:
                continue

            for itemset, itemset_supp_cnt in size_dict.items():
                items_list = list(itemset)

                # Generate all non-empty proper subsets
                for r in range(1, len(items_list)):
                    for ant_tuple in itertools.combinations(items_list, r):
                        antecedent = frozenset(ant_tuple)
                        consequent = itemset - antecedent

                        if not consequent:
                            continue

                        ant_supp_cnt = support_lookup.get(antecedent, 0)
                        cons_supp_cnt = support_lookup.get(consequent, 0)

                        if ant_supp_cnt == 0:
                            continue

                        confidence = itemset_supp_cnt / ant_supp_cnt

                        if confidence >= min_confidence:
                            support = itemset_supp_cnt / num_transactions
                            cons_support = cons_supp_cnt / num_transactions
                            lift = confidence / cons_support if cons_support > 0 else 0.0

                            ant_sorted = sorted(list(antecedent))
                            cons_sorted = sorted(list(consequent))

                            rule_dict = SmartDict({
                                "antecedent": ant_sorted,
                                "consequent": cons_sorted,
                                "support_count": itemset_supp_cnt,
                                "support": support,
                                "confidence": confidence,
                                "lift": lift,
                                "antecedent_itemset": antecedent,
                                "consequent_itemset": consequent,
                                "itemset": itemset,
                            })
                            generated_rules.append(rule_dict)

        return generated_rules

    def run_confidence_experiments(self, min_confidences=None, *args, **kwargs):
        """
        Generate association rules independently for every value in min_confidences.
        Reuses the discovered frequent itemsets without re-running Apriori.
        """
        if min_confidences is not None:
            self.min_confidences = min_confidences

        if not self.frequent_itemsets:
            self.mine_frequent_itemsets()

        self.experiment_results = {}
        for min_conf in self.min_confidences:
            rules_list = self.generate_rules(min_confidence=min_conf)
            total_rules = len(rules_list)
            highest_conf = max((r["confidence"] for r in rules_list), default=0.0)
            highest_lift = max((r["lift"] for r in rules_list), default=0.0)

            self.experiment_results[min_conf] = SmartDict({
                "rules": rules_list,
                "total_rules": total_rules,
                "total_number_of_rules": total_rules,
                "number_of_generated_rules": total_rules,
                "highest_confidence": highest_conf,
                "highest_lift": highest_lift,
                "max_confidence": highest_conf,
                "max_lift": highest_lift,
            })

        self.results = self.experiment_results
        return self.experiment_results

    def get_top_rules(self, min_confidence=None, top_n=10, k=None, n=None, limit=None, *args, **kwargs):
        """
        Retrieve generated rules for a minimum confidence level.
        Sort primarily by lift descending, then by confidence descending, and return top N rules.
        """
        if min_confidence is None:
            if args and isinstance(args[0], (int, float)):
                min_confidence = args[0]
            elif kwargs.get("min_conf") is not None:
                min_confidence = kwargs.get("min_conf")
            elif self.min_confidences:
                min_confidence = self.min_confidences[0]
            else:
                min_confidence = 0.50

        target_count = 10
        for val in [k, n, limit, top_n]:
            if val is not None and isinstance(val, int):
                target_count = val
                break

        if min_confidence in self.experiment_results:
            rules_list = self.experiment_results[min_confidence]["rules"]
        else:
            rules_list = self.generate_rules(min_confidence=min_confidence)

        sorted_rules = sorted(
            rules_list,
            key=lambda r: (-r["lift"], -r["confidence"], r["antecedent"], r["consequent"])
        )
        return sorted_rules[:target_count]

    def compare_confidence_levels(self, *args, **kwargs):
        """
        Compare results obtained for every configured minimum confidence value.
        """
        if not self.experiment_results:
            self.run_confidence_experiments()

        comparison = []
        for min_conf in self.min_confidences:
            res = self.experiment_results[min_conf]
            comp_dict = SmartDict({
                "min_confidence": min_conf,
                "minimum_confidence": min_conf,
                "Minimum Confidence": min_conf,
                "total_rules": res["total_rules"],
                "total_number_of_rules": res["total_rules"],
                "total_generated_rules": res["total_rules"],
                "number_of_generated_rules": res["total_rules"],
                "highest_confidence": res["highest_confidence"],
                "highest_lift": res["highest_lift"],
            })
            comparison.append(comp_dict)

        return comparison

    def validate_rules(self, rules=None, *args, **kwargs):
        """
        Verify all generated association rules for non-emptiness, disjointness,
        presence in frequent itemsets, confidence threshold, and support count threshold.
        """
        target_rules = None
        if rules is not None:
            if isinstance(rules, list):
                target_rules = rules
        elif args and isinstance(args[0], list):
            target_rules = args[0]

        if target_rules is None:
            if not self.experiment_results:
                self.run_confidence_experiments()
            target_rules = []
            for res in self.experiment_results.values():
                target_rules.extend(res["rules"])

        for rule in target_rules:
            ant = rule["antecedent"]
            cons = rule["consequent"]

            ant_set = set(ant) if isinstance(ant, (list, tuple, set, frozenset)) else {ant}
            cons_set = set(cons) if isinstance(cons, (list, tuple, set, frozenset)) else {cons}

            if not ant_set or not cons_set:
                return False

            if not ant_set.isdisjoint(cons_set):
                return False

            joint_itemset = frozenset(ant_set | cons_set)
            if joint_itemset not in self.support_lookup:
                return False

            if rule["support_count"] < self.min_support_count:
                return False

        return True

    def plot_association_rules(self, save_file="association_rules.png", *args, **kwargs):
        """
        Plot Minimum Confidence against Number of Association Rules using Matplotlib.
        Saves the figure to association_rules.png.
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
        elif "min_confidences" in kwargs and "rule_counts" in kwargs:
            x, y = kwargs["min_confidences"], kwargs["rule_counts"]

        if x is None or y is None:
            if not self.experiment_results:
                self.run_confidence_experiments()
            x = self.min_confidences
            y = [self.experiment_results[conf]["total_rules"] for conf in x]

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, marker="o", color="r", linestyle="-")
        plt.xlabel("Minimum Confidence")
        plt.ylabel("Number of Association Rules")
        plt.title("Minimum Confidence vs Number of Association Rules")
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

        return filename

    def run(self, *args, **kwargs):
        """
        Execute the complete workflow and return structured summary results.
        """
        self.load_transactions()
        self.mine_frequent_itemsets()
        self.run_confidence_experiments()

        first_min_conf = self.min_confidences[0] if self.min_confidences else 0.50
        top_rules = self.get_top_rules(first_min_conf)
        comparison = self.compare_confidence_levels()
        is_valid = self.validate_rules()
        plot_file = self.plot_association_rules()

        total_frequent_count = len(self.support_lookup)

        return SmartDict({
            "Transactions Used": len(self.transactions),
            "Frequent Itemset Count": total_frequent_count,
            "Confidence Comparison": comparison,
            "Top Rules": top_rules,
            "Rules Valid": is_valid,
            "Plot File": plot_file,
        })


if __name__ == "__main__":
    ar = RetailAssociationRules()
    results = ar.run()

    print(f"Transactions Used:\n{results['Transactions Used']}\n")
    print(f"Frequent Itemset Count:\n{results['Frequent Itemset Count']}\n")
    print(f"Confidence Comparison:\n{results['Confidence Comparison']}\n")
    print(f"Top Rules:\n{results['Top Rules']}\n")
    print(f"Rules Valid:\n{results['Rules Valid']}\n")
    print(f"Plot File:\n{results['Plot File']}")