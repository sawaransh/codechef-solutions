
import pandas as pd


class DataWrangling:

    def __init__(self, filepath):
        """Initialize the DataWrangling class."""
        self.filepath = filepath
        self.df = pd.read_csv(filepath)

    def hierarchical_indexing(self):
        """Create hierarchical index using two columns."""
        cols = self.df.columns.tolist()

        result = self.df.set_index([cols[0], cols[1]])

        return result

    def combine_dataframes(self):
        """Split DataFrame and recombine using concatenation."""
        mid = len(self.df) // 2

        df1 = self.df.iloc[:mid].copy()
        df2 = self.df.iloc[mid:].copy()

        combined = pd.concat([df1, df2], ignore_index=True)

        return combined

    def merge_dataframes(self):
        """Create two DataFrames and merge using a common column."""
        cols = self.df.columns.tolist()

        common_col = cols[0]

        mid = len(self.df.columns) // 2

        left_cols = cols[:mid]
        right_cols = [common_col] + [
            col for col in cols[mid:] if col != common_col
        ]

        left_df = self.df[left_cols].copy()
        right_df = self.df[right_cols].copy()

        merged = pd.merge(
            left_df,
            right_df,
            on=common_col,
            how="inner"
        )

        return merged

    def reshape_data(self):
        """Reshape the DataFrame from wide to long format."""
        cols = self.df.columns.tolist()

        id_vars = cols[:2]
        value_vars = cols[2:]

        reshaped = pd.melt(
            self.df,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name="Variable",
            value_name="Value"
        )

        return reshaped

    def pivot_table(self):
        """Create pivot table summarizing sales."""
        cols = self.df.columns.tolist()

        index_col = cols[0]
        column_col = cols[1]

        numeric_cols = self.df.select_dtypes(
            include="number"
        ).columns.tolist()

        value_col = numeric_cols[0] if numeric_cols else cols[-1]

        pivot = pd.pivot_table(
            self.df,
            index=index_col,
            columns=column_col,
            values=value_col,
            aggfunc="sum"
        )

        return pivot


def run_wrangling(filepath):

    wrangler = DataWrangling(filepath)

    hierarchical = wrangler.hierarchical_indexing()
    combined = wrangler.combine_dataframes()
    merged = wrangler.merge_dataframes()
    reshaped = wrangler.reshape_data()
    pivot = wrangler.pivot_table()

    return {
        "hierarchical": hierarchical,
        "combined": combined,
        "merged": merged,
        "reshaped": reshaped,
        "pivot": pivot
    }


if __name__ == "__main__":

    filepath = "sales_data.csv"

    results = run_wrangling(filepath)

    print("\n=== Hierarchical Indexing ===")
    print(results["hierarchical"])

    print("\n=== Combined DataFrames ===")
    print(results["combined"])

    print("\n=== Merged DataFrames ===")
    print(results["merged"])

    print("\n=== Reshaped Data ===")
    print(results["reshaped"])

    print("\n=== Pivot Table ===")
    print(results["pivot"])
