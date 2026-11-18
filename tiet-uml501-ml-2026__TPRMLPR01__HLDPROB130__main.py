import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class IrisVisualization:

    def __init__(self):
        """
        Initialize visualization pipeline.

        Steps students must perform:
        1. Load Iris dataset using sklearn
        2. Convert the dataset into a pandas DataFrame
        3. Assign feature names as column names
        4. Add a column representing the species label
        """
        print("Loading IRIS dataset... please wait")

        # TODO:
        # 1. Load Iris dataset
        # 2. Convert iris.data into pandas DataFrame
        # 3. Assign iris.feature_names as columns
        # 4. Add species column using iris.target
        iris = load_iris()
        self.df = pd.DataFrame(iris.data, columns=iris.feature_names)
        self.df["species"] = iris.target

    def scatter_plot(self):
        print("Generating scatter plot...")

        # TODO:
        # 1. Create a matplotlib figure
        # 2. Use seaborn scatterplot to visualize relationship between
        #    sepal length and
        #    petal length
        # 3. Color data points based on the species column
        # 4. Add an appropriate plot title
        # 5. Save the figure as "scatter_plot.png"
        # 6. Close the figure to free memory
        plt.figure()
        sns.scatterplot(
            data=self.df,
            x="sepal length (cm)",
            y="petal length (cm)",
            hue="species"
        )
        plt.title("Sepal vs Petal Length")
        plt.savefig("scatter_plot.png")
        plt.close()

        print("Scatter plot saved successfully")

    def histogram(self):
        print("Generating histogram...")

        # TODO:
        # 1. Create a figure
        # 2. Plot a histogram of one numeric feature from the dataset
        # 3. Use seaborn histplot or plt.hist
        # 4. Add a title describing the distribution
        # 5. Save plot as "histogram.png"
        # 6. Close the plot
        plt.figure()
        sns.histplot(self.df["sepal length (cm)"], kde=True)
        plt.title("Sepal Length Distribution")
        plt.savefig("histogram.png")
        plt.close()

        print("Histogram saved successfully")

    def box_plot(self):
        print("Generating box plot...")

        # TODO:
        # 1. Create a box plot comparing one numeric feature across species
        # 2. Use seaborn boxplot
        # 3. Label axes clearly and provide title
        # 4. Save the figure as "box_plot.png"
        # 5. Close the figure
        plt.figure()
        sns.boxplot(
            data=self.df,
            x="species",
            y="sepal length (cm)"
        )
        plt.title("Sepal Length by Species")
        plt.xlabel("Species")
        plt.ylabel("Sepal Length (cm)")
        plt.savefig("box_plot.png")
        plt.close()

        print("Box plot saved successfully")

    def pair_plot(self):
        print("Generating pair plot... please wait")

        # TODO:
        # 1. Use seaborn pairplot to visualize pairwise relationships
        # 2. Set the species column as hue parameter
        # 3. Save the resulting plot as "pair_plot.png"
        # 4. Close the plot
        pair_grid = sns.pairplot(self.df, hue="species")
        pair_grid.savefig("pair_plot.png")
        plt.close()

        print("Pair plot saved successfully")

    def heatmap(self):
        print("Generating correlation heatmap...")

        # TODO:
        # 1. Compute correlation matrix of the DataFrame
        # 2. Plot correlation matrix using seaborn heatmap
        # 3. Enable annotation to display correlation values
        # 4. Choose a color map for visualization
        # 5. Add a title
        # 6. Save the heatmap as "heatmap.png"
        # 7. Close the figure
        plt.figure(figsize=(8, 6))
        corr_matrix = self.df.drop(columns=["species"]).corr()
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
        plt.title("Iris Feature Correlation Heatmap")
        plt.tight_layout()
        plt.savefig("heatmap.png")
        plt.close()

        print("Heatmap saved successfully")


def run_visualizations():
    """
    This function runs the full visualization pipeline.

    Students DO NOT need to modify this function.
    It simply calls all visualization methods in sequence.
    """
    vis = IrisVisualization()

    vis.scatter_plot()
    vis.histogram()
    vis.box_plot()
    vis.pair_plot()
    vis.heatmap()

    return vis


if __name__ == "__main__":

    run_visualizations()

    print("\nAll visualizations generated successfully.")
