import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris


class IrisVisualization:

    def __init__(self):
        """
        Initialize visualization pipeline.

        Steps students must perform:
        1. Load the IRIS dataset using sklearn
        2. Convert the dataset into a pandas DataFrame
        3. Assign feature names as column names
        4. Add a column representing the species label
        """

        print("Loading IRIS dataset... please wait")

        # TODO:
        # - Load iris dataset
        # - Convert iris.data into pandas DataFrame
        # - Use iris.feature_names as column names
        # - Add species column using iris.target

        ______


    def scatter_plot(self):

        print("Generating scatter plot...")

        # TODO:
        # 1. Create a matplotlib figure
        # 2. Use seaborn scatterplot to visualize relationship between:
        #       sepal length
        #       petal length
        # 3. Use the DataFrame created in the constructor
        # 4. Add an appropriate plot title
        # 5. Save the plot as "scatter_plot.png"
        # 6. Close the figure to free memory

        ______

        print("Scatter plot saved successfully")


    def histogram(self):

        print("Generating histogram...")

        # TODO:
        # 1. Create a figure
        # 2. Plot a histogram of one numeric feature from the dataset
        # 3. Use seaborn histogram function
        # 4. Add title describing the distribution
        # 5. Save plot as "histogram.png"
        # 6. Close the plot

        ______

        print("Histogram saved successfully")


    def box_plot(self):

        print("Generating box plot...")

        # TODO:
        # 1. Create a box plot comparing one numeric feature across species
        # 2. Use seaborn boxplot
        # 3. Select appropriate x-axis and y-axis columns
        # 4. Add a title explaining the plot
        # 5. Save the figure as "box_plot.png"
        # 6. Close the figure

        ______

        print("Box plot saved successfully")


    def pair_plot(self):

        print("Generating pair plot... please wait")

        # TODO:
        # 1. Use seaborn pairplot to visualize pairwise relationships
        # 2. Use the species column as hue parameter
        # 3. Save the resulting plot as "pair_plot.png"
        # 4. Close the plot

        ______

        print("Pair plot saved successfully")


    def heatmap(self):

        print("Generating correlation heatmap...")

        # TODO:
        # 1. Compute correlation matrix of the DataFrame
        # 2. Create a heatmap using seaborn
        # 3. Enable annotation to display correlation values
        # 4. Choose a color map for visualization
        # 5. Add a plot title
        # 6. Save the heatmap as "heatmap.png"
        # 7. Close the figure

        ______

        print("Heatmap saved successfully")


def run_visualizations():

    """
    This function runs the full visualization pipeline.

    Students DO NOT need to modify this function.
    It simply calls all visualization methods in sequence.
    """

    viz = IrisVisualization()

    viz.scatter_plot()
    viz.histogram()
    viz.box_plot()
    viz.pair_plot()
    viz.heatmap()

    return True


if __name__ == "__main__":

    run_visualizations()

    print("\nAll visualizations generated successfully.")