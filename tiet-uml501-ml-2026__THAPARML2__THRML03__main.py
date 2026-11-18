import pandas as pd
import numpy as np

from sklearn.preprocessing import (
    MinMaxScaler,
    StandardScaler
)


class CustomerSimilarityAnalysis:
  """
  Perform customer similarity and
  correlation analysis using a
  transformed customer dataset.
  """

  def __init__(self):
      """
      Load the dataset from
      AWCustomers.csv.
      """

      # Load the dataset
      pass

  def preprocess_data(self):
      """
      Prepare the dataset by:

      - Removing duplicate rows.
      - Filling missing numeric
        values using the median.
      - Filling missing categorical
        values using the mode.
      - Applying Min-Max
        Normalization to
        YearlyIncome.
      - Applying Z-score
        Standardization to:
          * NumberCarsOwned
          * NumberChildrenAtHome
          * TotalChildren
      - Applying One Hot Encoding
        to:
          * Gender
          * MaritalStatus
          * Occupation
          * CountryRegionName
      """

      pass

  def simple_matching_similarity(self):
      """
      Calculate the Simple Matching
      Similarity between row 0 and
      row 1 of the transformed
      dataset.

      Return the similarity value.
      """

      pass

  def jaccard_similarity(self):
      """
      Calculate the Jaccard
      Similarity between row 0 and
      row 1 using only the one-hot
      encoded binary features.

      Return the similarity value.
      """

      pass

  def cosine_similarity(self):
      """
      Calculate the Cosine
      Similarity between row 0 and
      row 1 using only the numeric
      features.

      Return the similarity value.
      """

      pass

  def feature_correlation(self):
      """
      Calculate the Pearson
      Correlation Coefficient
      between:

      - NumberCarsOwned
      - YearlyIncome

      Return the correlation value.
      """

      pass

  def get_dataset(self):
      """
      Return the transformed
      dataset.
      """

      pass


if __name__ == "__main__":

  analyzer = CustomerSimilarityAnalysis()

  analyzer.preprocess_data()

  print("Simple Matching Similarity:")
  print(analyzer.simple_matching_similarity())

  print("\nJaccard Similarity:")
  print(analyzer.jaccard_similarity())

  print("\nCosine Similarity:")
  print(analyzer.cosine_similarity())

  print("\nCorrelation:")
  print(analyzer.feature_correlation())