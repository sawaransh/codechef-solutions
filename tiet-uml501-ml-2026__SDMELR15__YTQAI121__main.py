import re
import nltk
from nltk.corpus import stopwords
import pandas as pd

# -- Initial Setup: NLTK Stopwords --
# A one-time check and download for the NLTK stopwords list.
try:
  STOP_WORDS = set(stopwords.words('english'))
except LookupError:
  print('Downloading NLTK stopwords...')
  nltk.download('stopwords')
  STOP_WORDS = set(stopwords.words('english'))


def clean_text(text: str) -> str:
  # 1. Remove any HTML tags
  text = re.sub(r'<[^>]+>', '', str(text))

  # 2. Convert the entire string to lowercase
  text = text.lower()

  # 3. Remove any character that is not an alphabet letter (a-z) or a space
  text = re.sub(r'[^a-z ]', '', text)

  # 4. Split the text into individual words (tokenization)
  words = text.split()

  # 5. Remove all English stop words from the list of words
  words = [word for word in words if word not in STOP_WORDS]

  # 6. Join the remaining words back into a single string, separated by spaces
  cleaned_text = ' '.join(words)

  return cleaned_text


def main():
  INPUT_FILE = 'IMDB Dataset.csv'
  OUTPUT_FILE = 'cleaned_imdb_dataset.csv'

  # 1. Load the dataset from INPUT_FILE.
  print(f"Loading data from '{INPUT_FILE}'...")
  try:
    df = pd.read_csv(INPUT_FILE)
  except FileNotFoundError:
    print(f"Error: Dataset not found at '{INPUT_FILE}'.")
    print('Please download it and place it in the correct directory.')
    return

  # 2. Apply the cleaning function to the 'review' column.
  print('Applying text cleaning function to all reviews...')
  df['cleaned_review'] = df['review'].apply(clean_text)

  # 3. Prepare the final DataFrame for saving.
  final_df = df[['cleaned_review', 'sentiment']]

  # 4. Save the final DataFrame to OUTPUT_FILE.
  print(f"Saving cleaned data to '{OUTPUT_FILE}'...")
  final_df.to_csv(OUTPUT_FILE, index=False)

  print('Data cleaning process complete.')
  print(f'Cleaned data saved to {OUTPUT_FILE}')


# -- Script Entry Point --
if __name__ == '__main__':
  main()