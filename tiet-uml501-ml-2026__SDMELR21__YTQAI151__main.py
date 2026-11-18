import pandas as pd
import re
from sklearn.feature_extraction.text import CountVectorizer

def clean_text(text: str) -> str:
    """
    Processes a raw string of text to prepare it for analysis.
    
    The function must perform the following sequential operations:
    1. Remove any HTML tags.
    2. Convert the entire string to lowercase.
    3. Remove any character that is not an alphabet letter (a-z) or a space.
    4. Split the text into individual words (tokenization).
    5. Join the remaining words back into a single string, separated by spaces.
    
    Args:
        text (str): The raw input string.
        
    Returns:
        str: The fully processed string.
    """
    # 1. Remove HTML tags (like <br />)
    text = re.sub(r'<.*?>', '', text)
    
    # 2. Convert to lowercase
    text = text.lower()
    
    # 3. Remove punctuation and non-alphabet characters (keep letters and spaces)
    text = re.sub(r'[^a-z\s]', '', text)
    
    # 4 & 5. Split into words and join back together to normalize whitespace
    words = text.split()
    return ' '.join(words)

def main():
    """
    Main execution function to run the data cleaning pipeline.
    """
    INPUT_FILE = 'IMDB Dataset.csv'
    OUTPUT_FILE = 'cleaned_imdb_dataset.csv'
    
    # 1. Load the dataset from INPUT_FILE.
    print(f"Loading data from '{INPUT_FILE}'...")
    try:
        df = pd.read_csv(INPUT_FILE)
    except FileNotFoundError:
        print(f"Error: Dataset not found at '{INPUT_FILE}'.")
        return # Exit the function
        
    # 2. Apply the cleaning function to the 'review' column.
    print("Applying text cleaning function to all reviews...")
    df['cleaned_review'] = df['review'].apply(clean_text)
    
    # 3. Prepare the final DataFrame for saving.
    final_df = df[['cleaned_review', 'sentiment']]
    
    # 4. Save the final DataFrame to OUTPUT_FILE.
    print(f"Saving cleaned data to '{OUTPUT_FILE}'...")
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print("Data cleaning process complete.")
    print(f"Cleaned data saved to {OUTPUT_FILE}")
    
    # 5. Vectorize the cleaned reviews using CountVectorizer
    print("\nApplying CountVectorizer with stopwords removal...")
    vectorizer = CountVectorizer(stop_words='english')
    X = vectorizer.fit_transform(final_df['cleaned_review'])
    print("Vocabulary size:", len(vectorizer.get_feature_names_out()))

# --- Script Entry Point ---
if __name__ == "__main__":
    main()