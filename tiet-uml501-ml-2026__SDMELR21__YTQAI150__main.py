import pandas as pd
import re
from collections import Counter
from typing import Dict, List, Tuple, Any


def task_1_load_and_inspect(file_path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        raise


def task_2_check_dataset_health(df: pd.DataFrame) -> Dict[str, int]:
    return {
        "total_reviews": int(len(df)),
        "missing_reviews": int(df['review'].isnull().sum()),
        "missing_sentiments": int(df['sentiment'].isnull().sum())
    }


def task_3_analyze_sentiment_distribution(df: pd.DataFrame) -> pd.Series:
    return df['sentiment'].value_counts()


def task_4_analyze_review_length(df: pd.DataFrame) -> Dict[str, float]:
    df_copy = df.copy()
    df_copy['word_count'] = df_copy['review'].astype(str).str.split().str.len()

    overall_avg = float(df_copy['word_count'].mean())
    pos_avg = float(df_copy[df_copy['sentiment'] == 'positive']['word_count'].mean())
    neg_avg = float(df_copy[df_copy['sentiment'] == 'negative']['word_count'].mean())

    return {
        "overall_average_word_count": overall_avg,
        "average_positive_word_count": pos_avg,
        "average_negative_word_count": neg_avg
    }