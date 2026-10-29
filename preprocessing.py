"""
Step 2: Text Preprocessing (NLP)
Implements lowercasing, URL/HTML/punctuation removal, tokenization,
stop word removal, and stemming/lemmatization.
Serverless-safe (Vercel / AWS Lambda compatible) with built-in fallback
stopwords and regex tokenization so cold starts require zero network downloads.
"""

import os
import re
import string
import nltk
from nltk.stem import PorterStemmer

# Built-in standard English stopwords fallback for instant serverless cold starts
BUILTIN_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "couldn", "d", "did", "didn",
    "do", "does", "doesn", "doing", "don", "down", "during", "each", "few", "for",
    "from", "further", "had", "hadn", "has", "hasn", "have", "haven", "having",
    "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
    "if", "in", "into", "is", "isn", "it", "its", "itself", "just", "ll", "m",
    "ma", "me", "mightn", "more", "most", "mustn", "my", "myself", "needn", "no",
    "nor", "not", "now", "o", "of", "off", "on", "once", "only", "or", "other",
    "our", "ours", "ourselves", "out", "over", "own", "re", "s", "same", "shan",
    "she", "should", "shouldn", "so", "some", "such", "t", "than", "that", "the",
    "their", "theirs", "them", "themselves", "then", "there", "these", "they",
    "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very",
    "was", "wasn", "we", "were", "weren", "what", "when", "where", "which",
    "while", "who", "whom", "why", "will", "with", "won", "wouldn", "y", "you",
    "your", "yours", "yourself", "yourselves",
}


def _load_stopwords() -> set:
    """Load NLTK stopwords if present locally; otherwise use built-in set (serverless-safe)."""
    try:
        from nltk.corpus import stopwords
        return set(stopwords.words("english"))
    except LookupError:
        # On serverless platforms (like Vercel), /tmp is the only writable directory
        if not os.environ.get("VERCEL"):
            try:
                nltk.download("stopwords", quiet=True)
                from nltk.corpus import stopwords
                return set(stopwords.words("english"))
            except Exception:
                pass
        return BUILTIN_STOPWORDS


STOP_WORDS = _load_stopwords()
STEMMER = PorterStemmer()


def _tokenize(text: str) -> list:
    """Fast regex word tokenizer that matches NLTK word_tokenize on cleaned alpha text."""
    return re.findall(r"\b[a-z]+\b", text)


def preprocess_text(text: str, use_lemmatization: bool = False) -> str:
    """
    Clean and normalize raw email/SMS text through 5 standard NLP stages:
    1. Lowercasing
    2. Removing HTML tags, URLs, email addresses, punctuation & special characters
    3. Tokenization
    4. Stop Word Removal
    5. Stemming
    """
    if not isinstance(text, str):
        return ""

    # 1. Lowercasing
    text = text.lower()

    # 2. Remove HTML tags, URLs, numbers/special symbols, and punctuation
    text = re.sub(r"<[^>]+>", " ", text)  # Strip HTML tags
    text = re.sub(r"http\S+|www\.\S+", " httpaddr ", text)  # Normalize URLs
    text = re.sub(r"\S+@\S+", " emailaddr ", text)  # Normalize email addresses
    text = re.sub(r"[$£€¥]\s*\d+([.,]\d+)?|\d+\s*[$£€¥]", " moneysymb ", text)  # Currency
    text = re.sub(r"\b\d+\b", " numbr ", text)  # Numbers
    text = text.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    text = re.sub(r"[^a-z\s]", " ", text)

    # 3. Tokenization
    tokens = _tokenize(text)

    # 4. Stop Word Removal & short token filtering
    filtered_tokens = [
        token for token in tokens if token not in STOP_WORDS and len(token) > 1
    ]

    # 5. Stemming
    processed_tokens = [STEMMER.stem(token) for token in filtered_tokens]

    return " ".join(processed_tokens)


def preprocess_steps_breakdown(text: str) -> dict:
    """
    Return intermediate outputs of each preprocessing stage for educational
    inspection in the UI.
    """
    lowered = text.lower()
    cleaned = re.sub(r"<[^>]+>", " ", lowered)
    cleaned = re.sub(r"http\S+|www\.\S+", " httpaddr ", cleaned)
    cleaned = re.sub(r"\S+@\S+", " emailaddr ", cleaned)
    cleaned = re.sub(r"[$£€¥]\s*\d+([.,]\d+)?|\d+\s*[$£€¥]", " moneysymb ", cleaned)
    cleaned = re.sub(r"\b\d+\b", " numbr ", cleaned)
    cleaned = cleaned.translate(str.maketrans(string.punctuation, " " * len(string.punctuation)))
    cleaned = re.sub(r"[^a-z\s]", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    tokens = _tokenize(cleaned)
    no_stopwords = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    stemmed = [STEMMER.stem(t) for t in no_stopwords]

    return {
        "1_lowercased": lowered,
        "2_cleaned_no_punct": cleaned,
        "3_tokens": tokens,
        "4_without_stopwords": no_stopwords,
        "5_stemmed_final": " ".join(stemmed),
    }
