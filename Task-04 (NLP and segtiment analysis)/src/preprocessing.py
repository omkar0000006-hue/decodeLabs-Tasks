import re
import nltk
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag

# ---- one-time setup -------------------------------------------------
_REQUIRED_NLTK_PACKAGES = [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
    ("corpora/wordnet", "wordnet"),
    ("corpora/omw-1.4", "omw-1.4"),
    ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
    ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
]
for path, pkg in _REQUIRED_NLTK_PACKAGES:
    try:
        nltk.data.find(path)
    except LookupError:
        nltk.download(pkg, quiet=True)

NEGATION_WORDS = {
    "not", "no", "nor", "never", "none", "nobody", "nothing",
    "neither", "nowhere", "cannot", "can't", "won't", "don't",
    "doesn't", "didn't", "isn't", "aren't", "wasn't", "weren't",
    "haven't", "hasn't", "hadn't", "wouldn't", "shouldn't",
    "couldn't", "mustn't", "n't",
}

_STOPWORDS = set(stopwords.words("english")) - NEGATION_WORDS
_LEMMATIZER = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Lowercase, strip HTML tags and non-alphabetic noise."""
    text = str(text).lower()
    text = re.sub(r"<[^>]+>", " ", text)          # HTML tags e.g. <br>
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # URLs
    text = re.sub(r"[^a-z\s']", " ", text)         # keep letters/apostrophes
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _treebank_to_wordnet(tag: str):
    """Map a Treebank POS tag to a WordNet POS tag for the lemmatizer."""
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN  # default


def tokenize(text: str) -> list:
    return word_tokenize(text)


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]


def lemmatize(tokens: list) -> list:
    """POS-guided lemmatization (mandatory per brief: WordNetLemmatizer + POS)."""
    tagged = pos_tag(tokens)
    return [
        _LEMMATIZER.lemmatize(word, _treebank_to_wordnet(tag))
        for word, tag in tagged
    ]


def preprocess(text: str) -> str:
    """Full pipeline: clean -> tokenize -> stop-word removal -> lemmatize.
    Returns a single space-joined string ready for TF-IDF vectorization.
    """
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)


if __name__ == "__main__":
    samples = [
        "I am NOT happy with this product!! <br> It never worked.",
        "This laptop is not bad at all, it went beyond my expectations.",
    ]
    for s in samples:
        print(f"RAW : {s}")
        print(f"CLEAN: {preprocess(s)}\n")
