# Project 4: NLP & Sentiment Analysis

Goal: build a text pre-processing → TF-IDF → Naive Bayes/SVM pipeline that
classifies product reviews as Positive or Negative.

## Important note on the dataset

`Dataset_for_Data_Analytics_-_Sheet1.csv` is an **e-commerce orders**
dataset (OrderID, Product, Price, Shipping, OrderStatus...) — it has no
free-text review column, which this project needs. `data/generate_reviews.py`
converts each order row into a synthetic review sentence (using
`OrderStatus` to bias sentiment) so the pipeline has text to work with.

**For your internship submission**, mention this clearly, or swap in a
real review dataset (e.g. Amazon/Flipkart product reviews) — just replace
`data/reviews.csv` with a file that has `review_text` and `sentiment`
columns and everything downstream works unchanged.

Also note: because the synthetic reviews are template-based, the model
hits ~100% accuracy in testing. That's an artifact of the synthetic data
(too few templates = trivial pattern), not a realistic result — say so if
asked, and consider swapping in a real dataset for a genuine 80–90%-style
result.

## Project structure

```
Project4_NLP_Sentiment/
├── data/
│   ├── orders_dataset.csv       # original uploaded data
│   ├── generate_reviews.py      # builds synthetic reviews from orders
│   └── reviews.csv              # generated: review_text + sentiment
├── src/
│   ├── preprocessing.py         # clean -> tokenize -> stopwords -> lemmatize
│   ├── train.py                 # TF-IDF + train/evaluate NB & SVM
│   └── predict.py               # load saved model, predict new text
├── models/
│   ├── sentiment_model.joblib   # generated after training
│   ├── tfidf_vectorizer.joblib  # generated after training
│   └── model_info.txt
├── requirements.txt
└── README.md
```

## How to run

```bash
pip install -r requirements.txt

cd data
python generate_reviews.py      # creates reviews.csv

cd ../src
python train.py                 # trains + saves model to ../models
python predict.py                              # runs demo predictions
python predict.py "This laptop is not good"    # predict your own text
```

## Pipeline details (matches the brief)

1. **Pre-processing** (`preprocessing.py`)
   - Lowercase, strip HTML/URLs/punctuation
   - Tokenize (NLTK `word_tokenize`)
   - Stop-word removal — **negations excluded** from the stop-word set
     (`not`, `never`, `n't`, etc.) so "not happy" isn't reduced to "happy"
   - Lemmatization with `WordNetLemmatizer`, guided by POS tags (Treebank
     → WordNet mapping) so verbs/adjectives reduce correctly

2. **Vectorization** (`train.py`)
   - `TfidfVectorizer(ngram_range=(1,2), max_features=5000, min_df=2)`
   - Bigrams capture negated phrases ("not good") as one feature
   - `max_features`/`min_df` bound the sparse matrix size

3. **Classification** (`train.py`)
   - Trains `MultinomialNB`, `ComplementNB`, and `LinearSVC`
   - Picks whichever has the best test accuracy, saves it with `joblib`

4. **Inference** (`predict.py`)
   - Loads the saved model + vectorizer, preprocesses new text the same
     way, and predicts Positive/Negative
