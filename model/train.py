"""
TexScanner - Spam Email Classifier
Trains Naive Bayes and SVM models on email data.
"""

import os
import sys
import json
import re
import pickle
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, classification_report
)
from sklearn.calibration import CalibratedClassifierCV
import joblib
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

STOP_WORDS = set(stopwords.words('english'))
STEMMER = PorterStemmer()

SPAM_PATTERNS = [
    r'\b(free|winner|won|prize|claim|urgent|limited|offer|deal|discount)\b',
    r'\b(click here|buy now|order now|act now|call now|subscribe)\b',
    r'\b(viagra|casino|lottery|inheritance|nigerian|prince)\b',
    r'(\$\d+|\d+%\s*off|save\s*\$)',
    r'([!]{3,}|[?]{3,}|[\$]{3,})',
    r'\b(unsubscribe|opt.?out)\b',
    r'[A-Z]{5,}',
]

SPAM_SAMPLE_EMAILS = [
    ("Congratulations! You've WON $1,000,000!!! Click HERE to claim your PRIZE NOW!!!", 'spam'),
    ("URGENT: Your account has been selected for our special lottery. Send details to claim.", 'spam'),
    ("FREE VIAGRA! LIMITED TIME OFFER! Buy now and save 80%!!!", 'spam'),
    ("Make $5000 per week working from home! No experience needed! ACT NOW!!!", 'spam'),
    ("You have been pre-approved for a $50,000 loan. Bad credit OK! Apply now!", 'spam'),
    ("Dear Friend, I am a Nigerian Prince with $15 million dollars to transfer...", 'spam'),
    ("WINNER! Your email won our sweepstakes. Send your details to claim $50,000!", 'spam'),
    ("Hot singles in your area want to meet YOU! Click here for FREE access!", 'spam'),
    ("Lose 30 pounds in 30 days with our miracle pill! Order now, limited supply!", 'spam'),
    ("Your PayPal account has been limited! Verify immediately or lose access forever!", 'spam'),
    ("Get rich quick! Our system guarantees $10,000 per month! 100% guaranteed!", 'spam'),
    ("FINAL NOTICE: Your subscription expires today! Renew now to avoid charges!", 'spam'),
    ("Enlarge your... Special offer just for you! Discrete shipping guaranteed!", 'spam'),
    ("Rolex watches for $49! Authentic luxury at unbeatable prices. Order TODAY!", 'spam'),
    ("Your computer is infected! Download our FREE antivirus immediately!", 'spam'),
    ("Double your Bitcoin in 24 hours! Guaranteed returns! Invest now!", 'spam'),
    ("Work from home opportunity! Earn $500/day stuffing envelopes!", 'spam'),
    ("CHEAP SOFTWARE! Microsoft Office for $29! Adobe for $19! All legit!", 'spam'),
    ("You have been chosen! Exclusive VIP membership offer! Limited spots available!", 'spam'),
    ("Free iPhone 15! Just pay shipping! First 100 respondents only! Hurry!", 'spam'),
    ("Hi team, the meeting has been rescheduled to 3pm tomorrow. Please update your calendars.", 'ham'),
    ("Please find attached the quarterly report for your review. Let me know if you have questions.", 'ham'),
    ("Thanks for your prompt response. I'll review the documents and get back to you by Friday.", 'ham'),
    ("The project deadline has been extended to next Monday. Please adjust your timelines accordingly.", 'ham'),
    ("Can we schedule a call this week to discuss the new product roadmap?", 'ham'),
    ("I wanted to follow up on our conversation from last Tuesday regarding the budget proposal.", 'ham'),
    ("Your order #12345 has shipped and will arrive by Thursday. Track your package online.", 'ham'),
    ("Here is the feedback you requested on the design mockups. Overall looking great!", 'ham'),
    ("Please remember to submit your timesheet by end of day Friday.", 'ham'),
    ("The client approved the proposal! We're moving forward with the project.", 'ham'),
    ("Happy birthday! Hope you have a wonderful day celebrating with family and friends.", 'ham'),
    ("Could you please send me the updated contact list for the conference attendees?", 'ham'),
    ("I've reviewed your pull request and left some comments. Please address them when you get a chance.", 'ham'),
    ("The coffee machine on the third floor is out of order. Maintenance has been notified.", 'ham'),
    ("Reminder: Team lunch is tomorrow at noon at the Italian place on Main Street.", 'ham'),
    ("Your flight confirmation number is ABC123. Check-in opens 24 hours before departure.", 'ham'),
    ("Thanks for attending the webinar. Here are the slides from today's presentation.", 'ham'),
    ("Please review and sign the attached NDA before our meeting next week.", 'ham'),
    ("The server maintenance window is scheduled for this Sunday from 2-4am EST.", 'ham'),
    ("I found a bug in the authentication module. Working on a fix now.", 'ham'),
]


def preprocess_text(text: str) -> str:
    """Clean and normalize email text."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', ' URL ', text)
    text = re.sub(r'\S+@\S+', ' EMAIL ', text)
    text = re.sub(r'\$[\d,]+', ' MONEY ', text)
    text = re.sub(r'\d+%', ' PERCENT ', text)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    spam_features = []
    original_text = text
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, original_text, re.IGNORECASE):
            spam_features.append('SPAM_INDICATOR')

    tokens = text.split()
    tokens = [
        STEMMER.stem(w)
        for w in tokens
        if w not in STOP_WORDS and len(w) > 2
    ]

    return ' '.join(tokens + spam_features)


def extract_features(text: str) -> dict:
    """Extract handcrafted features for analysis."""
    features = {}
    features['length'] = len(text)
    features['exclamation_count'] = text.count('!')
    features['question_count'] = text.count('?')
    features['caps_ratio'] = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    features['dollar_count'] = text.count('$')
    features['url_count'] = len(re.findall(r'http\S+|www\S+', text))
    features['spam_word_count'] = sum(
        1 for pattern in SPAM_PATTERNS
        if re.search(pattern, text, re.IGNORECASE)
    )
    return features


def load_or_generate_data():
    """Load email dataset (uses built-in samples + optional CSV)."""
    texts, labels = zip(*SPAM_SAMPLE_EMAILS)
    df = pd.DataFrame({'text': texts, 'label': labels})
    logger.info(f"Using {len(df)} sample emails ({df['label'].value_counts().to_dict()})")
    return df


def build_naive_bayes_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ('clf', MultinomialNB(alpha=0.1)),
    ])


def build_svm_pipeline():
    return Pipeline([
        ('tfidf', TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=1,
        )),
        ('clf', CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000))),
    ])


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    """Evaluate model and return metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        'name': name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, pos_label='spam'),
        'recall': recall_score(y_test, y_pred, pos_label='spam'),
        'f1': f1_score(y_test, y_pred, pos_label='spam'),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classification_report': classification_report(y_test, y_pred),
    }

    logger.info(f"\n{'='*50}")
    logger.info(f"Model: {name}")
    logger.info(f"Accuracy:  {metrics['accuracy']:.4f}")
    logger.info(f"Precision: {metrics['precision']:.4f}")
    logger.info(f"Recall:    {metrics['recall']:.4f}")
    logger.info(f"F1 Score:  {metrics['f1']:.4f}")
    logger.info(f"\n{metrics['classification_report']}")

    return metrics


def train_and_save():
    """Main training pipeline."""
    output_dir = Path(__file__).parent / 'artifacts'
    output_dir.mkdir(exist_ok=True)

    logger.info("Loading data...")
    df = load_or_generate_data()

    logger.info("Preprocessing text...")
    df['processed'] = df['text'].apply(preprocess_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df['processed'], df['label'],
        test_size=0.25, random_state=42, stratify=df['label']
    )

    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    logger.info("Training Naive Bayes...")
    nb_model = build_naive_bayes_pipeline()
    nb_model.fit(X_train, y_train)
    results['naive_bayes'] = evaluate_model(nb_model, X_test, y_test, 'Naive Bayes')
    joblib.dump(nb_model, output_dir / 'naive_bayes_model.joblib')

    logger.info("Training SVM...")
    svm_model = build_svm_pipeline()
    svm_model.fit(X_train, y_train)
    results['svm'] = evaluate_model(svm_model, X_test, y_test, 'SVM')
    joblib.dump(svm_model, output_dir / 'svm_model.joblib')

    best_model_name = 'svm' if results['svm']['f1'] >= results['naive_bayes']['f1'] else 'naive_bayes'
    best_model = svm_model if best_model_name == 'svm' else nb_model
    joblib.dump(best_model, output_dir / 'best_model.joblib')
    logger.info(f"Best model: {best_model_name} (F1={results[best_model_name]['f1']:.4f})")

    metrics_path = output_dir / 'metrics.json'
    serializable_results = {}
    for k, v in results.items():
        serializable_results[k] = {
            kk: vv for kk, vv in v.items()
            if kk != 'classification_report'
        }
    serializable_results['best_model'] = best_model_name

    with open(metrics_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"Artifacts saved to {output_dir}")
    return results


if __name__ == '__main__':
    train_and_save()
