"""
TexScanner API - Spam Email Classifier Backend
FastAPI-compatible Flask server for spam detection.
"""

import os
import re
import json
import logging
import time
from pathlib import Path
from functools import wraps
from typing import Optional

import joblib
import numpy as np
from flask import Flask, request, jsonify, g
from flask_cors import CORS
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ── Setup ─────────────────────────────────────────────────────────────────────
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

SPAM_KEYWORDS = [
    'free', 'winner', 'won', 'prize', 'claim', 'urgent', 'limited', 'offer',
    'deal', 'discount', 'click here', 'buy now', 'order now', 'act now',
    'call now', 'subscribe', 'viagra', 'casino', 'lottery', 'inheritance',
    'nigerian', 'prince', 'unsubscribe', 'opt-out', 'congratulations',
    'selected', 'guaranteed', 'million', 'billion', 'rich', 'earn money',
]

app = Flask(__name__)
CORS(app, origins=['*'])

MODEL_DIR = Path(__file__).parent.parent / 'model' / 'artifacts'
METRICS_FILE = MODEL_DIR / 'metrics.json'

models = {}
metrics = {}


def load_models():
    global models, metrics
    try:
        for name in ['svm', 'naive_bayes', 'best']:
            path = MODEL_DIR / f'{name}_model.joblib'
            if path.exists():
                models[name] = joblib.load(path)
                logger.info(f"Loaded model: {name}")

        if METRICS_FILE.exists():
            with open(METRICS_FILE) as f:
                metrics = json.load(f)
            logger.info("Loaded metrics")

    except Exception as e:
        logger.error(f"Error loading models: {e}")


def preprocess_text(text: str) -> str:
    text_lower = text.lower()
    processed = text_lower
    processed = re.sub(r'http\S+|www\S+', ' URL ', processed)
    processed = re.sub(r'\S+@\S+', ' EMAIL ', processed)
    processed = re.sub(r'\$[\d,]+', ' MONEY ', processed)
    processed = re.sub(r'\d+%', ' PERCENT ', processed)
    processed = re.sub(r'[^a-zA-Z\s]', ' ', processed)

    spam_features = []
    for pattern in SPAM_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            spam_features.append('SPAM_INDICATOR')

    tokens = processed.split()
    tokens = [
        STEMMER.stem(w)
        for w in tokens
        if w not in STOP_WORDS and len(w) > 2
    ]

    return ' '.join(tokens + spam_features)


def extract_analysis(text: str) -> dict:
    """Detailed analysis of email for transparency."""
    found_keywords = [kw for kw in SPAM_KEYWORDS if kw.lower() in text.lower()]
    urls = re.findall(r'http\S+|www\S+', text)
    exclamations = text.count('!')
    questions = text.count('?')
    dollars = len(re.findall(r'\$[\d,]+', text))
    caps_words = re.findall(r'\b[A-Z]{3,}\b', text)
    caps_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    pattern_matches = []
    pattern_labels = [
        'Urgency/Offer words', 'Action phrases', 'Known spam topics',
        'Money references', 'Excessive punctuation', 'Opt-out language', 'ALL CAPS words'
    ]
    for i, pattern in enumerate(SPAM_PATTERNS):
        if re.search(pattern, text, re.IGNORECASE):
            pattern_matches.append(pattern_labels[i])

    return {
        'spam_keywords': found_keywords[:10],
        'url_count': len(urls),
        'exclamation_marks': exclamations,
        'question_marks': questions,
        'dollar_signs': dollars,
        'caps_words': caps_words[:5],
        'caps_ratio': round(caps_ratio * 100, 1),
        'pattern_matches': pattern_matches,
        'word_count': len(text.split()),
        'char_count': len(text),
    }


@app.before_request
def start_timer():
    g.start_time = time.time()


@app.after_request
def add_headers(response):
    elapsed = (time.time() - g.get('start_time', time.time())) * 1000
    response.headers['X-Response-Time'] = f'{elapsed:.1f}ms'
    response.headers['X-Powered-By'] = 'TexScanner'
    return response


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'models_loaded': list(models.keys()),
        'version': '1.0.0',
    })


@app.route('/classify', methods=['POST'])
def classify():
    """Main classification endpoint."""
    data = request.get_json(silent=True) or {}
    text = data.get('text', '').strip()
    model_name = data.get('model', 'best')
    include_analysis = data.get('analysis', True)

    if not text:
        return jsonify({'error': 'No email text provided'}), 400

    if len(text) > 50000:
        return jsonify({'error': 'Email text too long (max 50,000 chars)'}), 400

    if model_name not in models:
        model_name = 'best' if 'best' in models else list(models.keys())[0] if models else None

    if model_name is None:
        return jsonify({'error': 'No models available. Run training first.'}), 503

    model = models[model_name]
    processed = preprocess_text(text)

    prediction = model.predict([processed])[0]
    probabilities = model.predict_proba([processed])[0]

    classes = list(model.classes_)
    spam_idx = classes.index('spam') if 'spam' in classes else 1
    ham_idx = classes.index('ham') if 'ham' in classes else 0

    spam_prob = float(probabilities[spam_idx])
    ham_prob = float(probabilities[ham_idx])

    if spam_prob > 0.85:
        confidence_level = 'Very High'
    elif spam_prob > 0.70:
        confidence_level = 'High'
    elif spam_prob > 0.55:
        confidence_level = 'Moderate'
    else:
        confidence_level = 'Low'

    response = {
        'prediction': prediction,
        'is_spam': prediction == 'spam',
        'confidence': {
            'spam': round(spam_prob * 100, 2),
            'ham': round(ham_prob * 100, 2),
            'level': confidence_level,
        },
        'model_used': model_name,
    }

    if include_analysis:
        response['analysis'] = extract_analysis(text)

    return jsonify(response)


@app.route('/classify/batch', methods=['POST'])
def classify_batch():
    """Batch classification of multiple emails."""
    data = request.get_json(silent=True) or {}
    emails = data.get('emails', [])
    model_name = data.get('model', 'best')

    if not emails:
        return jsonify({'error': 'No emails provided'}), 400

    if len(emails) > 100:
        return jsonify({'error': 'Max 100 emails per batch'}), 400

    if model_name not in models:
        model_name = list(models.keys())[0] if models else None

    if model_name is None:
        return jsonify({'error': 'No models available'}), 503

    model = models[model_name]
    results = []

    for i, email_text in enumerate(emails):
        if not isinstance(email_text, str):
            results.append({'index': i, 'error': 'Invalid email format'})
            continue

        processed = preprocess_text(email_text)
        prediction = model.predict([processed])[0]
        probabilities = model.predict_proba([processed])[0]

        classes = list(model.classes_)
        spam_idx = classes.index('spam') if 'spam' in classes else 1

        results.append({
            'index': i,
            'prediction': prediction,
            'is_spam': prediction == 'spam',
            'spam_probability': round(float(probabilities[spam_idx]) * 100, 2),
        })

    spam_count = sum(1 for r in results if r.get('is_spam'))
    return jsonify({
        'results': results,
        'summary': {
            'total': len(results),
            'spam': spam_count,
            'ham': len(results) - spam_count,
            'spam_rate': round(spam_count / len(results) * 100, 1) if results else 0,
        },
        'model_used': model_name,
    })


@app.route('/models', methods=['GET'])
def list_models():
    """List available models and their metrics."""
    model_info = {}
    for name in models:
        metric_key = name if name in metrics else None
        m = metrics.get(metric_key, {}) if metric_key else {}
        model_info[name] = {
            'available': True,
            'accuracy': round(m.get('accuracy', 0) * 100, 2) if m else None,
            'precision': round(m.get('precision', 0) * 100, 2) if m else None,
            'recall': round(m.get('recall', 0) * 100, 2) if m else None,
            'f1': round(m.get('f1', 0) * 100, 2) if m else None,
        }

    return jsonify({
        'models': model_info,
        'best_model': metrics.get('best_model', 'unknown'),
    })


@app.route('/metrics', methods=['GET'])
def get_metrics():
    return jsonify(metrics)


@app.errorhandler(404)
def not_found(e):
    return jsonify({'error': 'Endpoint not found'}), 404


@app.errorhandler(500)
def server_error(e):
    return jsonify({'error': 'Internal server error'}), 500


load_models()

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)
