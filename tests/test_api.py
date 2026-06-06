"""
TexScanner - API Tests
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import json
from api.app import app


@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as c:
        yield c


class TestHealth:
    def test_health_ok(self, client):
        r = client.get('/health')
        assert r.status_code == 200
        data = r.get_json()
        assert data['status'] == 'ok'

    def test_health_has_models(self, client):
        r = client.get('/health')
        data = r.get_json()
        assert 'models_loaded' in data
        assert len(data['models_loaded']) > 0


class TestClassify:
    def test_spam_detected(self, client):
        r = client.post('/classify', json={
            'text': 'CONGRATULATIONS!!! You WON $1,000,000! Click HERE to CLAIM your FREE prize NOW!!!'
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data['prediction'] == 'spam'
        assert data['is_spam'] is True
        assert data['confidence']['spam'] > 50

    def test_ham_detected(self, client):
        r = client.post('/classify', json={
            'text': 'Hi, can we reschedule tomorrow meeting to 3pm? The design review will take longer than expected.'
        })
        assert r.status_code == 200
        data = r.get_json()
        assert data['prediction'] == 'ham'
        assert data['is_spam'] is False

    def test_no_text_returns_400(self, client):
        r = client.post('/classify', json={})
        assert r.status_code == 400

    def test_empty_text_returns_400(self, client):
        r = client.post('/classify', json={'text': '   '})
        assert r.status_code == 400

    def test_response_has_confidence(self, client):
        r = client.post('/classify', json={'text': 'Hello world this is a test email'})
        data = r.get_json()
        assert 'confidence' in data
        assert 'spam' in data['confidence']
        assert 'ham' in data['confidence']
        assert abs(data['confidence']['spam'] + data['confidence']['ham'] - 100) < 0.1

    def test_response_has_analysis(self, client):
        r = client.post('/classify', json={'text': 'FREE OFFER!!! Click here now!!!', 'analysis': True})
        data = r.get_json()
        assert 'analysis' in data
        analysis = data['analysis']
        assert 'word_count' in analysis
        assert 'exclamation_marks' in analysis

    def test_model_selection(self, client):
        for model in ['svm', 'naive_bayes', 'best']:
            r = client.post('/classify', json={'text': 'Test email', 'model': model})
            assert r.status_code == 200
            data = r.get_json()
            assert data['model_used'] is not None


class TestBatchClassify:
    def test_batch_classification(self, client):
        emails = [
            'FREE MONEY! WIN NOW!!!',
            'Please review the attached report.',
            'URGENT: Your account expires TODAY! Act NOW!',
            'Looking forward to our meeting next week.',
        ]
        r = client.post('/classify/batch', json={'emails': emails})
        assert r.status_code == 200
        data = r.get_json()
        assert len(data['results']) == 4
        assert 'summary' in data
        assert data['summary']['total'] == 4

    def test_batch_empty_returns_400(self, client):
        r = client.post('/classify/batch', json={'emails': []})
        assert r.status_code == 400

    def test_batch_too_large_returns_400(self, client):
        r = client.post('/classify/batch', json={'emails': ['test'] * 101})
        assert r.status_code == 400


class TestModels:
    def test_list_models(self, client):
        r = client.get('/models')
        assert r.status_code == 200
        data = r.get_json()
        assert 'models' in data

    def test_get_metrics(self, client):
        r = client.get('/metrics')
        assert r.status_code == 200
