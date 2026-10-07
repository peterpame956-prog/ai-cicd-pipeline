import time
from sklearn.metrics import accuracy_score

def test_model_accuracy_gate(trained_model, processed_data):
    X, y = processed_data
    predictions = trained_model.predict(X)
    assert accuracy_score(y, predictions) >= 0.85

def test_single_sample_latency_budget(trained_model, processed_data):
    X, _ = processed_data
    sample = X.iloc[[0]]
    for _ in range(5):
        trained_model.predict(sample)
    iterations = 50
    start = time.perf_counter()
    for _ in range(iterations):
        trained_model.predict(sample)
    avg_latency_ms = ((time.perf_counter() - start)/iterations)*1000
    assert avg_latency_ms < 50, f"Single-prediction latency {avg_latency_ms:.2f} ms exceeds 50ms"
