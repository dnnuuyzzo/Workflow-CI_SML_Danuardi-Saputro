import mlflow
import pandas as pd
from flask import Flask, request, jsonify, Response
from prometheus_client import (
    Counter, Histogram, Gauge,
    generate_latest, CONTENT_TYPE_LATEST
)
import time
import os

app = Flask(__name__)

# ─── Prometheus Metrics ────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "model_requests_total",
    "Total jumlah request prediksi yang masuk"
)
ERROR_COUNT = Counter(
    "model_errors_total",
    "Total jumlah request yang menghasilkan error"
)
CHURN_COUNT = Counter(
    "model_predictions_churn_total",
    "Total prediksi hasil Churn (1)"
)
NOT_CHURN_COUNT = Counter(
    "model_predictions_not_churn_total",
    "Total prediksi hasil Not Churn (0)"
)
INFERENCE_LATENCY = Histogram(
    "model_inference_latency_seconds",
    "Waktu eksekusi prediksi dalam detik",
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
)
BATCH_SIZE = Histogram(
    "model_batch_size",
    "Jumlah baris data per request prediksi"
)
MODEL_UP = Gauge(
    "model_is_up",
    "Status model: 1 = berjalan, 0 = error"
)

# ─── Load Model ────────────────────────────────────────────────────────────────
# Path relatif ke folder /app di dalam container (di-mount dari Workflow-CI/)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model")

try:
    model = mlflow.pyfunc.load_model(MODEL_PATH)
    MODEL_UP.set(1)
    print(f"[OK] Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    MODEL_UP.set(0)
    model = None
    print(f"[ERROR] Gagal memuat model: {e}")


# ─── Endpoints ─────────────────────────────────────────────────────────────────

@app.route("/ping", methods=["GET"])
def ping():
    """Health check endpoint untuk container orchestrator."""
    if model is not None:
        return jsonify({"status": "healthy"}), 200
    return jsonify({"status": "unhealthy", "reason": "model not loaded"}), 503


@app.route("/metrics", methods=["GET"])
def metrics():
    """Endpoint yang di-scrape oleh Prometheus."""
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/invocations", methods=["POST"])
def predict():
    """Endpoint prediksi utama, menerima format dataframe_split."""
    REQUEST_COUNT.inc()
    start = time.time()

    try:
        if model is None:
            ERROR_COUNT.inc()
            MODEL_UP.set(0)
            return jsonify({"error": "Model belum dimuat"}), 503

        data = request.get_json(force=True)
        df = pd.DataFrame(
            data["dataframe_split"]["data"],
            columns=data["dataframe_split"]["columns"]
        )

        BATCH_SIZE.observe(len(df))

        predictions = model.predict(df)

        for p in predictions:
            if int(p) == 1:
                CHURN_COUNT.inc()
            else:
                NOT_CHURN_COUNT.inc()

        MODEL_UP.set(1)
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        ERROR_COUNT.inc()
        MODEL_UP.set(0)
        return jsonify({"error": str(e)}), 500

    finally:
        INFERENCE_LATENCY.observe(time.time() - start)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)