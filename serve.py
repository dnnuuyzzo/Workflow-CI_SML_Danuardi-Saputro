import mlflow
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load model saat startup
model = mlflow.pyfunc.load_model("model")

@app.route("/ping", methods=["GET"])
def ping():
    return jsonify({"status": "healthy"})

@app.route("/invocations", methods=["POST"])
def predict():
    data = request.get_json()
    import pandas as pd
    df = pd.DataFrame(data["dataframe_split"]["data"], columns=data["dataframe_split"]["columns"])
    predictions = model.predict(df)
    return jsonify({"predictions": predictions.tolist()})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
