from flask import Flask, Response, make_response, request, send_file
from io import BytesIO
import json
from trainer import Trainer
import os
from base_schema import BaseSchema, all_schemas, get_schema

# Supported Schemas
from schemas.me3_female import ME3_Female_Schema # type: ignore
from schemas.me1_female import ME1_Female_Schema
from schemas.me1_male import ME1_Male_Schema
from flask_cors import CORS
from prometheus_client import Counter, generate_latest, CONTENT_TYPE_LATEST
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)
CORS(app)

schema_counter = Counter(
    "inference_requests_total",
    "Number of inference requests per schema",
    ["schema"]
)
# --------------------------

def get_schema_checkpoint(schema: str):
    ckpt = os.getenv(f'CKPT_{schema.upper()}')
    if ckpt is None:
        raise KeyError(f'Schema {schema} does not have a matching checkpoint variable')
    return ckpt

@app.route("/schemas", methods=["GET"])
def get_schemas():
    return all_schemas()


@app.route("/infer/<schema>", methods=["POST"])
def upload(schema: str):
    file = next(iter(request.files.values())) 
    image_bytes = BytesIO(file.read())

    trainer = Trainer(schema)

    res = trainer.infer(image_source=image_bytes, ckpt_location=get_schema_checkpoint(schema), verbose=False)
    buf = BytesIO()
    res[1].save(buf, format="JPEG")
    buf.seek(0)
    response = make_response(send_file(buf, mimetype="image/png"))

    # Add custom headers
    response.headers["X-Status-Code"] = "200"
    response.headers["X-Code"] = res[0]

    # Increment prometheus counter
    schema_counter.labels(schema=schema).inc()
    return response


@app.route("/metrics")
def metrics():
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)