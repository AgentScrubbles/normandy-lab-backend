from flask import Flask, make_response, request, send_file
from io import BytesIO
import json
from trainer import Trainer
import os
from base_schema import BaseSchema, all_schemas, get_schema

# Supported Schemas
from schemas.me3_female import ME3_Female_Schema # type: ignore


from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

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
    return response


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)