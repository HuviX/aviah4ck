from flask import Flask
from flask import send_file

app = Flask(__name__)


@app.route("/<path:path>")
def index(path):
    return send_file(f'../{path}', mimetype='image/gif')
