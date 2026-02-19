import os

from flask import Flask, redirect

from protracker import protracker_bp

APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("PORT", os.getenv("APP_PORT", "8080")))

app = Flask(__name__, template_folder="templates", static_folder="../static")
app.register_blueprint(protracker_bp)


@app.get("/")
def root_redirect():
    return redirect("/atari-tracker", code=302)


if __name__ == "__main__":
    app.run(host=APP_HOST, port=APP_PORT)
