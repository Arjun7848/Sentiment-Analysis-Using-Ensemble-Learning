from flask import Flask, render_template, request
from predict import predict_sentiment

app = Flask(__name__)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None

    if request.method == "POST":

        text = request.form.get("text", "").strip()

        if text:
            prediction = predict_sentiment(text)

    return render_template(
        "index.html",
        prediction=prediction
    )


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5001,
        debug=True
    )