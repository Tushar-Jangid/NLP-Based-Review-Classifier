from flask import Flask,request,jsonify
import joblib

app = Flask(__name__)

model = joblib.load(r"D:\DP learning\Project 1\model.pkl")
vectorizer = joblib.load(r"D:\DP learning\Project 1\tfidf_vectorizer.pkl")

@app.route('/predict', methods=["GET",'POST'])
def predict():
    data = request.json
    text = data.get("text", "")

    x = vectorizer.transform([text])
    prediction = model.predict(x)[0]


    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(debug=True)