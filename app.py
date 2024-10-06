from flask import Flask, render_template, request
import pickle
import pandas as pd

app = Flask(__name__)

with open('model.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        operator = int(request.form.get("operator"))
        inout_travelling = int(request.form.get('inout_travelling'))
        network_type = int(request.form.get("network_type"))
        state_name = int(request.form.get("state_name"))

       
        test_data = {
            'operator': [operator],
            'network_type': [network_type],
            'state_name': [state_name],
            'inout_travelling': [inout_travelling],
        }

        X_test = pd.DataFrame(test_data)

        y_pred = loaded_model.predict(X_test)[0]

        return render_template('index.html', prediction_text=y_pred)

if __name__ == "__main__":
    app.run()
