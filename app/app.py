from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello_world():
    request_type_str = request.method
    if request_type_str == "GET":
        return render_template("index.html", href="static/base_pic.svg")
    else:
        text = request.form["text"]
        random_str = uuid.uuid4().hex

        path = "app/static/" + random_str + ".svg"
        model = load("app/model.joblib")
        np_arr = floats_string_to_np_arr(text)
        make_picture("app/AgesAndHeights.pkl", model, np_arr, path)
        
        return render_template("index.html", href=path[4:])

def make_picture(training_data_filename, model, new_inp_np_arr, output_file):
    df = pd.read_pickle(training_data_filename)

    ages = df["Age"]
    df = df[ages > 0]
    ages = df["Age"]
    heights = df["Height"]

    x_new = np.array(list(range(19))).reshape(-1, 1)
    preds = model.predict(x_new)

    fig = px.scatter(x=ages, y=heights, title="Height vs. Age", labels={"x":"Age (years)", "y":"Height (inches)"})
    fig.add_trace(go.Scatter(x=x_new.reshape(-1), y=preds, mode="lines", name="Model"))

    new_preds = model.predict(new_inp_np_arr)
    fig.add_trace(go.Scatter(x=new_inp_np_arr.reshape(-1), y=new_preds, mode="markers", marker=dict(color="purple", size=15)))

    fig.write_image(output_file, width = 800, engine="kaleido")
    fig.show()

def floats_string_to_np_arr(floates_str):
    floats=list()

    for x in floates_str.split(","):
        try:
            float(x)
            floats.append(float(x))
        except: pass 

    return np.array(floats).reshape(-1, 1)