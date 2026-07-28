import os
import gradio as gr
import pandas as pd
import joblib

# ==========================================
# Developed By : Parth
# Roll No      : 241504
# College      : PIET, Samalkha
# ==========================================

# Load model
model = joblib.load("obesity_knn_model.pkl")

# =======================
# Encoding Dictionaries
# =======================

map_alc = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

map_calc = {
    "no": 0,
    "yes": 1
}

map_mtrans = {
    "Public_Transportation": 0,
    "Walking": 1,
    "Automobile": 2,
    "Motorbike": 3,
    "Bike": 4
}

map_caec = {
    "no": 0,
    "Sometimes": 1,
    "Frequently": 2,
    "Always": 3
}

map_gender = {
    "Female": 0,
    "Male": 1
}

map_smoke = {
    "no": 0,
    "yes": 1
}

map_scc = {
    "no": 0,
    "yes": 1
}

map_family = {
    "no": 0,
    "yes": 1
}

prediction_map = {
    0: "Normal Weight",
    1: "Overweight Level I",
    2: "Overweight Level II",
    3: "Obesity Type I",
    4: "Obesity Type II",
    5: "Obesity Type III",
    6: "Insufficient Weight"
}


# =======================
# Prediction Function
# =======================

def predict(
    gender,
    age,
    height,
    weight,
    family,
    favc,
    fcvc,
    ncp,
    caec,
    smoke,
    ch2o,
    scc,
    faf,
    tue,
    calc,
    mtrans,
):

    data = pd.DataFrame({
        "Age": [float(age)],
        "Gender": [map_gender[gender]],
        "Height": [float(height)],
        "Weight": [float(weight)],
        "CALC": [map_alc[calc]],
        "FAVC": [map_calc[favc]],
        "FCVC": [float(fcvc)],
        "NCP": [float(ncp)],
        "SCC": [map_scc[scc]],
        "SMOKE": [map_smoke[smoke]],
        "CH2O": [float(ch2o)],
        "family_history_with_overweight": [map_family[family]],
        "FAF": [float(faf)],
        "TUE": [float(tue)],
        "CAEC": [map_caec[caec]],
        "MTRANS": [map_mtrans[mtrans]]
    })

    pred = model.predict(data)[0]

    if isinstance(pred, str):
        return pred.replace("_", " ")

    return prediction_map.get(pred, str(pred))


# =======================
# Interface
# =======================

with gr.Blocks(theme=gr.themes.Soft(), title="Obesity Prediction System") as demo:

    gr.Markdown("""
# 🏥 Obesity Level Prediction System

### Machine Learning Based Health Prediction

**Developed By:** Parth

**Roll No:** 241504

**College:** PIET, Samalkha
""")

    with gr.Row():
        gender = gr.Radio(["Male", "Female"], label="Gender")
        age = gr.Number(label="Age")

    with gr.Row():
        height = gr.Number(label="Height (meters)")
        weight = gr.Number(label="Weight (kg)")

    with gr.Row():
        family = gr.Radio(["yes", "no"], label="Family History")
        favc = gr.Radio(["yes", "no"], label="Frequent High Calorie Food")

    with gr.Row():
        fcvc = gr.Slider(1, 3, step=0.1, label="Vegetable Consumption (FCVC)")
        ncp = gr.Slider(1, 4, step=0.5, label="Meals Per Day (NCP)")

    with gr.Row():
        caec = gr.Dropdown(
            ["no", "Sometimes", "Frequently", "Always"],
            label="Food Between Meals"
        )

        smoke = gr.Radio(["yes", "no"], label="Smoking")

    with gr.Row():
        ch2o = gr.Slider(1, 3, step=0.1, label="Water Intake")
        scc = gr.Radio(["yes", "no"], label="Calories Monitoring")

    with gr.Row():
        faf = gr.Slider(0, 3, step=0.1, label="Physical Activity")
        tue = gr.Slider(0, 2, step=0.1, label="Technology Usage")

    with gr.Row():
        calc = gr.Dropdown(
            ["no", "Sometimes", "Frequently", "Always"],
            label="Alcohol Consumption"
        )

        mtrans = gr.Dropdown(
            [
                "Public_Transportation",
                "Walking",
                "Automobile",
                "Motorbike",
                "Bike"
            ],
            label="Transportation"
        )

    output = gr.Textbox(label="Prediction")

    gr.Button(
        "Predict Obesity Level",
        variant="primary"
    ).click(
        fn=predict,
        inputs=[
            gender,
            age,
            height,
            weight,
            family,
            favc,
            fcvc,
            ncp,
            caec,
            smoke,
            ch2o,
            scc,
            faf,
            tue,
            calc,
            mtrans
        ],
        outputs=output
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("PORT", 7860))
    )
