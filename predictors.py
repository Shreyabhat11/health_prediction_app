import joblib
import numpy as np

diabetes_model = joblib.load(
    "models/diabetes_model.pkl"
)

anemia_model = joblib.load(
    "models/anemia_model.pkl"
)

heart_model = joblib.load(
    "models/heart_disease_model.pkl"
)


def predict_risks(
    glucose,
    haemoglobin,
    cholesterol
):

    diabetes_pred = diabetes_model.predict(
        np.array([[glucose]])
    )[0]

    anemia_pred = anemia_model.predict(
        np.array([[haemoglobin]])
    )[0]

    heart_pred = heart_model.predict(
        np.array([[cholesterol]])
    )[0]

    risks = []

    if diabetes_pred == 1:
        risks.append(
            "Possible Diabetes Risk"
        )

    if anemia_pred == 1:
        risks.append(
            "Possible Anemia Risk"
        )

    if heart_pred == 1:
        risks.append(
            "Possible Cardiovascular Risk"
        )

    if len(risks) == 0:

        risks.append(
            "No significant health risk detected"
        )

    return {
        "diabetes_risk": int(diabetes_pred),
        "anemia_risk": int(anemia_pred),
        "heart_risk": int(heart_pred),
        "risks": risks
    }