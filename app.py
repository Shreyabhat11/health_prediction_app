import streamlit as st
import pandas as pd
import plotly.express as px

from database import *
from validation import *
from predictors import *
from ai_services import *

create_table()

st.set_page_config(
    page_title="Health Prediction System",
    layout="wide"
)

page = st.sidebar.selectbox(
    "Navigation",
    [
        "Dashboard",
        "Add Patient",
        "Manage Patients"
    ]
)

if page == "Dashboard":

    st.title(
        "Health Prediction Dashboard"
    )

    records = pd.DataFrame(
        get_all_patients(),
        columns=[
            "id",
            "full_name",
            "dob",
            "email",
            "glucose",
            "haemoglobin",
            "cholesterol",
            "diabetes_risk",
            "anemia_risk",
            "heart_risk",
            "remarks"
        ]
    )

    col1,col2,col3,col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Patients",
            len(records)
        )

    with col2:
        st.metric(
            "Diabetes Risks",
            records["diabetes_risk"].sum()
        )

    with col3:
        st.metric(
            "Anemia Risks",
            records["anemia_risk"].sum()
        )

    with col4:
        st.metric(
            "Heart Risks",
            records["heart_risk"].sum()
        )
    st.divider()

    if len(records) > 0:
        risk_counts = pd.DataFrame({
            "Risk":[
                "Diabetes",
                "Anemia",
                "Heart Disease"
            ],
            "Count":[
                records["diabetes_risk"].sum(),
                records["anemia_risk"].sum(),
                records["heart_risk"].sum()
            ]
        })

        fig = px.bar(
            risk_counts,
            x="Risk",
            y="Count",
            title="Health Risk Distribution"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:
        st.info(
            "No patient records available."
        )

if page == "Add Patient":
    st.title("Add Patient")
    name = st.text_input("Full Name")
    dob = st.date_input("Date of Birth")
    email = st.text_input("Email")
    glucose = st.number_input("Glucose")
    haemoglobin = st.number_input("Haemoglobin")
    cholesterol = st.number_input("Cholesterol")

    if st.button("Predict & Save"):

        if not validate_email(email):
            st.error("Invalid Email")
            st.stop()

        elif not validate_dob(dob):
            st.error("Invalid DOB")
            st.stop()
        
        elif not validate_ranges(glucose,haemoglobin,cholesterol):
            st.error(
                "Please enter realistic blood values."
            )

        else:
            prediction = predict_risks(
                glucose,
                haemoglobin,
                cholesterol
            )

            remarks = generate_remarks(
                name,
                glucose,
                haemoglobin,
                cholesterol,
                prediction["risks"]
            )

            add_patient(
                full_name=name,
                dob=str(dob),
                email=email,
                glucose=glucose,
                haemoglobin=haemoglobin,
                cholesterol=cholesterol,
                diabetes_risk=prediction["diabetes_risk"],
                anemia_risk=prediction["anemia_risk"],
                heart_risk=prediction["heart_risk"],
                remarks=remarks
            )

            st.success(
                "Patient Saved"
            )

            st.text_area(
                "AI Remarks",
                remarks,
                height=150
            )

if page == "Manage Patients":

    st.title("Manage Patients")

    records = get_all_patients()

    if len(records) == 0:
        st.warning("No patient records found.")
        st.stop()

    st.dataframe(
        records,
        use_container_width=True
    )

    st.divider()

    selected_id = st.selectbox(
        "Select Patient",
        records["id"]
    )

    patient = records[
        records["id"] == selected_id
    ].iloc[0]

    st.subheader(
        "Edit Patient"
    )

    new_name = st.text_input(
        "Name",
        value=patient["full_name"]
    )

    new_email = st.text_input(
        "Email",
        value=patient["email"]
    )

    new_dob = st.date_input(
        "Date of Birth",
        value=pd.to_datetime(
            patient["dob"]
        )
    )

    new_glucose = st.number_input(
        "Glucose",
        value=float(
            patient["glucose"]
        )
    )

    new_haemoglobin = st.number_input(
        "Haemoglobin",
        value=float(
            patient["haemoglobin"]
        )
    )

    new_cholesterol = st.number_input(
        "Cholesterol",
        value=float(
            patient["cholesterol"]
        )
    )

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Update Patient"):
            prediction = predict_risks(
                new_glucose,
                new_haemoglobin,
                new_cholesterol
            )

            remarks = generate_remarks(
                new_name,
                new_glucose,
                new_haemoglobin,
                new_cholesterol,
                prediction["risks"]
            )

            update_patient(
                selected_id,
                new_name,
                str(new_dob),
                new_email,
                new_glucose,
                new_haemoglobin,
                new_cholesterol,
                prediction["diabetes_risk"],
                prediction["anemia_risk"],
                prediction["heart_risk"],
                remarks
            )

            st.success(
                "Patient updated successfully."
            )

            st.rerun()

    with col2:
        if st.button("Delete Patient"):
            delete_patient(selected_id)
            st.success(
                "Patient deleted."
            )
            st.rerun()
    st.divider()

    st.subheader(
        "Latest AI Assessment"
    )

    st.text_area(
        "Remarks",
        patient["remarks"],
        height=150
    )
