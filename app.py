"""
app.py
------
Main Streamlit application for the Health Prediction Application.

Provides a professional healthcare dashboard with three pages:
    1. Dashboard         - summary metrics and prediction distribution chart
    2. Add Patient       - form to create a new patient record with
                            automatic AI/ML health risk prediction
    3. Manage Patients   - interactive table to view, edit, and delete
                            existing patient records

Run with:
    streamlit run app.py
"""

from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import plotly.express as px
import streamlit as st

import database as db
from model import predict_with_probabilities
from validation import (
    validate_patient_form,
    validate_full_name,
    validate_email,
    validate_dob,
    validate_glucose,
    validate_haemoglobin,
    validate_cholesterol,
    GLUCOSE_MIN, GLUCOSE_MAX,
    HAEMOGLOBIN_MIN, HAEMOGLOBIN_MAX,
    CHOLESTEROL_MIN, CHOLESTEROL_MAX,
)

# ---------------------------------------------------------------------------
# Page configuration & global styling
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Health Prediction Dashboard",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #0f4c5c;
        margin-bottom: 0.1rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #5a6c72;
        margin-bottom: 1.5rem;
    }
    .risk-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .badge-healthy { background-color: #d4f4dd; color: #1b6e3c; }
    .badge-prediabetes { background-color: #fff3cd; color: #8a6500; }
    .badge-anemia { background-color: #fde0e0; color: #a3303d; }
    .badge-cholesterol { background-color: #ffe3d1; color: #b5500b; }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
    div[data-testid="stMetric"] {
        background-color: #f7f9fa;
        border: 1px solid #e3e8ea;
        border-radius: 12px;
        padding: 0.8rem 1rem;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# Badge styling lookup used in the patient management table and confirmations.
RISK_BADGE_CLASS = {
    "Healthy": "badge-healthy",
    "Prediabetes Risk": "badge-prediabetes",
    "Anemia Risk": "badge-anemia",
    "High Cholesterol Risk": "badge-cholesterol",
}

RISK_ICON = {
    "Healthy": "✅",
    "Prediabetes Risk": "⚠️",
    "Anemia Risk": "🩸",
    "High Cholesterol Risk": "🫀",
}


# ---------------------------------------------------------------------------
# App initialization
# ---------------------------------------------------------------------------

db.init_db()


def render_risk_badge(prediction: str) -> str:
    """Build an HTML badge span for a given prediction label."""
    css_class = RISK_BADGE_CLASS.get(prediction, "badge-healthy")
    icon = RISK_ICON.get(prediction, "•")
    return f'<span class="risk-badge {css_class}">{icon} {prediction}</span>'


# ---------------------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown("## 🩺 Health Predictor")
    st.caption("AI-powered patient risk assessment")
    st.divider()

    page = st.radio(
        "Navigate",
        options=["📊 Dashboard", "➕ Add Patient", "🗂️ Manage Patients"],
        label_visibility="collapsed",
    )

    st.divider()
    st.caption("Built with Streamlit · scikit-learn · SQLite")


# ---------------------------------------------------------------------------
# Page: Dashboard
# ---------------------------------------------------------------------------

def render_dashboard() -> None:
    """Render the dashboard page with summary metrics and a Plotly chart."""
    st.markdown('<div class="main-header">📊 Health Prediction Dashboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Overview of patient records and AI-generated risk predictions</div>',
        unsafe_allow_html=True,
    )

    counts = db.get_prediction_counts()
    total = counts.get("Total", 0)
    healthy = counts.get("Healthy", 0)
    prediabetes = counts.get("Prediabetes Risk", 0)
    anemia = counts.get("Anemia Risk", 0)
    high_chol = counts.get("High Cholesterol Risk", 0)

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Patients", total)
    col2.metric("✅ Healthy", healthy)
    col3.metric("⚠️ Prediabetes", prediabetes)
    col4.metric("🩸 Anemia Risk", anemia)
    col5.metric("🫀 High Cholesterol", high_chol)

    st.divider()

    if total == 0:
        st.info("No patient records yet. Add a patient to see the prediction distribution chart.")
        return

    chart_col, table_col = st.columns([3, 2])

    with chart_col:
        st.subheader("Prediction Distribution")
        chart_data = pd.DataFrame({
            "Risk Category": ["Healthy", "Prediabetes Risk", "Anemia Risk", "High Cholesterol Risk"],
            "Count": [healthy, prediabetes, anemia, high_chol],
        })
        chart_data = chart_data[chart_data["Count"] > 0]

        color_map = {
            "Healthy": "#2e9e5b",
            "Prediabetes Risk": "#d8a400",
            "Anemia Risk": "#c43d4f",
            "High Cholesterol Risk": "#d97a2b",
        }

        fig = px.pie(
            chart_data,
            names="Risk Category",
            values="Count",
            color="Risk Category",
            color_discrete_map=color_map,
            hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=True, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with table_col:
        st.subheader("Recent Patients")
        recent = db.get_all_patients()[:5]
        for patient in recent:
            st.markdown(
                f"**{patient.full_name}** — {render_risk_badge(patient.prediction)}",
                unsafe_allow_html=True,
            )
        if not recent:
            st.caption("No records to display.")


# ---------------------------------------------------------------------------
# Page: Add Patient
# ---------------------------------------------------------------------------

def render_add_patient() -> None:
    """Render the Add Patient page with a validated form and live prediction."""
    st.markdown('<div class="main-header">➕ Add New Patient</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Enter patient details and blood test results. '
        'The health risk prediction is generated automatically.</div>',
        unsafe_allow_html=True,
    )

    with st.form("add_patient_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            full_name = st.text_input("Full Name *", placeholder="e.g. Asha Patel")
            dob_input = st.date_input(
                "Date of Birth *",
                value=date(1990, 1, 1),
                min_value=date(1900, 1, 1),
                max_value=date.today(),
            )
            email = st.text_input("Email Address *", placeholder="e.g. asha.patel@example.com")

        with col2:
            glucose = st.number_input(
                f"Glucose (mg/dL) — range {GLUCOSE_MIN:g}-{GLUCOSE_MAX:g} *",
                min_value=0.0, max_value=1000.0, value=90.0, step=0.5,
            )
            haemoglobin = st.number_input(
                f"Haemoglobin (g/dL) — range {HAEMOGLOBIN_MIN:g}-{HAEMOGLOBIN_MAX:g} *",
                min_value=0.0, max_value=50.0, value=14.0, step=0.1,
            )
            cholesterol = st.number_input(
                f"Cholesterol (mg/dL) — range {CHOLESTEROL_MIN:g}-{CHOLESTEROL_MAX:g} *",
                min_value=0.0, max_value=1000.0, value=180.0, step=0.5,
            )

        st.caption("* Required fields")
        submitted = st.form_submit_button("🔍 Predict & Save Patient", use_container_width=True)

    if submitted:
        is_valid, errors = validate_patient_form(
            full_name, dob_input, email, glucose, haemoglobin, cholesterol
        )

        if not is_valid:
            for error in errors:
                st.error(f"⚠️ {error}")
            return

        prediction, remarks, probabilities = predict_with_probabilities(
            glucose, haemoglobin, cholesterol
        )

        patient_id = db.create_patient(
            full_name=full_name.strip(),
            dob=dob_input.isoformat(),
            email=email.strip(),
            glucose=glucose,
            haemoglobin=haemoglobin,
            cholesterol=cholesterol,
            remarks=remarks,
            prediction=prediction,
        )

        st.success(f"✅ Patient **{full_name.strip()}** saved successfully (ID #{patient_id}).")

        st.markdown("### Prediction Result")
        st.markdown(render_risk_badge(prediction), unsafe_allow_html=True)
        st.info(remarks)

        with st.expander("View prediction confidence breakdown"):
            proba_df = pd.DataFrame({
                "Risk Category": list(probabilities.keys()),
                "Probability": list(probabilities.values()),
            }).sort_values("Probability", ascending=False)
            proba_df["Probability"] = (proba_df["Probability"] * 100).round(1)
            st.dataframe(
                proba_df.rename(columns={"Probability": "Probability (%)"}),
                hide_index=True,
                use_container_width=True,
            )


# ---------------------------------------------------------------------------
# Page: Manage Patients
# ---------------------------------------------------------------------------

def render_manage_patients() -> None:
    """Render the Manage Patients page: view, search, edit, and delete records."""
    st.markdown('<div class="main-header">🗂️ Manage Patients</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">View, search, update, or delete patient records.</div>',
        unsafe_allow_html=True,
    )

    patients = db.get_all_patients()

    if not patients:
        st.info("No patient records found. Add a patient from the **Add Patient** page to get started.")
        return

    search_term = st.text_input("🔎 Search by name or email", placeholder="Start typing to filter...")

    filtered = patients
    if search_term.strip():
        term = search_term.strip().lower()
        filtered = [
            p for p in patients
            if term in p.full_name.lower() or term in p.email.lower()
        ]

    st.caption(f"Showing {len(filtered)} of {len(patients)} record(s).")

    table_data = pd.DataFrame([
        {
            "ID": p.id,
            "Full Name": p.full_name,
            "DOB": p.dob,
            "Email": p.email,
            "Glucose": p.glucose,
            "Haemoglobin": p.haemoglobin,
            "Cholesterol": p.cholesterol,
            "Prediction": p.prediction,
        }
        for p in filtered
    ])

    st.dataframe(table_data, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Edit or Delete a Patient")

    patient_options = {f"#{p.id} — {p.full_name}": p.id for p in filtered}
    if not patient_options:
        st.warning("No matching records to edit.")
        return

    selected_label = st.selectbox("Select a patient", options=list(patient_options.keys()))
    selected_id = patient_options[selected_label]
    patient = db.get_patient_by_id(selected_id)

    if patient is None:
        st.error("Selected patient could not be found. It may have just been deleted.")
        return

    edit_tab, delete_tab = st.tabs(["✏️ Edit Patient", "🗑️ Delete Patient"])

    # --- Edit tab -----------------------------------------------------
    with edit_tab:
        with st.form(f"edit_form_{selected_id}"):
            col1, col2 = st.columns(2)

            with col1:
                edit_name = st.text_input("Full Name *", value=patient.full_name)
                edit_dob = st.date_input(
                    "Date of Birth *",
                    value=datetime.strptime(patient.dob, "%Y-%m-%d").date(),
                    min_value=date(1900, 1, 1),
                    max_value=date.today(),
                )
                edit_email = st.text_input("Email Address *", value=patient.email)

            with col2:
                edit_glucose = st.number_input(
                    "Glucose (mg/dL) *", min_value=0.0, max_value=1000.0,
                    value=float(patient.glucose), step=0.5,
                )
                edit_haemoglobin = st.number_input(
                    "Haemoglobin (g/dL) *", min_value=0.0, max_value=50.0,
                    value=float(patient.haemoglobin), step=0.1,
                )
                edit_cholesterol = st.number_input(
                    "Cholesterol (mg/dL) *", min_value=0.0, max_value=1000.0,
                    value=float(patient.cholesterol), step=0.5,
                )

            st.caption("Saving will automatically re-run the AI prediction with the updated values.")
            save_changes = st.form_submit_button("💾 Save Changes & Re-Predict", use_container_width=True)

        if save_changes:
            is_valid, errors = validate_patient_form(
                edit_name, edit_dob, edit_email, edit_glucose, edit_haemoglobin, edit_cholesterol
            )

            if not is_valid:
                for error in errors:
                    st.error(f"⚠️ {error}")
            else:
                new_prediction, new_remarks, _ = predict_with_probabilities(
                    edit_glucose, edit_haemoglobin, edit_cholesterol
                )

                db.update_patient(
                    patient_id=selected_id,
                    full_name=edit_name.strip(),
                    dob=edit_dob.isoformat(),
                    email=edit_email.strip(),
                    glucose=edit_glucose,
                    haemoglobin=edit_haemoglobin,
                    cholesterol=edit_cholesterol,
                    remarks=new_remarks,
                    prediction=new_prediction,
                )

                st.success(f"✅ Patient **{edit_name.strip()}** updated successfully.")
                st.markdown(render_risk_badge(new_prediction), unsafe_allow_html=True)
                st.info(new_remarks)
                st.rerun()

    # --- Delete tab -----------------------------------------------------
    with delete_tab:
        st.warning(
            f"You are about to permanently delete the record for "
            f"**{patient.full_name}** (ID #{patient.id}). This action cannot be undone."
        )
        confirm = st.checkbox("I understand and want to delete this record.", key=f"confirm_{selected_id}")
        if st.button("🗑️ Permanently Delete", disabled=not confirm, use_container_width=True):
            db.delete_patient(selected_id)
            st.success(f"Patient **{patient.full_name}** was deleted.")
            st.rerun()


# ---------------------------------------------------------------------------
# Page router
# ---------------------------------------------------------------------------

if page == "📊 Dashboard":
    render_dashboard()
elif page == "➕ Add Patient":
    render_add_patient()
elif page == "🗂️ Manage Patients":
    render_manage_patients()
