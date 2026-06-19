import sqlite3

DB_PATH = "database/patients.db"


def get_connection():
    return sqlite3.connect(DB_PATH)


def create_table():

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   full_name TEXT NOT NULL,
                   dob TEXT NOT NULL,
                   email TEXT NOT NULL,
                   glucose REAL,
                   haemoglobin REAL,
                   cholesterol REAL,
                   diabetes_risk INTEGER,
                   anemia_risk INTEGER,
                   heart_risk INTEGER,
                   remarks TEXT
                   )
                   """)
    

    conn.commit()
    conn.close()


def add_patient(
    full_name,
    dob,
    email,
    glucose,
    haemoglobin,
    cholesterol,
    remarks
):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    INSERT INTO patients
    (
        full_name,
        dob,
        email,
        glucose,
        haemoglobin,
        cholesterol,
        remarks
    )
    VALUES (?,?,?,?,?,?,?)
    """,
    (
        full_name,
        dob,
        email,
        glucose,
        haemoglobin,
        cholesterol,
        remarks
    ))

    conn.commit()
    conn.close()


def get_all_patients():

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM patients")
    data = cursor.fetchall()
    conn.close()

    return data


def update_patient(patient_id,full_name,dob,email,glucose,haemoglobin,cholesterol,remarks):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute("""
    UPDATE patients
    SET
        full_name=?,
        dob=?,
        email=?,
        glucose=?,
        haemoglobin=?,
        cholesterol=?,
        remarks=?
    WHERE id=?
    """,
    (
        full_name,
        dob,
        email,
        glucose,
        haemoglobin,
        cholesterol,
        remarks,
        patient_id
    ))

    conn.commit()
    conn.close()


def delete_patient(patient_id):

    conn = get_connection()

    cursor = conn.cursor()

    cursor.execute(
        "DELETE FROM patients WHERE id=?",
        (patient_id,)
    )

    conn.commit()
    conn.close()