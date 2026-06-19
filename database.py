from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from typing import Iterator, List, Optional

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Database file is stored alongside this module so the path is stable
# regardless of the working directory the app is launched from.
_BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_BASE_DIR, "data", "health_records.db")

TABLE_NAME = "patients"


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass
class Patient:
    """Typed representation of a single patient record."""

    id: Optional[int]
    full_name: str
    dob: str            # Stored as ISO format string: YYYY-MM-DD
    email: str
    glucose: float
    haemoglobin: float
    cholesterol: float
    remarks: str
    prediction: str = ""  # Raw ML class label (e.g. "Healthy")

    @staticmethod
    def from_row(row: sqlite3.Row) -> "Patient":
        """Build a Patient instance from a sqlite3.Row result."""
        return Patient(
            id=row["id"],
            full_name=row["full_name"],
            dob=row["dob"],
            email=row["email"],
            glucose=row["glucose"],
            haemoglobin=row["haemoglobin"],
            cholesterol=row["cholesterol"],
            remarks=row["remarks"],
            prediction=row["prediction"] if "prediction" in row.keys() else "",
        )


# ---------------------------------------------------------------------------
# Connection management
# ---------------------------------------------------------------------------

@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    """
    Context manager that yields a SQLite connection with row factory set
    so columns can be accessed by name. Commits on success, rolls back
    on error, and always closes the connection.
    """
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """
    Create the patients table if it does not already exist.
    Safe to call on every application startup.
    """
    with get_connection() as conn:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                full_name TEXT NOT NULL,
                dob TEXT NOT NULL,
                email TEXT NOT NULL,
                glucose REAL NOT NULL,
                haemoglobin REAL NOT NULL,
                cholesterol REAL NOT NULL,
                remarks TEXT NOT NULL,
                prediction TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at TEXT NOT NULL DEFAULT (datetime('now'))
            )
            """
        )


# ---------------------------------------------------------------------------
# CRUD operations
# ---------------------------------------------------------------------------

def create_patient(
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str,
    prediction: str,
) -> int:
    """
    Insert a new patient record.

    Returns:
        The auto-generated integer ID of the newly created record.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            f"""
            INSERT INTO {TABLE_NAME}
                (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, prediction)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (full_name, dob, email, glucose, haemoglobin, cholesterol, remarks, prediction),
        )
        return int(cursor.lastrowid)


def get_all_patients() -> List[Patient]:
    """
    Retrieve every patient record, most recently created first.

    Returns:
        A list of Patient dataclass instances.
    """
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM {TABLE_NAME} ORDER BY id DESC"
        ).fetchall()
        return [Patient.from_row(row) for row in rows]


def get_patient_by_id(patient_id: int) -> Optional[Patient]:
    """
    Retrieve a single patient by primary key.

    Args:
        patient_id: The patient's database ID.

    Returns:
        A Patient instance, or None if no matching record exists.
    """
    with get_connection() as conn:
        row = conn.execute(
            f"SELECT * FROM {TABLE_NAME} WHERE id = ?", (patient_id,)
        ).fetchone()
        return Patient.from_row(row) if row else None


def update_patient(
    patient_id: int,
    full_name: str,
    dob: str,
    email: str,
    glucose: float,
    haemoglobin: float,
    cholesterol: float,
    remarks: str,
    prediction: str,
) -> bool:
    """
    Update an existing patient record by ID.

    Returns:
        True if a row was updated, False if no record matched the ID.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            f"""
            UPDATE {TABLE_NAME}
            SET full_name = ?,
                dob = ?,
                email = ?,
                glucose = ?,
                haemoglobin = ?,
                cholesterol = ?,
                remarks = ?,
                prediction = ?,
                updated_at = datetime('now')
            WHERE id = ?
            """,
            (full_name, dob, email, glucose, haemoglobin, cholesterol,
             remarks, prediction, patient_id),
        )
        return cursor.rowcount > 0


def delete_patient(patient_id: int) -> bool:
    """
    Delete a patient record by ID.

    Returns:
        True if a row was deleted, False if no record matched the ID.
    """
    with get_connection() as conn:
        cursor = conn.execute(
            f"DELETE FROM {TABLE_NAME} WHERE id = ?", (patient_id,)
        )
        return cursor.rowcount > 0


def get_prediction_counts() -> dict:
    """
    Aggregate patient counts grouped by prediction class, for use in
    dashboard metric cards and charts.

    Returns:
        A dictionary mapping prediction label -> count, plus a 'Total' key.
    """
    with get_connection() as conn:
        rows = conn.execute(
            f"""
            SELECT prediction, COUNT(*) as count
            FROM {TABLE_NAME}
            GROUP BY prediction
            """
        ).fetchall()

    counts = {row["prediction"]: row["count"] for row in rows}
    counts["Total"] = sum(counts.values())
    return counts
