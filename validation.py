

from __future__ import annotations

import re
from datetime import date, datetime
from typing import Tuple, List, Union

# ---------------------------------------------------------------------------
# Validation constants
# ---------------------------------------------------------------------------

# RFC 5322 "good enough" email pattern for practical form validation.
EMAIL_PATTERN = re.compile(
    r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
)

# Clinically reasonable bounds used for *form-level sanity checking only*.
# These are NOT diagnostic thresholds; they exist purely to catch typos
# and obviously invalid data entry (e.g. a glucose value of 5000).
GLUCOSE_MIN, GLUCOSE_MAX = 50.0, 500.0          # mg/dL
HAEMOGLOBIN_MIN, HAEMOGLOBIN_MAX = 5.0, 20.0     # g/dL
CHOLESTEROL_MIN, CHOLESTEROL_MAX = 100.0, 400.0  # mg/dL

MIN_NAME_LENGTH = 2
MAX_NAME_LENGTH = 100


# ---------------------------------------------------------------------------
# Individual field validators
# ---------------------------------------------------------------------------

def validate_full_name(full_name: str) -> Tuple[bool, str]:
    """
    Validate that a full name is present and reasonable in length.

    Args:
        full_name: The patient's full name as entered in the form.

    Returns:
        (is_valid, error_message)
    """
    if full_name is None or not full_name.strip():
        return False, "Full name is required."

    cleaned = full_name.strip()

    if len(cleaned) < MIN_NAME_LENGTH:
        return False, f"Full name must be at least {MIN_NAME_LENGTH} characters long."

    if len(cleaned) > MAX_NAME_LENGTH:
        return False, f"Full name must not exceed {MAX_NAME_LENGTH} characters."

    # Allow letters, spaces, hyphens, and apostrophes (e.g. "Mary-Jane O'Brien").
    if not re.match(r"^[A-Za-z\s'\-.]+$", cleaned):
        return False, "Full name must contain only letters, spaces, hyphens, and apostrophes."

    return True, ""


def validate_email(email: str) -> Tuple[bool, str]:
    """
    Validate an email address against a practical regex pattern.

    Args:
        email: The email address string to validate.

    Returns:
        (is_valid, error_message)
    """
    if email is None or not email.strip():
        return False, "Email address is required."

    cleaned = email.strip()

    if len(cleaned) > 254:
        return False, "Email address is too long."

    if not EMAIL_PATTERN.match(cleaned):
        return False, "Please enter a valid email address (e.g. name@example.com)."

    return True, ""


def validate_dob(dob: Union[date, datetime, str]) -> Tuple[bool, str]:
    """
    Validate a date of birth. The DOB cannot be in the future and must
    represent a realistic human lifespan (0-130 years).

    Args:
        dob: A date/datetime object, or an ISO-format date string (YYYY-MM-DD).

    Returns:
        (is_valid, error_message)
    """
    if dob is None or dob == "":
        return False, "Date of birth is required."

    parsed_dob: date

    if isinstance(dob, datetime):
        parsed_dob = dob.date()
    elif isinstance(dob, date):
        parsed_dob = dob
    elif isinstance(dob, str):
        try:
            parsed_dob = datetime.strptime(dob.strip(), "%Y-%m-%d").date()
        except ValueError:
            return False, "Date of birth must be in YYYY-MM-DD format."
    else:
        return False, "Date of birth has an invalid type."

    today = date.today()

    if parsed_dob > today:
        return False, "Date of birth cannot be in the future."

    age_years = (today - parsed_dob).days / 365.25
    if age_years > 130:
        return False, "Date of birth indicates an unrealistic age (over 130 years)."

    return True, ""


def validate_numeric_range(
    value: Union[int, float, str],
    field_name: str,
    min_value: float,
    max_value: float,
) -> Tuple[bool, str]:
    """
    Generic numeric range validator used for all blood test fields.

    Args:
        value: The raw value to validate (may arrive as a string from a form).
        field_name: Human-readable field name used in error messages.
        min_value: Inclusive lower bound of the acceptable range.
        max_value: Inclusive upper bound of the acceptable range.

    Returns:
        (is_valid, error_message)
    """
    if value is None or value == "":
        return False, f"{field_name} is required."

    try:
        numeric_value = float(value)
    except (TypeError, ValueError):
        return False, f"{field_name} must be a numeric value."

    if numeric_value != numeric_value:  # NaN check without importing math
        return False, f"{field_name} must be a valid number."

    if numeric_value < min_value or numeric_value > max_value:
        return False, (
            f"{field_name} must be between {min_value:g} and {max_value:g}. "
            f"Received: {numeric_value:g}."
        )

    return True, ""


def validate_glucose(value: Union[int, float, str]) -> Tuple[bool, str]:
    """Validate glucose level (mg/dL)."""
    return validate_numeric_range(value, "Glucose", GLUCOSE_MIN, GLUCOSE_MAX)


def validate_haemoglobin(value: Union[int, float, str]) -> Tuple[bool, str]:
    """Validate haemoglobin level (g/dL)."""
    return validate_numeric_range(value, "Haemoglobin", HAEMOGLOBIN_MIN, HAEMOGLOBIN_MAX)


def validate_cholesterol(value: Union[int, float, str]) -> Tuple[bool, str]:
    """Validate cholesterol level (mg/dL)."""
    return validate_numeric_range(value, "Cholesterol", CHOLESTEROL_MIN, CHOLESTEROL_MAX)


# ---------------------------------------------------------------------------
# Aggregate validator for the entire patient form
# ---------------------------------------------------------------------------

def validate_patient_form(
    full_name: str,
    dob: Union[date, datetime, str],
    email: str,
    glucose: Union[int, float, str],
    haemoglobin: Union[int, float, str],
    cholesterol: Union[int, float, str],
) -> Tuple[bool, List[str]]:
    """
    Run all field validators against a complete patient form submission.

    Args:
        full_name: Patient's full name.
        dob: Patient's date of birth.
        email: Patient's email address.
        glucose: Glucose blood test value.
        haemoglobin: Haemoglobin blood test value.
        cholesterol: Cholesterol blood test value.

    Returns:
        A tuple of (is_valid, list_of_error_messages). is_valid is True
        only if every field passes validation. The error list is empty
        when is_valid is True.
    """
    validators = [
        validate_full_name(full_name),
        validate_dob(dob),
        validate_email(email),
        validate_glucose(glucose),
        validate_haemoglobin(haemoglobin),
        validate_cholesterol(cholesterol),
    ]

    errors = [message for is_valid, message in validators if not is_valid]
    return (len(errors) == 0), errors
