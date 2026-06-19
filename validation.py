import re
from datetime import date


def validate_email(email):

    pattern = r'^[\w\.-]+@[\w\.-]+\.\w+$'

    return bool(
        re.match(pattern,email)
    )


def validate_dob(dob):

    return dob <= date.today()


def validate_numeric(value):

    try:
        float(value)
        return True

    except:
        return False


def validate_ranges(
    glucose,
    haemoglobin,
    cholesterol
):

    if glucose < 50 or glucose > 500:
        return False

    if haemoglobin < 5 or haemoglobin > 20:
        return False

    if cholesterol < 100 or cholesterol > 400:
        return False

    return True