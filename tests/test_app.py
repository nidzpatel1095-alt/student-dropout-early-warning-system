# ==============================================================================
# TEST SUITE: tests/test_app.py
# Automated tests for model loading, prediction logic, auth access control,
# input validation, and verification that the real database remains untouched.
# ==============================================================================

import os
import sys
import pytest

from config import MODEL_PATH
from ml.predict_utils import load_trained_model, predict_risk
from tests.conftest import REAL_DB_PATH, REAL_DB_EXISTS, REAL_DB_MTIME, REAL_DB_SIZE


def test_model_file_loads():
    """Test 1: Verify that trained model file exists and loads successfully."""
    assert os.path.exists(MODEL_PATH), f"Model file missing at '{MODEL_PATH}'"
    model = load_trained_model()
    assert model is not None, "Failed to load trained model instance."


def test_weak_vs_strong_student_probability():
    """
    Test 2: Verify that a weak student profile returns a higher dropout probability
    than a strong student profile.
    """
    model = load_trained_model()
    assert model is not None, "Model must be trained prior to running test."

    # Strong student profile: high grades, tuition paid, scholarship holder, no debt
    strong_student = {
        "Gender": 1,
        "Age at enrollment": 18,
        "Scholarship holder": 1,
        "Tuition fees up to date": 1,
        "Debtor": 0,
        "Displaced": 0,
        "Admission grade": 160.0,
        "Previous qualification (grade)": 160.0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 6,
        "Curricular units 1st sem (approved)": 6,
        "Curricular units 1st sem (grade)": 16.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 6,
        "Curricular units 2nd sem (approved)": 6,
        "Curricular units 2nd sem (grade)": 16.0
    }

    # Weak student profile: 0 approved units, unpaid tuition, debtor, low grades
    weak_student = {
        "Gender": 1,
        "Age at enrollment": 24,
        "Scholarship holder": 0,
        "Tuition fees up to date": 0,
        "Debtor": 1,
        "Displaced": 1,
        "Admission grade": 100.0,
        "Previous qualification (grade)": 100.0,
        "Curricular units 1st sem (enrolled)": 6,
        "Curricular units 1st sem (evaluations)": 8,
        "Curricular units 1st sem (approved)": 0,
        "Curricular units 1st sem (grade)": 0.0,
        "Curricular units 2nd sem (enrolled)": 6,
        "Curricular units 2nd sem (evaluations)": 8,
        "Curricular units 2nd sem (approved)": 0,
        "Curricular units 2nd sem (grade)": 0.0
    }

    prob_strong, risk_strong, _ = predict_risk(strong_student, model=model)
    prob_weak, risk_weak, _ = predict_risk(weak_student, model=model)

    assert prob_weak > prob_strong, (
        f"Weak student probability ({prob_weak:.4f}) must be higher than "
        f"strong student probability ({prob_strong:.4f})."
    )


def test_predict_page_access_and_auth(client):
    """
    Test 3: Verify that /predict redirects when logged out (302/303) and
    returns 200 OK when authenticated.
    """
    # Logged-out access should redirect to /login
    res_unauth = client.get("/predict")
    assert res_unauth.status_code in [302, 303], "Unauthenticated request should redirect."

    # Perform Admin Login
    client.post("/login", data={"username": "admin", "password": "admin123"})

    # Logged-in access should succeed with 200 OK
    res_auth = client.get("/predict")
    assert res_auth.status_code == 200, "Authenticated request should return 200 OK."


def test_invalid_input_rejected(client):
    """
    Test 4: Verify that out-of-range numerical input is rejected with an error message.
    """
    # Perform Admin Login
    client.post("/login", data={"username": "admin", "password": "admin123"})

    # Post invalid form payload (Age = 150, which exceeds valid range 15-80)
    invalid_payload = {
        "Gender": "1",
        "Age at enrollment": "150",
        "Scholarship holder": "0",
        "Tuition fees up to date": "1",
        "Debtor": "0",
        "Displaced": "0",
        "Admission grade": "130.0",
        "Previous qualification (grade)": "130.0",
        "Curricular units 1st sem (enrolled)": "6",
        "Curricular units 1st sem (evaluations)": "6",
        "Curricular units 1st sem (approved)": "5",
        "Curricular units 1st sem (grade)": "12.0",
        "Curricular units 2nd sem (enrolled)": "6",
        "Curricular units 2nd sem (evaluations)": "6",
        "Curricular units 2nd sem (approved)": "5",
        "Curricular units 2nd sem (grade)": "12.0"
    }

    res = client.post("/predict", data=invalid_payload)
    assert res.status_code == 200
    assert b"Input Validation Error" in res.data or b"must be between" in res.data, (
        "Response should contain validation error message."
    )


def test_real_database_untouched(client):
    """
    Test 5: Verify that the real production database file (database/dropout_system.db)
    is untouched after running HTTP requests against the test client.
    """
    # Execute several web requests against test client
    client.post("/login", data={"username": "admin", "password": "admin123"})
    client.get("/dashboard")
    client.get("/students")
    
    if REAL_DB_EXISTS:
        current_mtime = os.path.getmtime(REAL_DB_PATH)
        current_size = os.path.getsize(REAL_DB_PATH)
        
        assert current_mtime == REAL_DB_MTIME, "Real DB modification time changed!"
        assert current_size == REAL_DB_SIZE, "Real DB file size changed!"
