"""
Student Dropout Early Warning System
Flask web application with authentication, real-time risk prediction,
dashboard analytics, and student record management.
"""

import os
import sys
import json
import sqlite3
import csv
from functools import wraps
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    session,
    flash,
    send_from_directory
)
from werkzeug.security import check_password_hash

# Import project settings and ML utility functions
from config import (
    BASE_DIR,
    DB_PATH,
    FIGURES_DIR,
    TABLES_DIR,
    METRICS_PATH,
    APP_FEATURES,
    SECRET_KEY,
    RISK_LOW_THRESHOLD,
    RISK_HIGH_THRESHOLD
)
from database.init_db import init_db
from scripts.load_sample_students import load_sample_students
from ml.predict_utils import predict_risk, extract_risk_reasons, get_recommendations, load_trained_model

# Initialize Flask App
app = Flask(__name__)
app.secret_key = SECRET_KEY


def get_current_db_path():
    """Returns DB path, respecting environment variable overrides if present."""
    return os.environ.get("DROPOUT_DB_PATH", DB_PATH)


def get_db_connection():
    """Helper function to create SQLite connection with dictionary-like Row factory."""
    conn = sqlite3.connect(get_current_db_path())
    conn.row_factory = sqlite3.Row
    return conn


# Ensure database exists upon app launch
current_path = get_current_db_path()
if not os.path.exists(current_path):
    init_db()

# Pre-load ML Model into memory
ml_model = load_trained_model()


def login_required(f):
    """Decorator to enforce admin authentication on protected pages."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if "user_id" not in session:
            flash("Please log in to access the system.", "warning")
            return redirect(url_for("login"))
        return f(*args, **kwargs)
    return decorated_function


# ------------------------------------------------------------------------------
# AUTHENTICATION ROUTES
# ------------------------------------------------------------------------------

@app.route("/login", methods=["GET", "POST"])
def login():
    """Admin Login Page."""
    if "user_id" in session:
        return redirect(url_for("dashboard"))
        
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "").strip()
        
        conn = get_db_connection()
        user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
        conn.close()
        
        if user and check_password_hash(user["password_hash"], password):
            session["user_id"] = user["id"]
            session["username"] = user["username"]
            flash(f"Welcome back, {user['username']}!", "success")
            return redirect(url_for("dashboard"))
        else:
            flash("Invalid username or password. Please try again.", "danger")
            
    return render_template("login.html")


@app.route("/logout")
def logout():
    """Logs out admin user and clears session."""
    session.clear()
    flash("You have been logged out successfully.", "info")
    return redirect(url_for("login"))


# ------------------------------------------------------------------------------
# DASHBOARD ROUTE
# ------------------------------------------------------------------------------

@app.route("/")
@app.route("/dashboard")
@login_required
def dashboard():
    """Dashboard showing student statistics, risk doughnut chart, and top high-risk list."""
    conn = get_db_connection()
    
    # Get total count
    total_students = conn.execute("SELECT COUNT(*) FROM students").fetchone()[0]
    
    # Get risk counts
    low_count = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Low'").fetchone()[0]
    medium_count = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'Medium'").fetchone()[0]
    high_count = conn.execute("SELECT COUNT(*) FROM students WHERE risk_level = 'High'").fetchone()[0]
    
    # Calculate percentages
    low_pct = round((low_count / total_students * 100), 1) if total_students > 0 else 0.0
    medium_pct = round((medium_count / total_students * 100), 1) if total_students > 0 else 0.0
    high_pct = round((high_count / total_students * 100), 1) if total_students > 0 else 0.0
    
    # Query Top 10 Highest Risk Students
    top_high_risk = conn.execute(
        "SELECT * FROM students ORDER BY probability DESC LIMIT 10"
    ).fetchall()
    
    conn.close()
    
    return render_template(
        "dashboard.html",
        total_students=total_students,
        low_count=low_count,
        medium_count=medium_count,
        high_count=high_count,
        low_pct=low_pct,
        medium_pct=medium_pct,
        high_pct=high_pct,
        top_high_risk=top_high_risk
    )


@app.route("/load-samples")
@login_required
def load_samples_route():
    """Route to populate 30 sample dataset records if database is empty."""
    load_sample_students()
    flash("Successfully loaded 30 sample student records from dataset.", "success")
    return redirect(url_for("dashboard"))


# ------------------------------------------------------------------------------
# PREDICT RISK ROUTE
# ------------------------------------------------------------------------------

@app.route("/predict", methods=["GET", "POST"])
@login_required
def predict():
    """Risk Prediction page with 16 APP_FEATURES form & instant result card."""
    prediction_result = None
    form_data = {}
    
    if request.method == "POST":
        form_data = request.form.to_dict()
        
        # Server-side validation & parsing of the 16 features
        try:
            student_dict = {
                "Gender": int(form_data.get("Gender", 1)),
                "Age at enrollment": float(form_data.get("Age at enrollment", 20)),
                "Scholarship holder": int(form_data.get("Scholarship holder", 0)),
                "Tuition fees up to date": int(form_data.get("Tuition fees up to date", 1)),
                "Debtor": int(form_data.get("Debtor", 0)),
                "Displaced": int(form_data.get("Displaced", 1)),
                "Admission grade": float(form_data.get("Admission grade", 130.0)),
                "Previous qualification (grade)": float(form_data.get("Previous qualification (grade)", 130.0)),
                "Curricular units 1st sem (enrolled)": int(form_data.get("Curricular units 1st sem (enrolled)", 6)),
                "Curricular units 1st sem (evaluations)": int(form_data.get("Curricular units 1st sem (evaluations)", 6)),
                "Curricular units 1st sem (approved)": int(form_data.get("Curricular units 1st sem (approved)", 5)),
                "Curricular units 1st sem (grade)": float(form_data.get("Curricular units 1st sem (grade)", 12.0)),
                "Curricular units 2nd sem (enrolled)": int(form_data.get("Curricular units 2nd sem (enrolled)", 6)),
                "Curricular units 2nd sem (evaluations)": int(form_data.get("Curricular units 2nd sem (evaluations)", 6)),
                "Curricular units 2nd sem (approved)": int(form_data.get("Curricular units 2nd sem (approved)", 5)),
                "Curricular units 2nd sem (grade)": float(form_data.get("Curricular units 2nd sem (grade)", 12.0))
            }
            
            # Boundary range validation
            if not (15 <= student_dict["Age at enrollment"] <= 80):
                raise ValueError("Age at enrollment must be between 15 and 80.")
            if not (0 <= student_dict["Admission grade"] <= 200):
                raise ValueError("Admission grade must be between 0 and 200.")
            if not (0 <= student_dict["Previous qualification (grade)"] <= 200):
                raise ValueError("Previous qualification grade must be between 0 and 200.")
            if not (0 <= student_dict["Curricular units 1st sem (grade)"] <= 20):
                raise ValueError("1st Semester Grade must be between 0.0 and 20.0.")
            if not (0 <= student_dict["Curricular units 2nd sem (grade)"] <= 20):
                raise ValueError("2nd Semester Grade must be between 0.0 and 20.0.")

            # Run prediction & recommendation pipelines
            global ml_model
            if ml_model is None:
                ml_model = load_trained_model()

            dropout_prob, risk_level, top_reasons = predict_risk(student_dict, model=ml_model)
            recommendations = get_recommendations(risk_level, top_reasons)
            
            prediction_result = {
                "dropout_prob": dropout_prob,
                "risk_level": risk_level,
                "top_reasons": top_reasons,
                "recommendations": recommendations
            }
            
        except ValueError as e:
            flash(f"Input Validation Error: {str(e)}", "danger")
        except Exception as e:
            flash(f"Error during prediction: {str(e)}", "danger")

    return render_template("predict.html", prediction_result=prediction_result, form_data=form_data)


@app.route("/students/save", methods=["POST"])
@login_required
def save_student():
    """Saves a predicted student record to the SQLite database."""
    name = request.form.get("save_name", "").strip()
    roll_no = request.form.get("save_roll_no", "").strip()
    
    if not name or not roll_no:
        flash("Student Name and Roll Number are required to save a record.", "danger")
        return redirect(url_for("predict"))
        
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if roll_no already exists
        existing = cursor.execute("SELECT id FROM students WHERE roll_no = ?", (roll_no,)).fetchone()
        if existing:
            flash(f"Roll Number '{roll_no}' already exists in the database!", "danger")
            conn.close()
            return redirect(url_for("predict"))

        cursor.execute(
            """
            INSERT INTO students (
                name, roll_no, gender, age_at_enrollment, scholarship_holder,
                tuition_fees_up_to_date, debtor, displaced, admission_grade,
                previous_qualification_grade, sem1_enrolled, sem1_evaluations,
                sem1_approved, sem1_grade, sem2_enrolled, sem2_evaluations,
                sem2_approved, sem2_grade, probability, risk_level, is_sample
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
            """,
            (
                name,
                roll_no,
                int(request.form.get("Gender", 1)),
                float(request.form.get("Age at enrollment", 20)),
                int(request.form.get("Scholarship holder", 0)),
                int(request.form.get("Tuition fees up to date", 1)),
                int(request.form.get("Debtor", 0)),
                int(request.form.get("Displaced", 1)),
                float(request.form.get("Admission grade", 130.0)),
                float(request.form.get("Previous qualification (grade)", 130.0)),
                int(request.form.get("Curricular units 1st sem (enrolled)", 6)),
                int(request.form.get("Curricular units 1st sem (evaluations)", 6)),
                int(request.form.get("Curricular units 1st sem (approved)", 5)),
                float(request.form.get("Curricular units 1st sem (grade)", 12.0)),
                int(request.form.get("Curricular units 2nd sem (enrolled)", 6)),
                int(request.form.get("Curricular units 2nd sem (evaluations)", 6)),
                int(request.form.get("Curricular units 2nd sem (approved)", 5)),
                float(request.form.get("Curricular units 2nd sem (grade)", 12.0)),
                float(request.form.get("probability", 0.0)),
                request.form.get("risk_level", "Low")
            )
        )
        conn.commit()
        conn.close()
        
        flash(f"Student record for '{name}' ({roll_no}) saved successfully!", "success")
        return redirect(url_for("students"))
    except Exception as e:
        flash(f"Database Save Error: {str(e)}", "danger")
        return redirect(url_for("predict"))


# ------------------------------------------------------------------------------
# STUDENTS DIRECTORY & CRUD ROUTES
# ------------------------------------------------------------------------------

@app.route("/students")
@login_required
def students():
    """Students directory list with search and risk level filtering."""
    search_query = request.args.get("search", "").strip()
    selected_risk = request.args.get("risk_level", "All").strip()
    
    conn = get_db_connection()
    query = "SELECT * FROM students WHERE 1=1"
    params = []
    
    if search_query:
        query += " AND (name LIKE ? OR roll_no LIKE ?)"
        params.extend([f"%{search_query}%", f"%{search_query}%"])
        
    if selected_risk != "All" and selected_risk in ["Low", "Medium", "High"]:
        query += " AND risk_level = ?"
        params.append(selected_risk)
        
    query += " ORDER BY created_at DESC"
    
    students_list = conn.execute(query, params).fetchall()
    conn.close()
    
    return render_template(
        "students.html",
        students_list=students_list,
        search_query=search_query,
        selected_risk=selected_risk
    )


@app.route("/students/view/<int:student_id>")
@login_required
def view_student(student_id):
    """View detailed student profile."""
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    
    if not student:
        flash("Student record not found.", "danger")
        return redirect(url_for("students"))
        
    # Reconstruct student dict for reasoning & recommendations
    student_dict = {
        "Gender": student["gender"],
        "Age at enrollment": student["age_at_enrollment"],
        "Scholarship holder": student["scholarship_holder"],
        "Tuition fees up to date": student["tuition_fees_up_to_date"],
        "Debtor": student["debtor"],
        "Displaced": student["displaced"],
        "Admission grade": student["admission_grade"],
        "Previous qualification (grade)": student["previous_qualification_grade"],
        "Curricular units 1st sem (enrolled)": student["sem1_enrolled"],
        "Curricular units 1st sem (evaluations)": student["sem1_evaluations"],
        "Curricular units 1st sem (approved)": student["sem1_approved"],
        "Curricular units 1st sem (grade)": student["sem1_grade"],
        "Curricular units 2nd sem (enrolled)": student["sem2_enrolled"],
        "Curricular units 2nd sem (evaluations)": student["sem2_evaluations"],
        "Curricular units 2nd sem (approved)": student["sem2_approved"],
        "Curricular units 2nd sem (grade)": student["sem2_grade"]
    }
    
    top_reasons = extract_risk_reasons(student_dict)
    recommendations = get_recommendations(student["risk_level"], top_reasons)
    
    return render_template("student_detail.html", student=student, top_reasons=top_reasons, recommendations=recommendations)


@app.route("/students/add", methods=["GET", "POST"])
@login_required
def add_student():
    """Add a new student directly to database."""
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        
        try:
            student_dict = {
                "Gender": int(request.form.get("gender", 1)),
                "Age at enrollment": float(request.form.get("age_at_enrollment", 20)),
                "Scholarship holder": int(request.form.get("scholarship_holder", 0)),
                "Tuition fees up to date": int(request.form.get("tuition_fees_up_to_date", 1)),
                "Debtor": int(request.form.get("debtor", 0)),
                "Displaced": int(request.form.get("displaced", 1)),
                "Admission grade": float(request.form.get("admission_grade", 130.0)),
                "Previous qualification (grade)": float(request.form.get("previous_qualification_grade", 130.0)),
                "Curricular units 1st sem (enrolled)": int(request.form.get("sem1_enrolled", 6)),
                "Curricular units 1st sem (evaluations)": int(request.form.get("sem1_evaluations", 6)),
                "Curricular units 1st sem (approved)": int(request.form.get("sem1_approved", 5)),
                "Curricular units 1st sem (grade)": float(request.form.get("sem1_grade", 12.0)),
                "Curricular units 2nd sem (enrolled)": int(request.form.get("sem2_enrolled", 6)),
                "Curricular units 2nd sem (evaluations)": int(request.form.get("sem2_evaluations", 6)),
                "Curricular units 2nd sem (approved)": int(request.form.get("sem2_approved", 5)),
                "Curricular units 2nd sem (grade)": float(request.form.get("sem2_grade", 12.0))
            }
            
            global ml_model
            if ml_model is None:
                ml_model = load_trained_model()
                
            prob, risk_level, _ = predict_risk(student_dict, model=ml_model)
            
            conn = get_db_connection()
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO students (
                    name, roll_no, gender, age_at_enrollment, scholarship_holder,
                    tuition_fees_up_to_date, debtor, displaced, admission_grade,
                    previous_qualification_grade, sem1_enrolled, sem1_evaluations,
                    sem1_approved, sem1_grade, sem2_enrolled, sem2_evaluations,
                    sem2_approved, sem2_grade, probability, risk_level, is_sample
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    name, roll_no, student_dict["Gender"], student_dict["Age at enrollment"],
                    student_dict["Scholarship holder"], student_dict["Tuition fees up to date"],
                    student_dict["Debtor"], student_dict["Displaced"], student_dict["Admission grade"],
                    student_dict["Previous qualification (grade)"], student_dict["Curricular units 1st sem (enrolled)"],
                    student_dict["Curricular units 1st sem (evaluations)"], student_dict["Curricular units 1st sem (approved)"],
                    student_dict["Curricular units 1st sem (grade)"], student_dict["Curricular units 2nd sem (enrolled)"],
                    student_dict["Curricular units 2nd sem (evaluations)"], student_dict["Curricular units 2nd sem (approved)"],
                    student_dict["Curricular units 2nd sem (grade)"], prob, risk_level
                )
            )
            conn.commit()
            conn.close()
            
            flash(f"Student '{name}' added successfully!", "success")
            return redirect(url_for("students"))
        except Exception as e:
            flash(f"Error adding student: {str(e)}", "danger")
            
    return render_template("student_form.html", title="Add New Student", mode="add", student=None)


@app.route("/students/edit/<int:student_id>", methods=["GET", "POST"])
@login_required
def edit_student(student_id):
    """Edit an existing student record."""
    conn = get_db_connection()
    student = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    
    if not student:
        conn.close()
        flash("Student record not found.", "danger")
        return redirect(url_for("students"))
        
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        roll_no = request.form.get("roll_no", "").strip()
        
        try:
            student_dict = {
                "Gender": int(request.form.get("gender", 1)),
                "Age at enrollment": float(request.form.get("age_at_enrollment", 20)),
                "Scholarship holder": int(request.form.get("scholarship_holder", 0)),
                "Tuition fees up to date": int(request.form.get("tuition_fees_up_to_date", 1)),
                "Debtor": int(request.form.get("debtor", 0)),
                "Displaced": int(request.form.get("displaced", 1)),
                "Admission grade": float(request.form.get("admission_grade", 130.0)),
                "Previous qualification (grade)": float(request.form.get("previous_qualification_grade", 130.0)),
                "Curricular units 1st sem (enrolled)": int(request.form.get("sem1_enrolled", 6)),
                "Curricular units 1st sem (evaluations)": int(request.form.get("sem1_evaluations", 6)),
                "Curricular units 1st sem (approved)": int(request.form.get("sem1_approved", 5)),
                "Curricular units 1st sem (grade)": float(request.form.get("sem1_grade", 12.0)),
                "Curricular units 2nd sem (enrolled)": int(request.form.get("sem2_enrolled", 6)),
                "Curricular units 2nd sem (evaluations)": int(request.form.get("sem2_evaluations", 6)),
                "Curricular units 2nd sem (approved)": int(request.form.get("sem2_approved", 5)),
                "Curricular units 2nd sem (grade)": float(request.form.get("sem2_grade", 12.0))
            }
            
            global ml_model
            if ml_model is None:
                ml_model = load_trained_model()
                
            prob, risk_level, _ = predict_risk(student_dict, model=ml_model)
            
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE students SET
                    name = ?, roll_no = ?, gender = ?, age_at_enrollment = ?,
                    scholarship_holder = ?, tuition_fees_up_to_date = ?, debtor = ?,
                    displaced = ?, admission_grade = ?, previous_qualification_grade = ?,
                    sem1_enrolled = ?, sem1_evaluations = ?, sem1_approved = ?,
                    sem1_grade = ?, sem2_enrolled = ?, sem2_evaluations = ?,
                    sem2_approved = ?, sem2_grade = ?, probability = ?, risk_level = ?
                WHERE id = ?
                """,
                (
                    name, roll_no, student_dict["Gender"], student_dict["Age at enrollment"],
                    student_dict["Scholarship holder"], student_dict["Tuition fees up to date"],
                    student_dict["Debtor"], student_dict["Displaced"], student_dict["Admission grade"],
                    student_dict["Previous qualification (grade)"], student_dict["Curricular units 1st sem (enrolled)"],
                    student_dict["Curricular units 1st sem (evaluations)"], student_dict["Curricular units 1st sem (approved)"],
                    student_dict["Curricular units 1st sem (grade)"], student_dict["Curricular units 2nd sem (enrolled)"],
                    student_dict["Curricular units 2nd sem (evaluations)"], student_dict["Curricular units 2nd sem (approved)"],
                    student_dict["Curricular units 2nd sem (grade)"], prob, risk_level, student_id
                )
            )
            conn.commit()
            conn.close()
            
            flash(f"Student record for '{name}' updated successfully!", "success")
            return redirect(url_for("view_student", student_id=student_id))
        except Exception as e:
            flash(f"Error updating student: {str(e)}", "danger")
            
    conn.close()
    return render_template("student_form.html", title="Edit Student Record", mode="edit", student=student)


@app.route("/students/delete/<int:student_id>", methods=["POST"])
@login_required
def delete_student(student_id):
    """Delete a student record from SQLite."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM students WHERE id = ?", (student_id,))
    conn.commit()
    conn.close()
    
    flash("Student record deleted successfully.", "info")
    return redirect(url_for("students"))


if __name__ == "__main__":
    print("Starting Flask Web Application on http://127.0.0.1:5000")
    app.run(debug=True, host="127.0.0.1", port=5000)
