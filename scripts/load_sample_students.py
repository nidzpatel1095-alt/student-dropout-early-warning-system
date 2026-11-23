# ==============================================================================
# SAMPLE DATA SEED SCRIPT: scripts/load_sample_students.py
# Samples 30 real rows from data/data.csv and populates the database
# so the Flask dashboard has initial real data records for demonstration.
# ==============================================================================

import os
import sys
import sqlite3
import pandas as pd

# Add parent directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import DATA_PATH, DB_PATH, APP_FEATURES, RANDOM_STATE
from database.init_db import init_db
from ml.predict_utils import predict_risk, load_trained_model


def load_sample_students():
    current_db_path = os.environ.get("DROPOUT_DB_PATH", DB_PATH)
    current_db_dir = os.path.dirname(current_db_path)
    if current_db_dir:
        os.makedirs(current_db_dir, exist_ok=True)
        
    # Initialize database if not already initialized
    if not os.path.exists(current_db_path):
        init_db()
        
    model = load_trained_model()
    if model is None:
        raise FileNotFoundError("Trained model not found at outputs/model.joblib!")

    print(f"[SAMPLE LOADER] Reading dataset and sampling 30 rows for '{current_db_path}'...")
    df = pd.read_csv(DATA_PATH, sep=";")
    df.columns = df.columns.str.strip()
    
    # Sample 30 real rows with fixed random_state
    sample_df = df.sample(n=30, random_state=RANDOM_STATE).reset_index(drop=True)
    
    conn = sqlite3.connect(current_db_path)
    cursor = conn.cursor()
    
    # Check if sample records already exist to avoid duplicate inserts
    cursor.execute("SELECT COUNT(*) FROM students WHERE is_sample = 1")
    existing_samples = cursor.fetchone()[0]
    if existing_samples > 0:
        print(f"Sample records already loaded ({existing_samples} existing sample students). Skipping insert.")
        conn.close()
        return

    inserted_count = 0
    for idx, row in sample_df.iterrows():
        student_id_num = idx + 1
        name = f"Student {student_id_num:03d}"
        roll_no = f"STU-{student_id_num:03d}"
        
        # Build feature dictionary for predict_risk
        student_dict = {feat: row[feat] for feat in APP_FEATURES}
        
        prob, risk_level, _ = predict_risk(student_dict, model=model)
        
        cursor.execute(
            """
            INSERT INTO students (
                name, roll_no, gender, age_at_enrollment, scholarship_holder,
                tuition_fees_up_to_date, debtor, displaced, admission_grade,
                previous_qualification_grade, sem1_enrolled, sem1_evaluations,
                sem1_approved, sem1_grade, sem2_enrolled, sem2_evaluations,
                sem2_approved, sem2_grade, probability, risk_level, is_sample
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1)
            """,
            (
                name,
                roll_no,
                int(row["Gender"]),
                float(row["Age at enrollment"]),
                int(row["Scholarship holder"]),
                int(row["Tuition fees up to date"]),
                int(row["Debtor"]),
                int(row["Displaced"]),
                float(row["Admission grade"]),
                float(row["Previous qualification (grade)"]),
                int(row["Curricular units 1st sem (enrolled)"]),
                int(row["Curricular units 1st sem (evaluations)"]),
                int(row["Curricular units 1st sem (approved)"]),
                float(row["Curricular units 1st sem (grade)"]),
                int(row["Curricular units 2nd sem (enrolled)"]),
                int(row["Curricular units 2nd sem (evaluations)"]),
                int(row["Curricular units 2nd sem (approved)"]),
                float(row["Curricular units 2nd sem (grade)"]),
                float(prob),
                risk_level
            )
        )
        inserted_count += 1
        
    conn.commit()
    conn.close()
    print(f"Successfully inserted {inserted_count} sample student records into database.")


if __name__ == "__main__":
    load_sample_students()
