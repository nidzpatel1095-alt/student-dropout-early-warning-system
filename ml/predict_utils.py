# ==============================================================================
# PREDICTION UTILITIES: ml/predict_utils.py
# Helper functions to load the trained model, calculate dropout probability,
# categorize risk levels, extract rule-based risk indicators, and return actions.
# ==============================================================================

import os
import joblib
import pandas as pd
from config import MODEL_PATH, APP_FEATURES, RISK_LOW_THRESHOLD, RISK_HIGH_THRESHOLD


def load_trained_model(model_path=MODEL_PATH):
    """
    Load the trained Random Forest model from disk.
    
    Returns:
        model: Trained sklearn Random Forest model or None if file doesn't exist.
    """
    if not os.path.exists(model_path):
        return None
    return joblib.load(model_path)


def predict_risk(student_dict, model=None):
    """
    Predict dropout probability, risk level, and extract risk indicators for a student.
    
    Parameters:
        student_dict (dict): Feature dictionary containing values for the 16 APP_FEATURES.
        model (optional): Pre-loaded sklearn model instance.
        
    Returns:
        tuple: (dropout_probability, risk_level, top_reasons)
            - dropout_probability (float): Probability between 0.0 and 1.0
            - risk_level (str): 'Low', 'Medium', or 'High'
            - top_reasons (list): List of up to 3 rule-based risk indicator strings
    """
    # Load model if not provided
    if model is None:
        model = load_trained_model()
    
    if model is None:
        raise FileNotFoundError(
            f"Trained model file not found at '{MODEL_PATH}'. "
            "Please run 'python ml/train_model.py' first to train and save the model."
        )
    
    # Create DataFrame with a single row and exact column order required by the model
    student_df = pd.DataFrame([student_dict])[APP_FEATURES]
    
    # Get predicted probability for class 1 (Dropout)
    probabilities = model.predict_proba(student_df)[0]
    dropout_prob = float(probabilities[1])
    
    # Map probability to risk band
    if dropout_prob < RISK_LOW_THRESHOLD:
        risk_level = "Low"
    elif dropout_prob < RISK_HIGH_THRESHOLD:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    # Extract rule-based explainable risk indicators
    top_reasons = extract_risk_reasons(student_dict)
    
    return dropout_prob, risk_level, top_reasons


def extract_risk_reasons(student_dict):
    """
    Identify up to 3 explainable rule-based risk indicators from a student's feature inputs.
    
    Note: These are statistical risk indicators derived from empirical feature patterns,
    not direct causal proofs.
    """
    reasons = []
    
    # 1. Check if tuition fees are not up to date
    if int(student_dict.get("Tuition fees up to date", 1)) == 0:
        reasons.append("Tuition fees are not up to date")
        
    # 2. Check if student is marked as debtor
    if int(student_dict.get("Debtor", 0)) == 1:
        reasons.append("Student has outstanding student debt")
        
    # 3. Check 1st semester approval ratio (approved / enrolled)
    sem1_enrolled = float(student_dict.get("Curricular units 1st sem (enrolled)", 0))
    sem1_approved = float(student_dict.get("Curricular units 1st sem (approved)", 0))
    if sem1_enrolled > 0 and (sem1_approved / sem1_enrolled) < 0.5:
        reasons.append("Low 1st semester course pass rate (approved less than 50% of enrolled units)")
        
    # 4. Check 2nd semester approval ratio (approved / enrolled)
    sem2_enrolled = float(student_dict.get("Curricular units 2nd sem (enrolled)", 0))
    sem2_approved = float(student_dict.get("Curricular units 2nd sem (approved)", 0))
    if sem2_enrolled > 0 and (sem2_approved / sem2_enrolled) < 0.5:
        reasons.append("Low 2nd semester course pass rate (approved less than 50% of enrolled units)")

    # 5. Check 1st semester average grade (passing grade in Portugal is 10 out of 20)
    sem1_grade = float(student_dict.get("Curricular units 1st sem (grade)", 0))
    if sem1_approved > 0 and sem1_grade < 10.0:
        reasons.append(f"Low 1st semester average grade ({sem1_grade:.1f} out of 20)")

    # 6. Check 2nd semester average grade
    sem2_grade = float(student_dict.get("Curricular units 2nd sem (grade)", 0))
    if sem2_approved > 0 and sem2_grade < 10.0:
        reasons.append(f"Low 2nd semester average grade ({sem2_grade:.1f} out of 20)")

    # 7. Check if student has no scholarship
    if int(student_dict.get("Scholarship holder", 0)) == 0:
        reasons.append("No scholarship financial support")

    # 8. Check if age at enrollment is older than traditional student age
    age = float(student_dict.get("Age at enrollment", 18))
    if age > 25:
        reasons.append(f"Older age at enrollment ({int(age)} years old)")

    # Return top 3 flagged reasons (if none, return a positive status)
    if not reasons:
        reasons.append("No major risk indicators flagged (Good standing)")
        
    return reasons[:3]


def get_recommendations(risk_level, top_reasons=None):
    """
    Map risk level and risk indicators to suggested support actions.
    
    Parameters:
        risk_level (str): 'High', 'Medium', or 'Low'
        top_reasons (list): List of risk indicator strings.
        
    Returns:
        list: Recommended support actions for academic advisors.
    """
    actions = []
    reasons_str = " ".join(top_reasons) if top_reasons else ""
    
    if risk_level == "High":
        actions.append("Contact student immediately for an urgent academic and personal advising session.")
        actions.append("Assign a dedicated academic mentor for weekly progress check-ins.")
        if "tuition" in reasons_str.lower() or "debt" in reasons_str.lower():
            actions.append("Connect student with Financial Aid Office for payment plan options or emergency grants.")
        else:
            actions.append("Offer academic tutoring and remedial support for core courses.")
            
    elif risk_level == "Medium":
        actions.append("Schedule an academic counseling check-in before the end of the term.")
        if "pass rate" in reasons_str.lower() or "grade" in reasons_str.lower():
            actions.append("Recommend attending subject-specific peer tutoring and study skills workshops.")
        if "scholarship" in reasons_str.lower() or "tuition" in reasons_str.lower():
            actions.append("Provide guidance on institutional financial support and scholarship criteria.")
        actions.append("Monitor 2nd semester attendance and assignment submissions.")
        
    else:  # Low
        actions.append("Student is performing well. Maintain standard academic support.")
        actions.append("Encourage student to participate in honors projects or peer mentoring roles.")
        
    return actions
