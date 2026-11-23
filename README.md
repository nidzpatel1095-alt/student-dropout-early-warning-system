# Student Dropout Early Warning System using Machine Learning

A web-based early warning system that predicts student dropout risk using Machine Learning. The application classifies students into **Low**, **Medium**, or **High** risk levels, highlights key contributing risk factors, and provides academic advisors with actionable intervention suggestions.

---

## 🌟 Key Features

- **Real-Time Risk Prediction**: Evaluates 16 academic and demographic indicators to estimate dropout probability.
- **Risk Categorization**: Classifies students into Low (< 33%), Medium (33% - 66%), and High (>= 66%) risk tiers.
- **Advisor Recommendations**: Generates tailored academic and financial support suggestions based on flagged risk factors.
- **Interactive Dashboard**: Real-time statistics, risk distribution charts, and a high-risk student priority queue.
- **Student Record Management**: Full CRUD interface with search and filtering capabilities backed by SQLite.
- **Admin Authentication**: Secure session-based login for academic counselors and administrators.

---

## 🛠️ Tech Stack

- **Backend**: Python 3.14, Flask, SQLite
- **Machine Learning**: Scikit-Learn (Random Forest Classifier), Pandas, NumPy, Joblib
- **Frontend**: HTML5, CSS3, Bootstrap 5, Chart.js, FontAwesome
- **Testing**: Pytest

---

## 📁 Project Structure

```
├── app.py                        # Flask application entry point
├── config.py                     # Configuration settings and feature definitions
├── requirements.txt              # Project dependencies
├── README.md                     # Documentation
├── data/
│   └── data.csv                  # Student dataset
├── database/
│   ├── schema.sql                # SQLite schema
│   ├── init_db.py                # Database setup script
│   └── dropout_system.db         # SQLite database file
├── ml/
│   ├── train_model.py            # Model training pipeline
│   └── predict_utils.py          # Inference and recommendation utilities
├── scripts/
│   ├── load_sample_students.py   # Populates sample student records
│   └── verify_results.py         # Verification script
├── static/
│   └── style.css                 # Custom CSS stylesheet
├── templates/
│   ├── base.html                 # Base layout template
│   ├── login.html                # Admin login page
│   ├── dashboard.html            # Main dashboard
│   ├── predict.html              # Risk prediction form & results
│   ├── students.html             # Student records directory
│   ├── student_form.html         # Add/Edit student form
│   └── student_detail.html       # Individual student profile view
├── tests/
│   ├── conftest.py               # Pytest database isolation fixture
│   └── test_app.py               # Unit tests
└── outputs/
    ├── figures/                  # Visualization charts
    ├── tables/                   # Performance summary tables
    ├── model.joblib              # Serialized trained model
    └── metrics.json              # Evaluated model metrics
```

---

## 🚀 How to Run the Project

### 1. Clone the Repository & Navigate to Directory
```bash
git clone https://github.com/<your-username>/student-dropout-early-warning-system.git
cd student-dropout-early-warning-system
```

### 2. Install Required Packages
```bash
pip install -r requirements.txt
```

### 3. Initialize the Database (First-time setup)
```bash
python database/init_db.py
python scripts/load_sample_students.py
```

### 4. Start the Application
```bash
python app.py
```
Open your browser and navigate to: **`http://127.0.0.1:5000`**

**Default Admin Credentials:**
- **Username**: `admin`
- **Password**: `admin123`

---

## 🧪 Running Automated Tests

Run the test suite to verify model loading, predictions, and database safety:
```bash
python -m pytest -v
```

---

## 📊 Model Overview

- **Algorithm**: Random Forest Classifier (200 estimators, balanced class weights)
- **Validation**: 5-Fold Stratified Cross-Validation
- **Accuracy**: 87.68%
- **ROC-AUC**: 0.9219
- **F1-Score**: 0.8064
