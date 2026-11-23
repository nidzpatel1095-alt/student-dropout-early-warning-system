-- database/schema.sql
-- Database Schema for Student Dropout Early Warning System

CREATE TABLE IF NOT EXISTS users (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS students (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    roll_no TEXT UNIQUE NOT NULL,
    gender INTEGER NOT NULL,
    age_at_enrollment REAL NOT NULL,
    scholarship_holder INTEGER NOT NULL,
    tuition_fees_up_to_date INTEGER NOT NULL,
    debtor INTEGER NOT NULL,
    displaced INTEGER NOT NULL,
    admission_grade REAL NOT NULL,
    previous_qualification_grade REAL NOT NULL,
    sem1_enrolled INTEGER NOT NULL,
    sem1_evaluations INTEGER NOT NULL,
    sem1_approved INTEGER NOT NULL,
    sem1_grade REAL NOT NULL,
    sem2_enrolled INTEGER NOT NULL,
    sem2_evaluations INTEGER NOT NULL,
    sem2_approved INTEGER NOT NULL,
    sem2_grade REAL NOT NULL,
    probability REAL NOT NULL,
    risk_level TEXT NOT NULL,
    is_sample INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
