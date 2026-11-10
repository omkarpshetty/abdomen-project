-- AI Organ Dose System Database Schema
-- Run this in Supabase SQL Editor (SQL Editor → New Query → Paste → Run)

-- Patients table
CREATE TABLE IF NOT EXISTS patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_uid TEXT UNIQUE NOT NULL,
    name TEXT,
    age INTEGER,
    gender TEXT,
    scan_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Scans table
CREATE TABLE IF NOT EXISTS scans (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID REFERENCES patients(id) ON DELETE CASCADE,
    scan_type TEXT,
    kvp REAL,
    mas REAL,
    ctdivol REAL,
    num_slices INTEGER,
    slice_thickness REAL,
    manufacturer TEXT,
    study_description TEXT,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Organ measurements table
CREATE TABLE IF NOT EXISTS organ_measurements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE,
    organ_name TEXT NOT NULL,
    volume_cm3 REAL,
    mean_hu REAL,
    median_hu REAL,
    std_hu REAL,
    position_x REAL,
    position_y REAL,
    position_z REAL,
    compactness REAL
);

-- Organ doses table
CREATE TABLE IF NOT EXISTS organ_doses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE,
    organ_name TEXT NOT NULL,
    absorbed_dose_mgy REAL,
    effective_dose_msv REAL,
    tissue_weight REAL,
    model_used TEXT,
    confidence REAL
);

-- Patient doses table
CREATE TABLE IF NOT EXISTS patient_doses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE,
    total_absorbed_dose_mgy REAL,
    total_effective_dose_msv REAL,
    risk_category TEXT,
    highest_dose_organ TEXT,
    processing_time_seconds REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_patients_uid ON patients(patient_uid);
CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX IF NOT EXISTS idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX IF NOT EXISTS idx_patient_doses_scan ON patient_doses(scan_id);

-- Done! All tables and indexes created.
