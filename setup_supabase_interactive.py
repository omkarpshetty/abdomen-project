"""
Quick Supabase Setup - Interactive Script

This will guide you through connecting to Supabase.
"""
import os
from pathlib import Path

def main():
    print("="*80)
    print("SUPABASE SETUP - INTERACTIVE GUIDE")
    print("="*80)
    print()

    print("You're currently on the API Keys page. Great!")
    print()

    # Step 1: Get Project URL
    print("[Step 1] Get your Project URL")
    print("-"*80)
    print("1. In your Supabase dashboard, click 'Settings' (left sidebar)")
    print("2. Click 'General'")
    print("3. Look for 'Project URL' - it should be:")
    print("   https://dogywbhaeyuqawmhoreq.supabase.co")
    print()
    project_url = input("Paste your Project URL here: ").strip()
    print()

    # Step 2: Get Anon Key
    print("[Step 2] Get your Anon/Public Key")
    print("-"*80)
    print("You're already on Settings → API Keys!")
    print("1. Look for 'anon' or 'public' key (the publishable one)")
    print("2. Click the copy icon next to it")
    print()
    anon_key = input("Paste your anon key here: ").strip()
    print()

    # Save to .env
    print("[Step 3] Saving credentials...")
    env_content = f"""# Supabase Database Credentials
# Generated on {os.popen('date').read().strip()}

SUPABASE_URL={project_url}
SUPABASE_KEY={anon_key}
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✓ Saved to .env file")
    print()

    # Install dependencies
    print("[Step 4] Installing dependencies...")
    print()
    print("Run these commands:")
    print("  pip install supabase python-dotenv")
    print()

    # Create tables
    print("[Step 5] Create Database Tables")
    print("-"*80)
    print("Now you need to create the database tables:")
    print()
    print("1. Go to SQL Editor in Supabase:")
    print(f"   {project_url.replace('https://', 'https://supabase.com/dashboard/project/').replace('.supabase.co', '')}/sql")
    print()
    print("2. Click 'New query'")
    print()
    print("3. Copy the SQL from 'setup_tables.sql' file (I'll create it now)")
    print()
    print("4. Click 'Run' to create all tables")
    print()

    # Create SQL file
    sql_content = """-- AI Organ Dose System Database Schema
-- Run this in Supabase SQL Editor

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

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_patients_uid ON patients(patient_uid);
CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX IF NOT EXISTS idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX IF NOT EXISTS idx_patient_doses_scan ON patient_doses(scan_id);
"""

    with open('setup_tables.sql', 'w') as f:
        f.write(sql_content)

    print("✓ Created setup_tables.sql")
    print()

    # Final steps
    print("[Step 6] After creating tables, test the connection:")
    print("-"*80)
    print("  python test_database_connection.py")
    print()

    print("="*80)
    print("SETUP COMPLETE!")
    print("="*80)
    print()
    print("Next steps:")
    print("1. Install: pip install supabase python-dotenv")
    print("2. Create tables using setup_tables.sql in Supabase SQL Editor")
    print("3. Test: python test_database_connection.py")
    print("4. Run your system with database: python test_complete_system.py")
    print()

if __name__ == "__main__":
    main()
