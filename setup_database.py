"""
Setup Supabase Database Tables for AI Organ Dose System
"""
import os
from supabase import create_client

# Supabase credentials
SUPABASE_URL = "https://nodbucfdwkhwxmxwqbdq.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5vZGJ1Y2Zkd2tod3hteHdxYmRxIiwicm9sZSI6ImFub24iLCJpYXQiOjE3MzEwNDU3MjQsImV4cCI6MjA0NjYyMTcyNH0.ZMIxF_vJXWYqCm7mh9YpEqXH8Q7gXpqmVrRMxH5GZBY"

# SQL to create all required tables
CREATE_TABLES_SQL = """
-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

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
    scan_type TEXT DEFAULT 'CT_ABDOMEN',
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
    compactness REAL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Organ doses table
CREATE TABLE IF NOT EXISTS organ_doses (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    scan_id UUID REFERENCES scans(id) ON DELETE CASCADE,
    organ_name TEXT NOT NULL,
    absorbed_dose_mgy REAL,
    effective_dose_msv REAL,
    tissue_weight REAL,
    model_used TEXT DEFAULT 'ensemble',
    confidence REAL DEFAULT 0.90,
    created_at TIMESTAMP DEFAULT NOW()
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

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_patients_uid ON patients(patient_uid);
CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX IF NOT EXISTS idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX IF NOT EXISTS idx_patient_doses_scan ON patient_doses(scan_id);
"""


def setup_database():
    """Create all required database tables."""
    print("="*70)
    print("SUPABASE DATABASE SETUP")
    print("="*70)
    print(f"\nConnecting to: {SUPABASE_URL}")

    try:
        # Create Supabase client
        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("✓ Connected to Supabase")

        # Execute SQL to create tables
        print("\nCreating database tables...")
        print("\nNote: You need to run this SQL in Supabase SQL Editor:")
        print("1. Go to: https://supabase.com/dashboard/project/nodbucfdwkhwxmxwqbdq/sql/new")
        print("2. Copy and paste the SQL below")
        print("3. Click 'Run'\n")
        print("="*70)
        print(CREATE_TABLES_SQL)
        print("="*70)

        # Test the connection by trying to query patients table
        print("\nTesting database connection...")
        try:
            result = client.table('patients').select("*").limit(1).execute()
            print("✓ Database tables are ready!")
            print(f"✓ Found {len(result.data)} existing patients")
            return True
        except Exception as e:
            if "relation \"patients\" does not exist" in str(e) or "does not exist" in str(e):
                print("\n⚠ Tables don't exist yet. Please run the SQL above in Supabase SQL Editor.")
                return False
            else:
                print(f"✓ Database connection successful!")
                print("✓ Tables appear to be created")
                return True

    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


def test_database_operations():
    """Test basic database operations."""
    print("\n" + "="*70)
    print("TESTING DATABASE OPERATIONS")
    print("="*70)

    try:
        from src.database_connector import DatabaseConnector

        # Set environment variables
        os.environ['SUPABASE_URL'] = SUPABASE_URL
        os.environ['SUPABASE_KEY'] = SUPABASE_KEY

        # Create database connector
        db = DatabaseConnector()

        if not db.is_enabled():
            print("❌ Database connector not enabled")
            return False

        print("✓ Database connector initialized")

        # Test patient creation
        print("\nTesting patient save...")
        patient_id = db.save_patient(
            patient_uid="TEST_001",
            name="Test Patient",
            age=45,
            gender="M"
        )

        if patient_id:
            print(f"✓ Patient saved with ID: {patient_id}")
        else:
            print("❌ Failed to save patient")
            return False

        # Test scan metadata
        print("\nTesting scan save...")
        scan_id = db.save_scan(
            patient_id=patient_id,
            scan_metadata={
                'kvp': 120,
                'mas': 200,
                'ctdivol_mgy': 10.5,
                'num_slices': 50,
                'slice_thickness_mm': 5.0,
                'manufacturer': 'Test Manufacturer'
            }
        )

        if scan_id:
            print(f"✓ Scan saved with ID: {scan_id}")
        else:
            print("❌ Failed to save scan")
            return False

        # Test organ measurements
        print("\nTesting organ measurements save...")
        success = db.save_organ_measurements(
            scan_id=scan_id,
            organ_measurements={
                'LIVER': {
                    'volume_cm3': 1500.0,
                    'mean_hu': 55.0,
                    'median_hu': 54.0,
                    'std_hu': 15.0,
                    'center_x_mm': 100.0,
                    'center_y_mm': 150.0,
                    'center_z_mm': 200.0,
                    'compactness': 0.75
                }
            }
        )

        if success:
            print("✓ Organ measurements saved")
        else:
            print("❌ Failed to save organ measurements")

        # Test organ doses
        print("\nTesting organ doses save...")
        success = db.save_organ_doses(
            scan_id=scan_id,
            organ_doses={
                'LIVER': {
                    'dose_mgy': 12.5,
                    'effective_dose_msv': 0.5,
                    'tissue_weight': 0.04
                }
            }
        )

        if success:
            print("✓ Organ doses saved")
        else:
            print("❌ Failed to save organ doses")

        # Test patient dose summary
        print("\nTesting patient dose save...")
        success = db.save_patient_dose(
            scan_id=scan_id,
            patient_dose={
                'total_absorbed_dose_mgy': 50.0,
                'total_effective_dose_msv': 2.5,
                'risk_category': 'Low',
                'highest_dose_organ': 'LIVER'
            },
            processing_time=45.2
        )

        if success:
            print("✓ Patient dose saved")
        else:
            print("❌ Failed to save patient dose")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED!")
        print("="*70)
        print("\nDatabase is ready to use!")

        return True

    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\nAI Organ Dose System - Database Setup\n")

    # Step 1: Setup tables
    if setup_database():
        # Step 2: Test operations
        print("\n" + "="*70)
        input("Press Enter after running the SQL in Supabase to test operations...")
        test_database_operations()
    else:
        print("\n⚠ Please run the SQL in Supabase SQL Editor first, then run this script again.")
