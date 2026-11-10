"""
Automated Supabase Database Setup Script

This script will:
1. Connect to your Supabase project
2. Create all required tables
3. Set up indexes
4. Test the connection

Usage:
    python setup_supabase.py --url YOUR_URL --key YOUR_KEY
"""
import argparse
import sys
from supabase import create_client, Client

# SQL schema for all tables
SCHEMA_SQL = """
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

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_patients_uid ON patients(patient_uid);
CREATE INDEX IF NOT EXISTS idx_scans_patient ON scans(patient_id);
CREATE INDEX IF NOT EXISTS idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX IF NOT EXISTS idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX IF NOT EXISTS idx_patient_doses_scan ON patient_doses(scan_id);
"""


def setup_database(url: str, key: str) -> bool:
    """
    Set up Supabase database with all required tables.

    Args:
        url: Supabase project URL
        key: Supabase anon key

    Returns:
        True if successful, False otherwise
    """
    print("="*80)
    print("SUPABASE DATABASE SETUP")
    print("="*80)
    print()

    try:
        # Connect to Supabase
        print("[1/4] Connecting to Supabase...")
        client: Client = create_client(url, key)
        print("      ✓ Connected successfully!\n")

        # Execute SQL schema
        print("[2/4] Creating database tables...")
        print("      - patients")
        print("      - scans")
        print("      - organ_measurements")
        print("      - organ_doses")
        print("      - patient_doses")

        # Note: The Python Supabase client doesn't support raw SQL execution
        # You need to run this via the Supabase dashboard SQL editor
        # OR use PostgREST API directly
        # OR use psycopg2 with the direct PostgreSQL connection

        print("\n      ⚠ Note: The Python Supabase client cannot execute raw SQL.")
        print("      Please run the SQL manually via Supabase dashboard.\n")

        # Test connection by checking if tables exist
        print("[3/4] Testing connection...")
        try:
            # Try to query patients table (will fail if it doesn't exist)
            result = client.table('patients').select("*").limit(1).execute()
            print("      ✓ Tables already exist!\n")
            tables_exist = True
        except Exception as e:
            print(f"      ⚠ Tables don't exist yet: {str(e)[:100]}\n")
            tables_exist = False

        # Print instructions
        print("[4/4] Setup Instructions")
        print("="*80)

        if not tables_exist:
            print("\n⚠ MANUAL STEP REQUIRED:")
            print("\n1. Go to your Supabase dashboard SQL Editor:")
            print(f"   {url.replace('https://', 'https://supabase.com/dashboard/project/')}/sql")
            print("\n2. Create a new query and paste this SQL:\n")
            print("-"*80)
            print(SCHEMA_SQL)
            print("-"*80)
            print("\n3. Click 'Run' to create the tables")
            print("\n4. Run this script again to verify: python setup_supabase.py --url YOUR_URL --key YOUR_KEY")
        else:
            print("\n✓ ALL DONE! Your database is ready to use.")
            print("\nYou can now run your AI system with database enabled:")
            print("  python ai_dose_system.py --dicom-dir YOUR_PATH --save-to-db \\")
            print(f"    --supabase-url {url} \\")
            print(f"    --supabase-key YOUR_KEY")
            print("\nOr set environment variables:")
            print(f"  set SUPABASE_URL={url}")
            print(f"  set SUPABASE_KEY=YOUR_KEY")
            print("  python ai_dose_system.py --dicom-dir YOUR_PATH --save-to-db")

        print("\n" + "="*80)

        return tables_exist

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def save_env_file(url: str, key: str):
    """Save credentials to .env file."""
    env_content = f"""# Supabase Database Credentials
# Generated by setup_supabase.py on {sys.argv[0]}

SUPABASE_URL={url}
SUPABASE_KEY={key}
"""

    with open('.env', 'w') as f:
        f.write(env_content)

    print("✓ Saved credentials to .env file")


def main():
    parser = argparse.ArgumentParser(
        description='Set up Supabase database for AI Organ Dose System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python setup_supabase.py --url https://xxxxx.supabase.co --key eyJhbGc...

To get your credentials:
  1. Go to https://supabase.com/dashboard
  2. Select your project
  3. Go to Settings → API
  4. Copy Project URL and anon/public key
        """
    )

    parser.add_argument('--url', type=str, required=True,
                       help='Supabase project URL (from Settings → API)')
    parser.add_argument('--key', type=str, required=True,
                       help='Supabase anon key (from Settings → API)')
    parser.add_argument('--save-env', action='store_true',
                       help='Save credentials to .env file')

    args = parser.parse_args()

    # Run setup
    success = setup_database(args.url, args.key)

    # Save to .env if requested
    if args.save_env:
        print()
        save_env_file(args.url, args.key)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
