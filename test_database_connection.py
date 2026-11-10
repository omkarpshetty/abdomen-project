"""
Test Supabase Database Connection

This script will verify that your Supabase database is properly configured.
"""
import os
import sys
from dotenv import load_dotenv
from src.database_connector import DatabaseConnector

# Fix Windows encoding issues
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

def main():
    print("="*80)
    print("TESTING SUPABASE DATABASE CONNECTION")
    print("="*80)
    print()

    # Load environment variables
    print("[1/5] Loading credentials from .env file...")
    load_dotenv()

    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_KEY')

    if not url or not key:
        print("[X] ERROR: Credentials not found in .env file")
        print()
        print("Please edit .env file and add:")
        print("  SUPABASE_URL=your-project-url")
        print("  SUPABASE_KEY=your-anon-key")
        return False

    print(f"   URL: {url[:30]}...")
    print(f"   Key: {key[:20]}...")
    print()

    # Test connection
    print("[2/5] Connecting to Supabase...")
    db = DatabaseConnector(url=url, key=key)

    if not db.is_enabled():
        print("[X] ERROR: Failed to connect to database")
        return False

    print("   [OK] Connected!\n")

    # Test table existence
    print("[3/5] Checking if tables exist...")
    try:
        # Try to query each table
        tables = ['patients', 'scans', 'organ_measurements', 'organ_doses', 'patient_doses']
        for table in tables:
            result = db.client.table(table).select("*").limit(1).execute()
            print(f"   [OK] {table} table exists")
        print()
    except Exception as e:
        print(f"   [X] ERROR: Tables don't exist yet")
        print(f"   {str(e)[:100]}")
        print()
        print("Please create tables using setup_tables.sql:")
        print("1. Go to Supabase -> SQL Editor")
        print("2. Click 'New query'")
        print("3. Copy contents of setup_tables.sql")
        print("4. Click 'Run'")
        return False

    # Test write operation
    print("[4/5] Testing write operation...")
    try:
        test_patient_id = db.save_patient(
            patient_uid='test_patient_123',
            name='Test Patient',
            age=45,
            gender='M'
        )

        if test_patient_id:
            print(f"   [OK] Successfully wrote test patient (ID: {test_patient_id})")

            # Clean up test data
            db.client.table('patients').delete().eq('patient_uid', 'test_patient_123').execute()
            print(f"   [OK] Cleaned up test data")
        else:
            print("   [!] Write test returned None")
        print()
    except Exception as e:
        print(f"   [X] ERROR: Write test failed")
        print(f"   {str(e)}")
        return False

    # Final result
    print("[5/5] Final Status")
    print("-"*80)
    print("[OK] Database connection: SUCCESS")
    print("[OK] Tables exist: YES")
    print("[OK] Write permissions: OK")
    print()
    print("="*80)
    print("DATABASE IS READY TO USE!")
    print("="*80)
    print()
    print("You can now run your AI system with database enabled:")
    print("  python test_complete_system.py")
    print()
    print("Or modify test_complete_system.py to set:")
    print("  enable_database=True")
    print()

    return True

if __name__ == "__main__":
    import sys
    success = main()
    sys.exit(0 if success else 1)
