"""
Supabase Database Connector

Handles storage and retrieval of patient data, organ measurements,
and dose results in Supabase (PostgreSQL).
"""
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
import json

try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("Warning: Supabase library not installed. Database features disabled.")
    print("Install with: pip install supabase")


class DatabaseConnector:
    """
    Connect to Supabase database for storing patient dose data.

    Set environment variables:
        SUPABASE_URL=your-project-url
        SUPABASE_KEY=your-anon-key
    """

    def __init__(self, url: str = None, key: str = None):
        """
        Initialize database connection.

        Args:
            url: Supabase project URL (or uses SUPABASE_URL env var)
            key: Supabase anon key (or uses SUPABASE_KEY env var)
        """
        if not SUPABASE_AVAILABLE:
            self.client = None
            self.enabled = False
            return

        self.url = url or os.getenv('SUPABASE_URL')
        self.key = key or os.getenv('SUPABASE_KEY')

        if not self.url or not self.key:
            print("Warning: Supabase credentials not provided. Database disabled.")
            self.client = None
            self.enabled = False
        else:
            try:
                self.client: Client = create_client(self.url, self.key)
                self.enabled = True
                print("✓ Connected to Supabase database")
            except Exception as e:
                print(f"Warning: Failed to connect to Supabase: {e}")
                self.client = None
                self.enabled = False

    def is_enabled(self) -> bool:
        """Check if database is available."""
        return self.enabled and self.client is not None

    def save_patient(self, patient_uid: str, name: str = None,
                     age: int = None, gender: str = None,
                     scan_date: str = None) -> Optional[str]:
        """
        Save or update patient record.

        Returns:
            Patient ID (UUID) if successful, None otherwise
        """
        if not self.is_enabled():
            return None

        try:
            data = {
                'patient_uid': patient_uid,
                'name': name,
                'age': age,
                'gender': gender,
                'scan_date': scan_date or datetime.utcnow().isoformat(),
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            # Upsert patient
            result = self.client.table('patients').upsert(
                data,
                on_conflict='patient_uid'
            ).execute()

            if result.data:
                return result.data[0]['id']
            return None

        except Exception as e:
            print(f"Error saving patient: {e}")
            return None

    def save_scan(self, patient_id: str, scan_metadata: Dict) -> Optional[str]:
        """
        Save scan metadata.

        Returns:
            Scan ID (UUID) if successful, None otherwise
        """
        if not self.is_enabled():
            return None

        try:
            data = {
                'patient_id': patient_id,
                'scan_type': scan_metadata.get('scan_type', 'CT_ABDOMEN'),
                'kvp': scan_metadata.get('kvp'),
                'mas': scan_metadata.get('mas'),
                'ctdivol': scan_metadata.get('ctdivol_mgy'),
                'num_slices': scan_metadata.get('num_slices'),
                'slice_thickness': scan_metadata.get('slice_thickness_mm'),
                'manufacturer': scan_metadata.get('manufacturer'),
                'study_description': scan_metadata.get('study_description'),
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            result = self.client.table('scans').insert(data).execute()

            if result.data:
                return result.data[0]['id']
            return None

        except Exception as e:
            print(f"Error saving scan: {e}")
            return None

    def save_organ_measurements(self, scan_id: str,
                                organ_measurements: Dict[str, Dict]) -> bool:
        """Save organ measurements for a scan."""
        if not self.is_enabled():
            return False

        try:
            records = []
            for organ_name, features in organ_measurements.items():
                record = {
                    'scan_id': scan_id,
                    'organ_name': organ_name,
                    'volume_cm3': features.get('volume_cm3'),
                    'mean_hu': features.get('mean_hu'),
                    'median_hu': features.get('median_hu'),
                    'std_hu': features.get('std_hu'),
                    'position_x': features.get('center_x_mm'),
                    'position_y': features.get('center_y_mm'),
                    'position_z': features.get('center_z_mm'),
                    'compactness': features.get('compactness'),
                }

                # Remove None values
                record = {k: v for k, v in record.items() if v is not None}
                records.append(record)

            if records:
                self.client.table('organ_measurements').insert(records).execute()

            return True

        except Exception as e:
            print(f"Error saving organ measurements: {e}")
            return False

    def save_organ_doses(self, scan_id: str, organ_doses: Dict[str, Dict],
                        model_name: str = 'ensemble') -> bool:
        """Save organ dose predictions."""
        if not self.is_enabled():
            return False

        try:
            records = []
            for organ_name, dose_info in organ_doses.items():
                record = {
                    'scan_id': scan_id,
                    'organ_name': organ_name,
                    'absorbed_dose_mgy': dose_info.get('dose_mgy'),
                    'effective_dose_msv': dose_info.get('effective_dose_msv'),
                    'tissue_weight': dose_info.get('tissue_weight'),
                    'model_used': model_name,
                    'confidence': dose_info.get('confidence', 0.90),
                }

                # Remove None values
                record = {k: v for k, v in record.items() if v is not None}
                records.append(record)

            if records:
                self.client.table('organ_doses').insert(records).execute()

            return True

        except Exception as e:
            print(f"Error saving organ doses: {e}")
            return False

    def save_patient_dose(self, scan_id: str, patient_dose: Dict,
                         processing_time: float = None) -> bool:
        """Save total patient dose summary."""
        if not self.is_enabled():
            return False

        try:
            data = {
                'scan_id': scan_id,
                'total_absorbed_dose_mgy': patient_dose.get('total_absorbed_dose_mgy'),
                'total_effective_dose_msv': patient_dose.get('total_effective_dose_msv'),
                'risk_category': patient_dose.get('risk_category'),
                'highest_dose_organ': patient_dose.get('highest_dose_organ'),
                'processing_time_seconds': processing_time,
            }

            # Remove None values
            data = {k: v for k, v in data.items() if v is not None}

            self.client.table('patient_doses').insert(data).execute()

            return True

        except Exception as e:
            print(f"Error saving patient dose: {e}")
            return False

    def save_complete_results(self, patient_uid: str, results: Dict) -> bool:
        """
        Save complete processing results to database.

        Args:
            patient_uid: Unique patient identifier
            results: Complete results dictionary from processing

        Returns:
            True if successful, False otherwise
        """
        if not self.is_enabled():
            print("Database not enabled. Skipping save.")
            return False

        try:
            # 1. Save patient
            patient_id = self.save_patient(
                patient_uid=patient_uid,
                name=results.get('patient_name')
            )

            if not patient_id:
                print("Failed to save patient")
                return False

            # 2. Save scan
            scan_id = self.save_scan(
                patient_id=patient_id,
                scan_metadata=results.get('scan_metadata', {})
            )

            if not scan_id:
                print("Failed to save scan")
                return False

            # 3. Save organ measurements
            if 'organ_measurements' in results:
                self.save_organ_measurements(
                    scan_id=scan_id,
                    organ_measurements=results['organ_measurements']
                )

            # 4. Save organ doses
            if 'organ_doses' in results:
                self.save_organ_doses(
                    scan_id=scan_id,
                    organ_doses=results['organ_doses']
                )

            # 5. Save patient dose summary
            if 'patient_dose' in results:
                self.save_patient_dose(
                    scan_id=scan_id,
                    patient_dose=results['patient_dose'],
                    processing_time=results.get('processing_time')
                )

            print(f"✓ Results saved to database (scan_id: {scan_id})")
            return True

        except Exception as e:
            print(f"Error saving complete results: {e}")
            return False

    def get_patient_history(self, patient_uid: str) -> List[Dict]:
        """Get all scans for a patient."""
        if not self.is_enabled():
            return []

        try:
            result = self.client.table('patients').select(
                '*, scans(*, patient_doses(*))'
            ).eq('patient_uid', patient_uid).execute()

            return result.data if result.data else []

        except Exception as e:
            print(f"Error fetching patient history: {e}")
            return []

    def get_scan_details(self, scan_id: str) -> Optional[Dict]:
        """Get complete details for a scan."""
        if not self.is_enabled():
            return None

        try:
            result = self.client.table('scans').select(
                '*, organ_measurements(*), organ_doses(*), patient_doses(*)'
            ).eq('id', scan_id).execute()

            return result.data[0] if result.data else None

        except Exception as e:
            print(f"Error fetching scan details: {e}")
            return None


# SQL schema for reference (create these tables in Supabase)
SCHEMA_SQL = """
-- Patients table
CREATE TABLE patients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_uid TEXT UNIQUE NOT NULL,
    name TEXT,
    age INTEGER,
    gender TEXT,
    scan_date TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Scans table
CREATE TABLE scans (
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
CREATE TABLE organ_measurements (
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
CREATE TABLE organ_doses (
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
CREATE TABLE patient_doses (
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
CREATE INDEX idx_patients_uid ON patients(patient_uid);
CREATE INDEX idx_scans_patient ON scans(patient_id);
CREATE INDEX idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX idx_patient_doses_scan ON patient_doses(scan_id);
"""


if __name__ == "__main__":
    # Test database connection
    print("Testing database connector...")
    print("\nTo enable database:")
    print("1. Create a Supabase project at https://supabase.com")
    print("2. Run the SQL schema above in Supabase SQL Editor")
    print("3. Set environment variables:")
    print("   export SUPABASE_URL='your-project-url'")
    print("   export SUPABASE_KEY='your-anon-key'")
    print("\nDatabase Schema SQL:")
    print(SCHEMA_SQL)

    # Try to connect
    db = DatabaseConnector()
    if db.is_enabled():
        print("\n✓ Database connection successful!")
    else:
        print("\n✗ Database not configured (optional)")
