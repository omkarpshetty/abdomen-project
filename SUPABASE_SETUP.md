# Supabase Database Setup Guide

## Step 1: Get Your Credentials

1. Go to your Supabase project dashboard: https://supabase.com/dashboard/project/dogywbhaeyuqawmhoreq
2. Click on **Settings** (gear icon) in the left sidebar
3. Click on **API** under Project Settings
4. Copy these two values:
   - **Project URL** (looks like: `https://dogywbhaeyuqawmhoreq.supabase.co`)
   - **anon/public key** (long string starting with `eyJ...`)

## Step 2: Create Database Tables

1. In your Supabase dashboard, click **SQL Editor** in the left sidebar
2. Click **New Query**
3. Copy and paste the SQL below, then click **Run**

```sql
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

-- Create indexes for better query performance
CREATE INDEX idx_patients_uid ON patients(patient_uid);
CREATE INDEX idx_scans_patient ON scans(patient_id);
CREATE INDEX idx_organ_measurements_scan ON organ_measurements(scan_id);
CREATE INDEX idx_organ_doses_scan ON organ_doses(scan_id);
CREATE INDEX idx_patient_doses_scan ON patient_doses(scan_id);
```

## Step 3: Set Environment Variables

### Option A: Using .env file (Recommended)

Create a file named `.env` in your project root:

```bash
SUPABASE_URL=https://dogywbhaeyuqawmhoreq.supabase.co
SUPABASE_KEY=your-anon-key-here
```

Then install python-dotenv:
```bash
pip install python-dotenv
```

### Option B: Command line arguments

Use the `--save-to-db` flag with your credentials:

```bash
python ai_dose_system.py --dicom-dir "C:\Users\omkar\OneDrive\Desktop\data\55 plain" --save-to-db --supabase-url "https://dogywbhaeyuqawmhoreq.supabase.co" --supabase-key "your-anon-key"
```

### Option C: Set system environment variables (Windows)

```bash
setx SUPABASE_URL "https://dogywbhaeyuqawmhoreq.supabase.co"
setx SUPABASE_KEY "your-anon-key-here"
```

Then restart your terminal.

## Step 4: Install Supabase Python Client

```bash
pip install supabase
```

## Step 5: Test Connection

Test that everything works:

```bash
python src/database_connector.py
```

## Step 6: Run Your System with Database

Now run the complete system with database enabled:

```bash
python test_complete_system.py
```

To enable database in the test, edit `test_complete_system.py` and change:
```python
enable_database=False
```
to:
```python
enable_database=True
```

Or create a new test file that uses the database.

## Viewing Your Data

Go to your Supabase dashboard and click **Table Editor** to see:
- patients
- scans
- organ_measurements
- organ_doses
- patient_doses

All your CT scan results will be stored there!
