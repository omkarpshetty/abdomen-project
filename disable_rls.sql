-- Disable Row Level Security (RLS) for AI Organ Dose System tables
-- Run this in Supabase SQL Editor

-- Disable RLS on all tables to allow writes
ALTER TABLE patients DISABLE ROW LEVEL SECURITY;
ALTER TABLE scans DISABLE ROW LEVEL SECURITY;
ALTER TABLE organ_measurements DISABLE ROW LEVEL SECURITY;
ALTER TABLE organ_doses DISABLE ROW LEVEL SECURITY;
ALTER TABLE patient_doses DISABLE ROW LEVEL SECURITY;
