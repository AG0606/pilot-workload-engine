-- ============================================================================
-- e-Pilot Database Schema (DDL)
-- A Database-Centric, IoT-Enabled Framework for Real-Time Multimodal Cognitive
-- State Assessment and Cockpit Display Decluttering
-- BCSE302L / BCSE302P Database Systems - BCNF Normalized Schema
-- ============================================================================

-- Enable foreign keys for SQLite
PRAGMA foreign_keys = ON;

-- ----------------------------------------------------------------------------
-- 1. Aircraft Table
-- Holds airframe specifications and operational airworthiness records.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS aircraft (
    tail_number VARCHAR(16) PRIMARY KEY,
    model VARCHAR(64) NOT NULL,
    manufacturer VARCHAR(64) NOT NULL,
    max_gross_weight_kg REAL NOT NULL CHECK (max_gross_weight_kg > 0),
    seating_capacity INTEGER NOT NULL CHECK (seating_capacity >= 1),
    certification_status VARCHAR(32) NOT NULL DEFAULT 'AIRWORTHY',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 2. Pilot Table
-- Flight crew demographic, licensing, and medical certification tracking.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS pilot (
    pilot_id VARCHAR(32) PRIMARY KEY,
    license_number VARCHAR(32) NOT NULL UNIQUE,
    first_name VARCHAR(64) NOT NULL,
    last_name VARCHAR(64) NOT NULL,
    total_flight_hours REAL NOT NULL DEFAULT 0.0 CHECK (total_flight_hours >= 0.0),
    medical_cert_expiry DATE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ----------------------------------------------------------------------------
-- 3. Flight Table
-- Represents individual commercial, freighter, or simulator flight sectors.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS flight (
    flight_id VARCHAR(64) PRIMARY KEY,
    flight_number VARCHAR(16) NOT NULL,
    aircraft_tail_number VARCHAR(16) NOT NULL,
    origin_icao VARCHAR(4) NOT NULL,
    destination_icao VARCHAR(4) NOT NULL,
    scheduled_departure TIMESTAMP NOT NULL,
    actual_departure TIMESTAMP,
    scheduled_arrival TIMESTAMP NOT NULL,
    actual_arrival TIMESTAMP,
    flight_status VARCHAR(32) NOT NULL DEFAULT 'SCHEDULED' 
        CHECK (flight_status IN ('SCHEDULED', 'ACTIVE', 'COMPLETED', 'DIVERTED', 'CANCELLED')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (aircraft_tail_number) REFERENCES aircraft(tail_number) ON DELETE RESTRICT ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 4. Crew Assignment Table
-- Normalizes M:N relationship between Flight and Pilot with crew roles.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS crew_assignment (
    assignment_id VARCHAR(64) PRIMARY KEY,
    flight_id VARCHAR(64) NOT NULL,
    pilot_id VARCHAR(32) NOT NULL,
    crew_role VARCHAR(32) NOT NULL 
        CHECK (crew_role IN ('PILOT_IN_COMMAND', 'FIRST_OFFICER', 'RELIEF_PILOT', 'OBSERVER')),
    seat_position VARCHAR(16) NOT NULL 
        CHECK (seat_position IN ('LEFT_SEAT', 'RIGHT_SEAT', 'OBSERVER_SEAT')),
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (pilot_id) REFERENCES pilot(pilot_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    UNIQUE(flight_id, seat_position),
    UNIQUE(flight_id, pilot_id)
);

-- ----------------------------------------------------------------------------
-- 5. Sensor Device Table
-- Hardware metadata factored out of readings to preserve 3NF / BCNF.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_device (
    device_id VARCHAR(32) PRIMARY KEY,
    device_serial VARCHAR(64) NOT NULL UNIQUE,
    device_type VARCHAR(32) NOT NULL 
        CHECK (device_type IN ('EEG_HEADSET', 'PPG_GSR_WRISTBAND', 'EYE_TRACKER', 'QAR_TELEMETRY', 'COCKPIT_CAMERA')),
    hardware_model VARCHAR(64) NOT NULL,
    firmware_version VARCHAR(32) NOT NULL,
    sampling_rate_hz REAL NOT NULL CHECK (sampling_rate_hz > 0),
    calibration_date DATE NOT NULL,
    is_active INTEGER NOT NULL DEFAULT 1 CHECK (is_active IN (0, 1))
);

-- ----------------------------------------------------------------------------
-- 6. Sensor Reading Table
-- High-frequency time-series measurements captured by avionics and IoT edge nodes.
-- Indexed heavily with composite B+ Tree for high-throughput range scans.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS sensor_reading (
    reading_id VARCHAR(64) PRIMARY KEY,
    flight_id VARCHAR(64) NOT NULL,
    device_id VARCHAR(32) NOT NULL,
    pilot_id VARCHAR(32) NOT NULL,
    timestamp REAL NOT NULL CHECK (timestamp >= 0.0),
    channel_name VARCHAR(32) NOT NULL,
    raw_value REAL NOT NULL,
    unit VARCHAR(16) NOT NULL,
    quality_index REAL NOT NULL DEFAULT 1.0 CHECK (quality_index >= 0.0 AND quality_index <= 1.0),
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (device_id) REFERENCES sensor_device(device_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    FOREIGN KEY (pilot_id) REFERENCES pilot(pilot_id) ON DELETE RESTRICT ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 7. Flight Phase Table
-- Tracks operational flight stages and emergency event simulation states.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS flight_phase (
    phase_id VARCHAR(64) PRIMARY KEY,
    flight_id VARCHAR(64) NOT NULL,
    phase_name VARCHAR(32) NOT NULL 
        CHECK (phase_name IN ('TAXI_GROUND', 'CLIMB', 'CRUISE', 'DESCENT', 'EMERGENCY_UPSET')),
    start_timestamp REAL NOT NULL CHECK (start_timestamp >= 0.0),
    end_timestamp REAL CHECK (end_timestamp IS NULL OR end_timestamp >= start_timestamp),
    is_emergency_simulated INTEGER NOT NULL DEFAULT 0 CHECK (is_emergency_simulated IN (0, 1)),
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 8. Emergency Event Table
-- In-flight abnormal regimes, aerodynamic upsets, or simulated injects.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS emergency_event (
    event_id VARCHAR(64) PRIMARY KEY,
    flight_id VARCHAR(64) NOT NULL,
    phase_id VARCHAR(64) NOT NULL,
    event_type VARCHAR(64) NOT NULL,
    onset_timestamp REAL NOT NULL CHECK (onset_timestamp >= 0.0),
    resolution_timestamp REAL CHECK (resolution_timestamp IS NULL OR resolution_timestamp >= onset_timestamp),
    severity_level VARCHAR(16) NOT NULL CHECK (severity_level IN ('LOW', 'MODERATE', 'SEVERE', 'CATASTROPHIC')),
    trigger_cause VARCHAR(128) NOT NULL,
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (phase_id) REFERENCES flight_phase(phase_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 9. Cognitive State Snapshot Table
-- Materialized analytical snapshots derived periodically by ML inference & Kalman fusion.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cognitive_state_snapshot (
    snapshot_id VARCHAR(64) PRIMARY KEY,
    flight_id VARCHAR(64) NOT NULL,
    pilot_id VARCHAR(32) NOT NULL,
    timestamp REAL NOT NULL CHECK (timestamp >= 0.0),
    flight_phase_id VARCHAR(64),
    predicted_state VARCHAR(32) NOT NULL 
        CHECK (predicted_state IN ('BASELINE', 'CHANNELIZED_ATTENTION', 'DIVERTED_ATTENTION', 'STARTLE')),
    confidence_score REAL NOT NULL CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
    cognitive_load_index REAL NOT NULL CHECK (cognitive_load_index >= 0.0 AND cognitive_load_index <= 1.0),
    raw_eeg_metric REAL NOT NULL,
    raw_hrv_metric REAL NOT NULL,
    raw_gsr_metric REAL NOT NULL,
    perclos_score REAL NOT NULL CHECK (perclos_score >= 0.0 AND perclos_score <= 1.0),
    is_alert_condition INTEGER NOT NULL DEFAULT 0 CHECK (is_alert_condition IN (0, 1)),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (pilot_id) REFERENCES pilot(pilot_id) ON DELETE RESTRICT ON UPDATE CASCADE,
    FOREIGN KEY (flight_phase_id) REFERENCES flight_phase(phase_id) ON DELETE SET NULL ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 10. Alert Table
-- Safety-critical cockpit alerts committed via atomic ACID transactions.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alert (
    alert_id VARCHAR(64) PRIMARY KEY,
    snapshot_id VARCHAR(64) NOT NULL UNIQUE,
    flight_id VARCHAR(64) NOT NULL,
    pilot_id VARCHAR(32) NOT NULL,
    timestamp REAL NOT NULL CHECK (timestamp >= 0.0),
    severity_level VARCHAR(16) NOT NULL CHECK (severity_level IN ('LOW', 'MEDIUM', 'CRITICAL')),
    alert_category VARCHAR(32) NOT NULL 
        CHECK (alert_category IN ('TUNNEL_VISION', 'DISTRACTION', 'ACUTE_STARTLE', 'MICROSLEEP', 'HIGH_G_UPSET')),
    declutter_level VARCHAR(32) NOT NULL 
        CHECK (declutter_level IN ('LEVEL_0_FULL', 'LEVEL_1_MODERATE', 'LEVEL_2_ESSENTIALS')),
    advisory_audio VARCHAR(32) NOT NULL 
        CHECK (advisory_audio IN ('NONE', 'CROSS_CHECK_CHIME', 'HEADS_UP_CHIME', 'ATTITUDE_RECOVERY_WARNING')),
    task_allocation VARCHAR(32) NOT NULL 
        CHECK (task_allocation IN ('PILOT_FLYING', 'SHARED_COCKPIT', 'AUTOPILOT_AUTO_ASSIST')),
    acknowledged INTEGER NOT NULL DEFAULT 0 CHECK (acknowledged IN (0, 1)),
    acknowledged_at TIMESTAMP,
    FOREIGN KEY (snapshot_id) REFERENCES cognitive_state_snapshot(snapshot_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (flight_id) REFERENCES flight(flight_id) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (pilot_id) REFERENCES pilot(pilot_id) ON DELETE RESTRICT ON UPDATE CASCADE
);

-- ----------------------------------------------------------------------------
-- 11. Alert Audit Log Table
-- Immutable audit log for post-flight safety debriefing and FAA regulatory compliance.
-- ----------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS alert_audit_log (
    audit_id VARCHAR(64) PRIMARY KEY,
    alert_id VARCHAR(64) NOT NULL,
    action_taken VARCHAR(64) NOT NULL,
    operator_notes TEXT,
    logged_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (alert_id) REFERENCES alert(alert_id) ON DELETE CASCADE ON UPDATE CASCADE
);

-- ============================================================================
-- INDEXING SPECIFICATION: Physical Database Design & B+ Tree Optimizations
-- ============================================================================

-- 1. Dominant Query Pattern: Composite B+ Tree Index on (flight_id, timestamp)
-- Enables logarithmic search and sequential page traversal for time-series window scans.
CREATE INDEX IF NOT EXISTS idx_sensor_reading_flight_time 
    ON sensor_reading(flight_id, timestamp);

-- 2. Materialized Snapshot Time-Series Range Scans
CREATE INDEX IF NOT EXISTS idx_snapshot_flight_time 
    ON cognitive_state_snapshot(flight_id, timestamp);

-- 3. High-Speed Filter on Cognitive State & Critical Alerts
CREATE INDEX IF NOT EXISTS idx_snapshot_flight_state 
    ON cognitive_state_snapshot(flight_id, predicted_state);

CREATE INDEX IF NOT EXISTS idx_snapshot_alert_flag 
    ON cognitive_state_snapshot(is_alert_condition, timestamp);

-- 4. Alert Retrieval by Flight and Severity
CREATE INDEX IF NOT EXISTS idx_alert_flight_severity 
    ON alert(flight_id, severity_level, timestamp);

-- 5. Operational Flight Phase Time Indexing
CREATE INDEX IF NOT EXISTS idx_phase_flight_time 
    ON flight_phase(flight_id, start_timestamp, end_timestamp);

CREATE INDEX IF NOT EXISTS idx_phase_flight_emergency 
    ON flight_phase(flight_id, is_emergency_simulated);
