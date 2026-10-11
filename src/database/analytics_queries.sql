-- ============================================================================
-- e-Pilot Analytical Query Suite (OLAP, Grouping, & SQL Window Functions)
-- Section 6.5: Normal vs. Simulated Emergency Cognitive & Behavioural Contrast
-- ============================================================================

-- ----------------------------------------------------------------------------
-- 1. Normal Cruise vs. Emergency Cognitive & Autonomic Stress Contrast
-- Aggregates Cognitive Load Index (CLI), EEG engagement, HRV, and GSR
-- across normal cruise operations vs. emergency upset scenarios.
-- ----------------------------------------------------------------------------
-- QUERY_NAME: normal_vs_emergency_contrast
SELECT 
    fp.phase_name,
    fp.is_emergency_simulated,
    COUNT(cs.snapshot_id) AS total_snapshots,
    ROUND(AVG(cs.cognitive_load_index), 4) AS mean_cognitive_load_index,
    ROUND(MIN(cs.cognitive_load_index), 4) AS min_cli,
    ROUND(MAX(cs.cognitive_load_index), 4) AS max_cli,
    ROUND(AVG(cs.raw_eeg_metric), 4) AS mean_eeg_engagement,
    ROUND(AVG(cs.raw_hrv_metric), 4) AS mean_hrv_stress,
    ROUND(AVG(cs.raw_gsr_metric), 4) AS mean_gsr_conductance,
    ROUND(AVG(cs.perclos_score), 4) AS mean_perclos_score,
    SUM(cs.is_alert_condition) AS total_alerts_triggered
FROM cognitive_state_snapshot cs
JOIN flight_phase fp ON cs.flight_phase_id = fp.phase_id
GROUP BY fp.phase_name, fp.is_emergency_simulated
ORDER BY fp.is_emergency_simulated DESC, mean_cognitive_load_index DESC;


-- ----------------------------------------------------------------------------
-- 2. Rolling 10-Window Moving Average & Volatility (SQL Window Functions)
-- Demonstrates Module 3/4 Window Functions for real-time trend smoothing.
-- ----------------------------------------------------------------------------
-- QUERY_NAME: rolling_cognitive_trend
SELECT 
    cs.snapshot_id,
    cs.flight_id,
    cs.pilot_id,
    cs.timestamp,
    cs.predicted_state,
    cs.cognitive_load_index,
    ROUND(AVG(cs.cognitive_load_index) OVER (
        PARTITION BY cs.flight_id, cs.pilot_id 
        ORDER BY cs.timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ), 4) AS rolling_avg_cli_10_windows,
    ROUND(MAX(cs.cognitive_load_index) OVER (
        PARTITION BY cs.flight_id, cs.pilot_id 
        ORDER BY cs.timestamp 
        ROWS BETWEEN 9 PRECEDING AND CURRENT ROW
    ), 4) AS rolling_peak_cli_10_windows,
    cs.is_alert_condition
FROM cognitive_state_snapshot cs
ORDER BY cs.flight_id, cs.timestamp ASC;


-- ----------------------------------------------------------------------------
-- 3. Safety Alert Taxonomy Distribution & Severity Breakdown
-- Analyzes frequency of Tunnel Vision, Distraction, and Startle alerts.
-- ----------------------------------------------------------------------------
-- QUERY_NAME: alert_severity_distribution
SELECT 
    a.severity_level,
    a.alert_category,
    a.declutter_level,
    a.advisory_audio,
    a.task_allocation,
    COUNT(a.alert_id) AS trigger_count,
    ROUND(AVG(cs.cognitive_load_index), 4) AS avg_trigger_cli,
    ROUND(AVG(cs.confidence_score), 4) AS avg_model_confidence
FROM alert a
JOIN cognitive_state_snapshot cs ON a.snapshot_id = cs.snapshot_id
GROUP BY a.severity_level, a.alert_category, a.declutter_level, a.advisory_audio, a.task_allocation
ORDER BY trigger_count DESC;


-- ----------------------------------------------------------------------------
-- 4. Dual-Pilot Crew Cross-Check & Cognitive Asymmetry Detection
-- Compares Captain vs First Officer workload during the same flight.
-- ----------------------------------------------------------------------------
-- QUERY_NAME: crew_cognitive_asymmetry
SELECT 
    f.flight_id,
    ca.crew_role,
    p.pilot_id,
    p.first_name || ' ' || p.last_name AS pilot_name,
    COUNT(cs.snapshot_id) AS snapshot_count,
    ROUND(AVG(cs.cognitive_load_index), 4) AS avg_crew_cli,
    ROUND(MAX(cs.cognitive_load_index), 4) AS peak_crew_cli,
    SUM(cs.is_alert_condition) AS total_crew_alerts
FROM flight f
JOIN crew_assignment ca ON f.flight_id = ca.flight_id
JOIN pilot p ON ca.pilot_id = p.pilot_id
LEFT JOIN cognitive_state_snapshot cs ON cs.flight_id = f.flight_id AND cs.pilot_id = p.pilot_id
GROUP BY f.flight_id, ca.crew_role, p.pilot_id, pilot_name
ORDER BY f.flight_id, ca.crew_role;


-- ----------------------------------------------------------------------------
-- 5. Regime Dwell Time & Percentage Allocation Analysis
-- Quantifies the percentage of flight duration spent in each cognitive state.
-- ----------------------------------------------------------------------------
-- QUERY_NAME: regime_dwell_time_percentage
WITH state_counts AS (
    SELECT 
        flight_id,
        predicted_state,
        COUNT(*) AS state_window_count,
        COUNT(*) * 1.0 AS state_seconds -- 1.0s stride between windows
    FROM cognitive_state_snapshot
    GROUP BY flight_id, predicted_state
),
flight_totals AS (
    SELECT 
        flight_id,
        SUM(state_window_count) AS total_windows
    FROM state_counts
    GROUP BY flight_id
)
SELECT 
    sc.flight_id,
    sc.predicted_state,
    sc.state_window_count,
    sc.state_seconds AS dwell_time_seconds,
    ROUND((sc.state_window_count * 100.0) / ft.total_windows, 2) AS percentage_of_flight
FROM state_counts sc
JOIN flight_totals ft ON sc.flight_id = ft.flight_id
ORDER BY sc.flight_id, percentage_of_flight DESC;
