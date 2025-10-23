-- ============================================
-- Medical Portal System - Seed Data
-- Development and Testing Data
-- ============================================

-- Clear existing data (for development only!)
-- TRUNCATE TABLE audit_logs, notifications, doctor_reviews, model_metrics, 
-- ai_analysis, report_images, medical_reports, patient_medical_history, 
-- ai_models, sessions, users CASCADE;

-- ============================================
-- USERS
-- ============================================

-- Insert sample users
-- Note: In production, use proper password hashing (bcrypt with 10+ rounds)
-- Password for all users: "password123" (hashed)

INSERT INTO users (id, name, email, password_hash, role, phone, date_of_birth, is_active, is_email_verified, preferences) VALUES
-- Patients
('11111111-1111-1111-1111-111111111111', 'John Smith', 'patient@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'patient', '+1-555-0101', '1985-03-15', true, true, '{"theme": "light", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),
('11111111-1111-1111-1111-111111111112', 'Emma Johnson', 'emma.j@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'patient', '+1-555-0102', '1990-07-22', true, true, '{"theme": "dark", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),
('11111111-1111-1111-1111-111111111113', 'Michael Brown', 'michael.b@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'patient', '+1-555-0103', '1978-11-30', true, true, '{"theme": "system", "notifications_enabled": true, "email_notifications": false, "language": "en"}'::jsonb),

-- Radiologists
('22222222-2222-2222-2222-222222222221', 'Dr. Sarah Chen', 'radiologist@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'radiologist', '+1-555-0201', '1982-05-10', true, true, '{"theme": "dark", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),
('22222222-2222-2222-2222-222222222222', 'Dr. James Wilson', 'james.w@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'radiologist', '+1-555-0202', '1975-09-18', true, true, '{"theme": "light", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),

-- Doctors
('33333333-3333-3333-3333-333333333331', 'Dr. Michael Johnson', 'doctor@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'doctor', '+1-555-0301', '1980-02-25', true, true, '{"theme": "light", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),
('33333333-3333-3333-3333-333333333332', 'Dr. Lisa Martinez', 'lisa.m@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'doctor', '+1-555-0302', '1987-12-08', true, true, '{"theme": "dark", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),

-- Tech Team
('44444444-4444-4444-4444-444444444441', 'Alex Rodriguez', 'tech@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'tech', '+1-555-0401', '1992-06-14', true, true, '{"theme": "dark", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb),
('44444444-4444-4444-4444-444444444442', 'Sophia Lee', 'sophia.l@example.com', '$2a$10$rBV2uGVHY5ZPJfKfaKDDPOXKJJi9tV5X.xMKPqg7z7Z5zQGqGqGqG', 'tech', '+1-555-0402', '1989-04-20', true, true, '{"theme": "system", "notifications_enabled": true, "email_notifications": true, "language": "en"}'::jsonb);

-- ============================================
-- PATIENT MEDICAL HISTORY
-- ============================================

INSERT INTO patient_medical_history (history_id, patient_id, conditions, allergies, medications, family_history, smoking_status, alcohol_consumption, exercise_frequency) VALUES
('aaaa1111-1111-1111-1111-111111111111', '11111111-1111-1111-1111-111111111111', 
    '["Hypertension", "Seasonal Allergies"]'::jsonb,
    '["Penicillin", "Pollen"]'::jsonb,
    '["Lisinopril 10mg daily", "Loratadine 10mg as needed"]'::jsonb,
    '{"heart_disease": "Father", "diabetes": "Mother"}'::jsonb,
    'Never', 'Occasional', '3-4 times per week'),

('aaaa1111-1111-1111-1111-111111111112', '11111111-1111-1111-1111-111111111112',
    '["Type 2 Diabetes"]'::jsonb,
    '["Latex"]'::jsonb,
    '["Metformin 500mg twice daily", "Atorvastatin 20mg daily"]'::jsonb,
    '{"diabetes": "Both parents", "hypertension": "Father"}'::jsonb,
    'Former (quit 5 years ago)', 'Never', '2-3 times per week'),

('aaaa1111-1111-1111-1111-111111111113', '11111111-1111-1111-1111-111111111113',
    '[]'::jsonb,
    '["Shellfish"]'::jsonb,
    '[]'::jsonb,
    '{}'::jsonb,
    'Never', 'Social', 'Daily');

-- ============================================
-- AI MODELS
-- ============================================

INSERT INTO ai_models (model_id, model_name, version, architecture, framework, model_file_url, is_active, trained_date, deployed_date, baseline_accuracy, baseline_precision, baseline_recall, baseline_f1_score, description) VALUES
('55555555-5555-5555-5555-555555555551', 'XRay Risk Classifier', 'v1.0.0', 'CNN', 'TensorFlow', 's3://models/xray-classifier-v1.0.0.h5', true, '2024-09-15 10:00:00+00', '2024-10-01 08:00:00+00', 0.9250, 0.8920, 0.9410, 0.9160, 'Primary X-ray risk assessment model using 4-layer CNN architecture'),
('55555555-5555-5555-5555-555555555552', 'XRay Risk Classifier', 'v0.9.5', 'CNN', 'TensorFlow', 's3://models/xray-classifier-v0.9.5.h5', false, '2024-08-01 10:00:00+00', '2024-08-15 08:00:00+00', 0.9100, 0.8750, 0.9200, 0.8970, 'Previous version - deprecated');

-- ============================================
-- MEDICAL REPORTS
-- ============================================

INSERT INTO medical_reports (report_id, patient_id, radiologist_id, patient_name, radiologist_name, study_type, report_image_url, risk_score, confidence_score, findings, recommendations, upload_date, analysis_date, status, patient_age, patient_gender) VALUES
-- High risk case
('66666666-6666-6666-6666-666666666661', '11111111-1111-1111-1111-111111111111', '22222222-2222-2222-2222-222222222221', 'John Smith', 'Dr. Sarah Chen', 'Chest X-Ray', 'https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=512', 'high', 0.9200, 'Suspicious opacity detected in right upper lobe. Irregular borders suggest possible malignancy. Recommend immediate follow-up.', '["Immediate consultation with pulmonologist", "CT scan within 48 hours", "Consider bronchoscopy", "Monitor for respiratory symptoms"]'::jsonb, NOW() - INTERVAL '2 days', NOW() - INTERVAL '2 days', 'analyzed', 39, 'Male'),

-- Medium risk case
('66666666-6666-6666-6666-666666666662', '11111111-1111-1111-1111-111111111112', '22222222-2222-2222-2222-222222222221', 'Emma Johnson', 'Dr. Sarah Chen', 'Chest X-Ray', 'https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=512', 'medium', 0.6500, 'Mild inflammation detected in lower left lobe. Possible early-stage infection or inflammation.', '["Follow-up in 2-4 weeks", "Monitor symptoms", "Complete course of antibiotics if prescribed", "Maintain healthy lifestyle"]'::jsonb, NOW() - INTERVAL '5 days', NOW() - INTERVAL '5 days', 'analyzed', 34, 'Female'),

-- Low risk case
('66666666-6666-6666-6666-666666666663', '11111111-1111-1111-1111-111111111113', '22222222-2222-2222-2222-222222222222', 'Michael Brown', 'Dr. James Wilson', 'Chest X-Ray', 'https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=512', 'low', 0.9500, 'No significant abnormalities detected. Lung fields are clear. Heart size is normal.', '["Continue routine medical check-ups", "Maintain healthy lifestyle", "No immediate action required", "Schedule next routine scan as advised"]'::jsonb, NOW() - INTERVAL '7 days', NOW() - INTERVAL '7 days', 'reviewed', 46, 'Male'),

-- Pending analysis
('66666666-6666-6666-6666-666666666664', '11111111-1111-1111-1111-111111111111', '22222222-2222-2222-2222-222222222222', 'John Smith', 'Dr. James Wilson', 'Chest X-Ray', 'https://images.unsplash.com/photo-1631217868264-e5b90bb7e133?w=512', NULL, NULL, NULL, NULL, NOW() - INTERVAL '30 minutes', NULL, 'pending', 39, 'Male');

-- ============================================
-- AI ANALYSIS
-- ============================================

INSERT INTO ai_analysis (analysis_id, report_id, model_id, risk_category, risk_probability, low_probability, medium_probability, high_probability, findings, processing_time_ms, processed_at, image_quality_score, model_version) VALUES
-- Analysis for high risk case
('77777777-7777-7777-7777-777777777771', '66666666-6666-6666-6666-666666666661', '55555555-5555-5555-5555-555555555551', 'high', 0.9200, 0.0300, 0.0500, 0.9200, 
'{"abnormalities": [{"type": "opacity", "location": "right upper lobe", "severity": "high", "confidence": 0.92, "coordinates": {"x": 245, "y": 180}}], "features": {"edge_sharpness": 0.78, "contrast_level": 0.85, "texture_uniformity": 0.45}}'::jsonb, 
8500, NOW() - INTERVAL '2 days', 0.8900, 'v1.0.0'),

-- Analysis for medium risk case
('77777777-7777-7777-7777-777777777772', '66666666-6666-6666-6666-666666666662', '55555555-5555-5555-5555-555555555551', 'medium', 0.6500, 0.2000, 0.6500, 0.1500,
'{"abnormalities": [{"type": "inflammation", "location": "lower left lobe", "severity": "moderate", "confidence": 0.65, "coordinates": {"x": 180, "y": 320}}], "features": {"edge_sharpness": 0.82, "contrast_level": 0.75, "texture_uniformity": 0.68}}'::jsonb,
7800, NOW() - INTERVAL '5 days', 0.9200, 'v1.0.0'),

-- Analysis for low risk case
('77777777-7777-7777-7777-777777777773', '66666666-6666-6666-6666-666666666663', '55555555-5555-5555-5555-555555555551', 'low', 0.9500, 0.9500, 0.0300, 0.0200,
'{"abnormalities": [], "features": {"edge_sharpness": 0.88, "contrast_level": 0.92, "texture_uniformity": 0.85}, "notes": "Clear lung fields, no abnormalities detected"}'::jsonb,
6200, NOW() - INTERVAL '7 days', 0.9500, 'v1.0.0');

-- ============================================
-- DOCTOR REVIEWS
-- ============================================

INSERT INTO doctor_reviews (review_id, report_id, doctor_id, review_notes, clinical_findings, confirmed_ai_diagnosis, ai_accuracy_rating, doctor_risk_assessment, urgency_level, recommended_action, follow_up_required, follow_up_date, reviewed_at) VALUES
-- Review for low risk case (completed)
('88888888-8888-8888-8888-888888888881', '66666666-6666-6666-6666-666666666663', '33333333-3333-3333-3333-333333333331', 
'Reviewed X-ray images. Agree with AI assessment. No significant abnormalities detected. Lungs are clear, heart size is within normal limits. Patient can continue with routine care.',
'No acute findings. Routine follow-up in 12 months unless symptoms develop.',
true, 5, 'low', 'routine', 
'Continue routine annual check-ups. No additional testing required at this time. Maintain healthy lifestyle with regular exercise.',
false, NULL, NOW() - INTERVAL '6 days'),

-- Review for high risk case (urgent - in progress)
('88888888-8888-8888-8888-888888888882', '66666666-6666-6666-6666-666666666661', '33333333-3333-3333-3333-333333333332',
'URGENT: Confirmed suspicious mass in right upper lobe. Irregular borders and high density concerning for malignancy. Patient has been contacted and scheduled for CT scan tomorrow. Pulmonology consult requested.',
'3.5cm irregular opacity in RUL. Spiculated margins. No previous imaging available for comparison. Patient reports mild cough for 3 weeks, no hemoptysis.',
true, 5, 'high', 'urgent',
'1. Urgent CT chest with contrast within 24 hours\n2. Pulmonology consultation scheduled for 2 days\n3. Consider PET scan based on CT results\n4. Biopsy likely required\n5. Patient counseling regarding findings',
true, CURRENT_DATE + INTERVAL '2 days', NOW() - INTERVAL '1 day');

-- ============================================
-- NOTIFICATIONS
-- ============================================

INSERT INTO notifications (notification_id, user_id, type, priority, title, message, action_url, is_read, related_report_id, sent_via_email, created_at) VALUES
-- High priority notification for patient
('99999999-9999-9999-9999-999999999991', '11111111-1111-1111-1111-111111111111', 'urgent_case', 'urgent', 
'URGENT: Your X-Ray Results Require Immediate Attention',
'Your recent chest X-ray has been reviewed by Dr. Lisa Martinez and requires immediate follow-up. Please contact your healthcare provider within 24 hours.',
'/reports/66666666-6666-6666-6666-666666666661', false, '66666666-6666-6666-6666-666666666661', true, NOW() - INTERVAL '1 day'),

-- Doctor notification for high-risk case
('99999999-9999-9999-9999-999999999992', '33333333-3333-3333-3333-333333333332', 'new_report', 'high',
'High Risk Case Requires Review',
'A high-risk chest X-ray (Patient: John Smith) has been flagged by the AI system and requires your immediate review.',
'/reports/66666666-6666-6666-6666-666666666661', true, '66666666-6666-6666-6666-666666666661', true, NOW() - INTERVAL '2 days'),

-- Normal notification for patient
('99999999-9999-9999-9999-999999999993', '11111111-1111-1111-1111-111111111112', 'new_report', 'normal',
'Your X-Ray Results Are Ready',
'Your chest X-ray results are now available. Your doctor will contact you if any follow-up is needed.',
'/reports/66666666-6666-6666-6666-666666666662', true, '66666666-6666-6666-6666-666666666662', true, NOW() - INTERVAL '5 days'),

-- Completed review notification
('99999999-9999-9999-9999-999999999994', '11111111-1111-1111-1111-111111111113', 'review_complete', 'normal',
'Doctor Review Complete',
'Dr. Michael Johnson has completed the review of your chest X-ray. No significant abnormalities detected. Continue routine care.',
'/reports/66666666-6666-6666-6666-666666666663', true, '66666666-6666-6666-6666-666666666663', true, NOW() - INTERVAL '6 days');

-- ============================================
-- MODEL METRICS
-- ============================================

INSERT INTO model_metrics (metric_id, model_id, accuracy, precision, recall, f1_score, true_positives, true_negatives, false_positives, false_negatives, total_predictions, correct_predictions, recorded_at, period_start, period_end, dataset_version, test_set_size) VALUES
-- Initial baseline metrics
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', '55555555-5555-5555-5555-555555555551', 0.9250, 0.8920, 0.9410, 0.9160, 850, 8900, 100, 150, 10000, 9750, '2024-10-01 00:00:00+00', '2024-09-15 00:00:00+00', '2024-09-30 23:59:59+00', 'v1.0', 10000),

-- Recent performance metrics (last 7 days)
('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaab', '55555555-5555-5555-5555-555555555551', 0.9280, 0.8950, 0.9430, 0.9180, 92, 895, 8, 5, 1000, 987, NOW(), NOW() - INTERVAL '7 days', NOW(), 'v1.0', 1000);

-- ============================================
-- AUDIT LOGS (Sample entries)
-- ============================================

INSERT INTO audit_logs (log_id, user_id, user_email, user_role, action, entity_type, entity_id, ip_address, user_agent, status, created_at) VALUES
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '11111111-1111-1111-1111-111111111111', 'patient@example.com', 'patient', 'login', 'user', '11111111-1111-1111-1111-111111111111', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'success', NOW() - INTERVAL '1 hour'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbc', '22222222-2222-2222-2222-222222222221', 'radiologist@example.com', 'radiologist', 'upload_report', 'report', '66666666-6666-6666-6666-666666666661', '192.168.1.101', 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)', 'success', NOW() - INTERVAL '2 days'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbd', '33333333-3333-3333-3333-333333333331', 'doctor@example.com', 'doctor', 'create_review', 'review', '88888888-8888-8888-8888-888888888881', '192.168.1.102', 'Mozilla/5.0 (iPhone; CPU iPhone OS 14_6 like Mac OS X)', 'success', NOW() - INTERVAL '6 days'),
('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', '11111111-1111-1111-1111-111111111111', 'patient@example.com', 'patient', 'view_report', 'report', '66666666-6666-6666-6666-666666666661', '192.168.1.100', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)', 'success', NOW() - INTERVAL '1 day');

-- ============================================
-- SYSTEM SETTINGS
-- ============================================

INSERT INTO system_settings (setting_id, setting_key, setting_value, description, is_sensitive) VALUES
('cccccccc-cccc-cccc-cccc-cccccccccccc', 'ai_model_active_version', '"v1.0.0"'::jsonb, 'Currently active AI model version', false),
('cccccccc-cccc-cccc-cccc-cccccccccccd', 'max_upload_size_mb', '50'::jsonb, 'Maximum file upload size in megabytes', false),
('cccccccc-cccc-cccc-cccc-cccccccccccf', 'session_timeout_hours', '24'::jsonb, 'User session timeout in hours', false),
('cccccccc-cccc-cccc-cccc-ccccccccccce', 'notification_retention_days', '90'::jsonb, 'Number of days to retain read notifications', false),
('cccccccc-cccc-cccc-cccc-ccccccccccg', 'risk_thresholds', '{"low": 0.4, "medium": 0.7, "high": 1.0}'::jsonb, 'AI risk score thresholds', false);

-- ============================================
-- SUMMARY
-- ============================================

-- Display summary of seeded data
DO $$
BEGIN
    RAISE NOTICE 'Seed data loaded successfully!';
    RAISE NOTICE 'Users: %', (SELECT COUNT(*) FROM users);
    RAISE NOTICE 'Medical Reports: %', (SELECT COUNT(*) FROM medical_reports);
    RAISE NOTICE 'AI Analysis: %', (SELECT COUNT(*) FROM ai_analysis);
    RAISE NOTICE 'Doctor Reviews: %', (SELECT COUNT(*) FROM doctor_reviews);
    RAISE NOTICE 'Notifications: %', (SELECT COUNT(*) FROM notifications);
    RAISE NOTICE 'AI Models: %', (SELECT COUNT(*) FROM ai_models);
END $$;

-- ============================================
-- TEST QUERIES
-- ============================================

-- View all patients
-- SELECT id, name, email, role FROM users WHERE role = 'patient';

-- View all reports with risk scores
-- SELECT report_id, patient_name, study_type, risk_score, confidence_score, status FROM medical_reports;

-- View doctor priority queue
-- SELECT * FROM v_doctor_priority_queue LIMIT 5;

-- View patient reports
-- SELECT * FROM v_patient_reports WHERE patient_id = '11111111-1111-1111-1111-111111111111';