# Database Quick Reference Guide

Quick reference for common database operations in the Medical Portal System.

## 📋 Table of Contents

- [Common Queries](#common-queries)
- [Insert Operations](#insert-operations)
- [Update Operations](#update-operations)
- [Delete Operations](#delete-operations)
- [Views](#views)
- [Performance Tips](#performance-tips)

---

## Common Queries

### User Management

#### Get all users by role
```sql
-- Get all patients
SELECT id, name, email, phone FROM users WHERE role = 'patient';

-- Get all radiologists
SELECT id, name, email, phone FROM users WHERE role = 'radiologist';

-- Get all doctors
SELECT id, name, email, phone FROM users WHERE role = 'doctor';

-- Get all tech team members
SELECT id, name, email, phone FROM users WHERE role = 'tech';
```

#### Find user by email
```sql
SELECT * FROM users WHERE email = 'patient@example.com';
```

#### Get user's active sessions
```sql
SELECT 
    s.session_id,
    s.login_time,
    s.last_activity,
    s.ip_address
FROM sessions s
WHERE s.user_id = 'user-uuid-here'
  AND s.is_active = true
  AND s.expires_at > CURRENT_TIMESTAMP;
```

---

### Medical Reports

#### Get all reports for a patient
```sql
SELECT 
    report_id,
    study_type,
    risk_score,
    confidence_score,
    upload_date,
    analysis_date,
    status
FROM medical_reports
WHERE patient_id = 'patient-uuid-here'
ORDER BY upload_date DESC;
```

#### Get reports by risk score
```sql
-- High risk reports
SELECT 
    r.report_id,
    r.patient_name,
    r.study_type,
    r.risk_score,
    r.confidence_score,
    r.upload_date
FROM medical_reports r
WHERE r.risk_score = 'high'
  AND r.status = 'analyzed'
ORDER BY r.upload_date DESC;
```

#### Get reports pending review
```sql
SELECT 
    r.report_id,
    r.patient_name,
    r.study_type,
    r.risk_score,
    r.analysis_date
FROM medical_reports r
WHERE r.status = 'analyzed'
  AND NOT EXISTS (
      SELECT 1 FROM doctor_reviews dr 
      WHERE dr.report_id = r.report_id
  )
ORDER BY 
    CASE r.risk_score
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'low' THEN 3
    END,
    r.analysis_date ASC;
```

---

### AI Analysis

#### Get AI analysis for a report
```sql
SELECT 
    aa.analysis_id,
    aa.risk_category,
    aa.risk_probability,
    aa.low_probability,
    aa.medium_probability,
    aa.high_probability,
    aa.findings,
    aa.processed_at,
    am.model_name,
    am.version
FROM ai_analysis aa
JOIN ai_models am ON aa.model_id = am.model_id
WHERE aa.report_id = 'report-uuid-here';
```

#### Get AI model performance
```sql
SELECT 
    mm.accuracy,
    mm.precision,
    mm.recall,
    mm.f1_score,
    mm.total_predictions,
    mm.correct_predictions,
    mm.recorded_at
FROM model_metrics mm
WHERE mm.model_id = 'model-uuid-here'
ORDER BY mm.recorded_at DESC
LIMIT 10;
```

---

### Doctor Reviews

#### Get all reviews by a doctor
```sql
SELECT 
    dr.review_id,
    dr.report_id,
    r.patient_name,
    dr.urgency_level,
    dr.confirmed_ai_diagnosis,
    dr.reviewed_at
FROM doctor_reviews dr
JOIN medical_reports r ON dr.report_id = r.report_id
WHERE dr.doctor_id = 'doctor-uuid-here'
ORDER BY dr.reviewed_at DESC;
```

#### Get urgent cases
```sql
SELECT 
    dr.review_id,
    r.patient_name,
    r.study_type,
    dr.urgency_level,
    dr.recommended_action,
    dr.follow_up_date
FROM doctor_reviews dr
JOIN medical_reports r ON dr.report_id = r.report_id
WHERE dr.urgency_level IN ('urgent', 'critical')
  AND (dr.follow_up_date IS NULL OR dr.follow_up_date >= CURRENT_DATE)
ORDER BY 
    CASE dr.urgency_level
        WHEN 'critical' THEN 1
        WHEN 'urgent' THEN 2
    END,
    dr.reviewed_at DESC;
```

---

### Notifications

#### Get unread notifications for a user
```sql
SELECT 
    notification_id,
    type,
    priority,
    title,
    message,
    created_at
FROM notifications
WHERE user_id = 'user-uuid-here'
  AND is_read = false
ORDER BY 
    CASE priority
        WHEN 'urgent' THEN 1
        WHEN 'high' THEN 2
        WHEN 'normal' THEN 3
        WHEN 'low' THEN 4
    END,
    created_at DESC;
```

#### Mark notification as read
```sql
UPDATE notifications
SET is_read = true,
    read_at = CURRENT_TIMESTAMP
WHERE notification_id = 'notification-uuid-here';
```

---

## Insert Operations

### Create a new user
```sql
INSERT INTO users (
    name, 
    email, 
    password_hash, 
    role, 
    phone
) VALUES (
    'Jane Doe',
    'jane.doe@example.com',
    '$2a$10$hashedpassword', -- Use bcrypt to hash the password
    'patient',
    '+1-555-0123'
);
```

### Upload a new medical report
```sql
-- Step 1: Insert report
INSERT INTO medical_reports (
    patient_id,
    radiologist_id,
    patient_name,
    radiologist_name,
    study_type,
    report_image_url,
    upload_date,
    status
) VALUES (
    'patient-uuid-here',
    'radiologist-uuid-here',
    'Patient Name',
    'Dr. Radiologist Name',
    'Chest X-Ray',
    'https://example.com/xray.jpg',
    CURRENT_TIMESTAMP,
    'pending'
) RETURNING report_id;

-- Step 2: AI analysis (after processing)
INSERT INTO ai_analysis (
    report_id,
    model_id,
    risk_category,
    risk_probability,
    low_probability,
    medium_probability,
    high_probability,
    findings,
    processing_time_ms
) VALUES (
    'report-uuid-from-above',
    'active-model-uuid',
    'medium',
    0.65,
    0.20,
    0.65,
    0.15,
    '{"abnormalities": []}'::jsonb,
    7500
);

-- Step 3: Update report with AI results
UPDATE medical_reports
SET risk_score = 'medium',
    confidence_score = 0.65,
    findings = 'AI-generated findings text',
    recommendations = '["Recommendation 1", "Recommendation 2"]'::jsonb,
    analysis_date = CURRENT_TIMESTAMP,
    status = 'analyzed'
WHERE report_id = 'report-uuid-from-above';
```

### Create a doctor review
```sql
INSERT INTO doctor_reviews (
    report_id,
    doctor_id,
    review_notes,
    clinical_findings,
    confirmed_ai_diagnosis,
    ai_accuracy_rating,
    doctor_risk_assessment,
    urgency_level,
    recommended_action,
    follow_up_required,
    follow_up_date
) VALUES (
    'report-uuid-here',
    'doctor-uuid-here',
    'Detailed review notes...',
    'Clinical findings...',
    true,
    5,
    'medium',
    'follow_up',
    'Schedule follow-up in 2 weeks',
    true,
    CURRENT_DATE + INTERVAL '14 days'
);

-- Update report status
UPDATE medical_reports
SET status = 'reviewed'
WHERE report_id = 'report-uuid-here';
```

### Send a notification
```sql
INSERT INTO notifications (
    user_id,
    type,
    priority,
    title,
    message,
    related_report_id,
    sent_via_email
) VALUES (
    'patient-uuid-here',
    'review_complete',
    'normal',
    'Doctor Review Complete',
    'Your X-ray has been reviewed by Dr. Smith.',
    'report-uuid-here',
    true
);
```

---

## Update Operations

### Update user profile
```sql
UPDATE users
SET name = 'New Name',
    phone = '+1-555-9999',
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'user-uuid-here';
```

### Update user preferences
```sql
UPDATE users
SET preferences = jsonb_set(
    preferences,
    '{theme}',
    '"dark"'
)
WHERE id = 'user-uuid-here';
```

### Change report status
```sql
UPDATE medical_reports
SET status = 'reviewed',
    updated_at = CURRENT_TIMESTAMP
WHERE report_id = 'report-uuid-here';
```

---

## Delete Operations

### Soft delete (deactivate) user
```sql
UPDATE users
SET is_active = false,
    updated_at = CURRENT_TIMESTAMP
WHERE id = 'user-uuid-here';
```

### Delete old notifications (cleanup)
```sql
DELETE FROM notifications
WHERE is_read = true
  AND created_at < CURRENT_TIMESTAMP - INTERVAL '90 days';
```

### Archive old reports
```sql
UPDATE medical_reports
SET status = 'archived',
    updated_at = CURRENT_TIMESTAMP
WHERE upload_date < CURRENT_TIMESTAMP - INTERVAL '2 years'
  AND status = 'reviewed';
```

---

## Views

### Patient Reports View
```sql
-- Use pre-created view
SELECT * FROM v_patient_reports 
WHERE patient_id = 'patient-uuid-here';
```

### Doctor Priority Queue
```sql
-- Use pre-created view
SELECT * FROM v_doctor_priority_queue
LIMIT 20;
```

---

## Performance Tips

### Use indexes for filtering
```sql
-- ✅ Good - uses index on patient_id
SELECT * FROM medical_reports WHERE patient_id = 'uuid';

-- ❌ Bad - full table scan
SELECT * FROM medical_reports WHERE LOWER(patient_name) = 'john';
```

### Use LIMIT for large result sets
```sql
-- ✅ Good
SELECT * FROM medical_reports 
ORDER BY upload_date DESC 
LIMIT 100;

-- ❌ Bad - returns all rows
SELECT * FROM medical_reports 
ORDER BY upload_date DESC;
```

### Use EXISTS instead of IN for subqueries
```sql
-- ✅ Good
SELECT r.* FROM medical_reports r
WHERE EXISTS (
    SELECT 1 FROM doctor_reviews dr 
    WHERE dr.report_id = r.report_id
);

-- ❌ Less efficient
SELECT r.* FROM medical_reports r
WHERE r.report_id IN (
    SELECT report_id FROM doctor_reviews
);
```

### Batch inserts
```sql
-- ✅ Good - single query
INSERT INTO notifications (user_id, type, title, message)
VALUES 
    ('uuid1', 'new_report', 'Title 1', 'Message 1'),
    ('uuid2', 'new_report', 'Title 2', 'Message 2'),
    ('uuid3', 'new_report', 'Title 3', 'Message 3');

-- ❌ Bad - multiple queries
-- INSERT INTO notifications ... (repeated 3 times)
```

---

## Common Joins

### Report with AI analysis and doctor review
```sql
SELECT 
    r.report_id,
    r.patient_name,
    r.study_type,
    r.risk_score,
    aa.risk_probability,
    aa.findings as ai_findings,
    dr.review_notes,
    dr.doctor_risk_assessment,
    u.name as doctor_name
FROM medical_reports r
LEFT JOIN ai_analysis aa ON r.report_id = aa.report_id
LEFT JOIN doctor_reviews dr ON r.report_id = dr.report_id
LEFT JOIN users u ON dr.doctor_id = u.id
WHERE r.patient_id = 'patient-uuid-here'
ORDER BY r.upload_date DESC;
```

### User activity with audit logs
```sql
SELECT 
    u.name,
    u.email,
    u.role,
    al.action,
    al.entity_type,
    al.created_at
FROM users u
JOIN audit_logs al ON u.id = al.user_id
WHERE u.id = 'user-uuid-here'
ORDER BY al.created_at DESC
LIMIT 50;
```

---

## Useful Aggregations

### Reports count by risk score
```sql
SELECT 
    risk_score,
    COUNT(*) as total_count,
    COUNT(CASE WHEN status = 'reviewed' THEN 1 END) as reviewed_count,
    COUNT(CASE WHEN status = 'analyzed' THEN 1 END) as pending_review_count
FROM medical_reports
WHERE upload_date >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY risk_score
ORDER BY 
    CASE risk_score
        WHEN 'high' THEN 1
        WHEN 'medium' THEN 2
        WHEN 'low' THEN 3
    END;
```

### AI model performance over time
```sql
SELECT 
    DATE(mm.recorded_at) as date,
    AVG(mm.accuracy) as avg_accuracy,
    AVG(mm.precision) as avg_precision,
    AVG(mm.recall) as avg_recall,
    SUM(mm.total_predictions) as total_predictions
FROM model_metrics mm
WHERE mm.model_id = 'model-uuid-here'
  AND mm.recorded_at >= CURRENT_DATE - INTERVAL '30 days'
GROUP BY DATE(mm.recorded_at)
ORDER BY date DESC;
```

### User activity statistics
```sql
SELECT 
    u.name,
    u.role,
    COUNT(DISTINCT s.session_id) as total_sessions,
    MAX(s.login_time) as last_login,
    COUNT(al.log_id) as total_actions
FROM users u
LEFT JOIN sessions s ON u.id = s.user_id
LEFT JOIN audit_logs al ON u.id = al.user_id
WHERE u.is_active = true
GROUP BY u.id, u.name, u.role
ORDER BY total_actions DESC;
```

---

## Transaction Examples

### Upload report with transaction
```sql
BEGIN;

-- Insert report
INSERT INTO medical_reports (
    patient_id, radiologist_id, patient_name, 
    radiologist_name, study_type, report_image_url
) VALUES (
    'patient-uuid', 'radiologist-uuid', 'Patient Name',
    'Dr. Radiologist', 'Chest X-Ray', 'https://...'
) RETURNING report_id INTO @report_id;

-- Insert AI analysis
INSERT INTO ai_analysis (
    report_id, model_id, risk_category, risk_probability
) VALUES (
    @report_id, 'model-uuid', 'medium', 0.65
);

-- Send notification
INSERT INTO notifications (
    user_id, type, title, message, related_report_id
) VALUES (
    'patient-uuid', 'new_report', 'Analysis Complete',
    'Your X-ray analysis is ready', @report_id
);

COMMIT;
```

---

## Debugging Queries

### Check table sizes
```sql
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Find slow queries
```sql
SELECT 
    query,
    calls,
    total_exec_time,
    mean_exec_time,
    max_exec_time
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;
```

### Check index usage
```sql
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_scan,
    idx_tup_read,
    idx_tup_fetch
FROM pg_stat_user_indexes
ORDER BY idx_scan DESC;
```

---

**Last Updated**: October 2, 2025  
**Database Version**: 1.0.0