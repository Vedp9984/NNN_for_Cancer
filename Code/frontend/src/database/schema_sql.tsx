-- ============================================
-- Medical Portal System - Database Schema
-- Version: 1.0.0
-- Database: PostgreSQL 14+
-- ============================================

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- USERS TABLE
-- Stores all system users (patients, radiologists, doctors, tech team)
-- ============================================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) NOT NULL CHECK (role IN ('patient', 'radiologist', 'doctor', 'tech')),
    profile_picture_url TEXT,
    phone VARCHAR(20),
    date_of_birth DATE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    is_email_verified BOOLEAN DEFAULT false,
    
    -- User preferences (stored as JSONB)
    preferences JSONB DEFAULT '{
        "theme": "system",
        "notifications_enabled": true,
        "email_notifications": true,
        "sms_notifications": false,
        "language": "en"
    }'::jsonb,
    
    -- Audit fields
    created_by UUID,
    updated_by UUID
);

-- Indexes for users table
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_role ON users(role);
CREATE INDEX idx_users_is_active ON users(is_active);
CREATE INDEX idx_users_created_at ON users(created_at DESC);

-- ============================================
-- SESSIONS TABLE
-- Manages user authentication sessions
-- ============================================
CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Session details
    token VARCHAR(512) NOT NULL UNIQUE,
    refresh_token VARCHAR(512),
    
    -- Session metadata
    ip_address INET,
    user_agent TEXT,
    device_info JSONB,
    
    -- Timestamps
    login_time TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    
    -- Session status
    is_active BOOLEAN DEFAULT true,
    logout_time TIMESTAMP WITH TIME ZONE
);

-- Indexes for sessions table
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token);
CREATE INDEX idx_sessions_is_active ON sessions(is_active);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);

-- ============================================
-- MEDICAL_REPORTS TABLE
-- Central table for X-ray reports and patient data
-- ============================================
CREATE TABLE medical_reports (
    report_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Patient and radiologist references
    patient_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    radiologist_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    
    -- Report details
    patient_name VARCHAR(255) NOT NULL,
    radiologist_name VARCHAR(255) NOT NULL,
    study_type VARCHAR(100) NOT NULL, -- e.g., 'Chest X-Ray', 'Lung CT', etc.
    
    -- Image storage
    report_image_url TEXT NOT NULL,
    image_metadata JSONB, -- stores width, height, format, size, etc.
    
    -- AI Analysis results
    risk_score VARCHAR(20) CHECK (risk_score IN ('low', 'medium', 'high')),
    confidence_score DECIMAL(5,4) CHECK (confidence_score BETWEEN 0 AND 1),
    
    -- AI findings and recommendations
    findings TEXT,
    recommendations JSONB, -- array of recommendation strings
    
    -- Timestamps
    upload_date TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    analysis_date TIMESTAMP WITH TIME ZONE,
    
    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'analyzing', 'analyzed', 'reviewed', 'archived')),
    
    -- Additional metadata
    patient_age INTEGER,
    patient_gender VARCHAR(20),
    medical_history JSONB,
    symptoms JSONB,
    
    -- Audit fields
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id),
    updated_by UUID REFERENCES users(id)
);

-- Indexes for medical_reports table
CREATE INDEX idx_reports_patient_id ON medical_reports(patient_id);
CREATE INDEX idx_reports_radiologist_id ON medical_reports(radiologist_id);
CREATE INDEX idx_reports_status ON medical_reports(status);
CREATE INDEX idx_reports_risk_score ON medical_reports(risk_score);
CREATE INDEX idx_reports_upload_date ON medical_reports(upload_date DESC);
CREATE INDEX idx_reports_analysis_date ON medical_reports(analysis_date DESC);

-- ============================================
-- REPORT_IMAGES TABLE
-- Stores multiple images per report (if needed)
-- ============================================
CREATE TABLE report_images (
    image_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    
    -- Image details
    image_url TEXT NOT NULL,
    image_type VARCHAR(50) NOT NULL, -- 'xray', 'ct_scan', 'mri', etc.
    image_format VARCHAR(20), -- 'JPEG', 'PNG', 'DICOM'
    
    -- Image dimensions and metadata
    width INTEGER,
    height INTEGER,
    file_size BIGINT, -- in bytes
    
    -- DICOM metadata (if applicable)
    dicom_metadata JSONB,
    
    -- Timestamps
    uploaded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Order for multiple images
    display_order INTEGER DEFAULT 0
);

-- Indexes for report_images table
CREATE INDEX idx_report_images_report_id ON report_images(report_id);
CREATE INDEX idx_report_images_image_type ON report_images(image_type);

-- ============================================
-- AI_MODELS TABLE
-- Tracks different AI model versions and configurations
-- ============================================
CREATE TABLE ai_models (
    model_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Model identification
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    
    -- Model details
    architecture VARCHAR(100), -- 'CNN', 'ResNet', 'EfficientNet', etc.
    framework VARCHAR(50), -- 'TensorFlow', 'PyTorch', etc.
    
    -- Model file storage
    model_file_url TEXT,
    model_file_size BIGINT,
    
    -- Model configuration
    model_config JSONB,
    
    -- Training information
    trained_date TIMESTAMP WITH TIME ZONE,
    training_dataset_version VARCHAR(50),
    training_duration_hours DECIMAL(10,2),
    
    -- Deployment information
    deployed_date TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT false,
    
    -- Performance baseline
    baseline_accuracy DECIMAL(5,4),
    baseline_precision DECIMAL(5,4),
    baseline_recall DECIMAL(5,4),
    baseline_f1_score DECIMAL(5,4),
    
    -- Metadata
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_by UUID REFERENCES users(id)
);

-- Indexes for ai_models table
CREATE INDEX idx_ai_models_is_active ON ai_models(is_active);
CREATE INDEX idx_ai_models_version ON ai_models(version);
CREATE INDEX idx_ai_models_deployed_date ON ai_models(deployed_date DESC);

-- ============================================
-- AI_ANALYSIS TABLE
-- Stores detailed AI analysis results for each report
-- ============================================
CREATE TABLE ai_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    model_id UUID NOT NULL REFERENCES ai_models(model_id) ON DELETE RESTRICT,
    
    -- Analysis results
    risk_category VARCHAR(20) NOT NULL CHECK (risk_category IN ('low', 'medium', 'high')),
    risk_probability DECIMAL(5,4) NOT NULL CHECK (risk_probability BETWEEN 0 AND 1),
    
    -- Class probabilities
    low_probability DECIMAL(5,4) CHECK (low_probability BETWEEN 0 AND 1),
    medium_probability DECIMAL(5,4) CHECK (medium_probability BETWEEN 0 AND 1),
    high_probability DECIMAL(5,4) CHECK (high_probability BETWEEN 0 AND 1),
    
    -- Detailed findings
    findings JSONB, -- structured findings from the AI
    feature_maps JSONB, -- feature extraction data
    attention_maps JSONB, -- attention/heatmap data for visualization
    
    -- Processing information
    processed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER, -- processing time in milliseconds
    
    -- Quality metrics
    image_quality_score DECIMAL(5,4),
    confidence_threshold DECIMAL(5,4),
    
    -- Additional metadata
    preprocessing_steps JSONB,
    model_version VARCHAR(50),
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for ai_analysis table
CREATE INDEX idx_ai_analysis_report_id ON ai_analysis(report_id);
CREATE INDEX idx_ai_analysis_model_id ON ai_analysis(model_id);
CREATE INDEX idx_ai_analysis_risk_category ON ai_analysis(risk_category);
CREATE INDEX idx_ai_analysis_processed_at ON ai_analysis(processed_at DESC);

-- ============================================
-- MODEL_METRICS TABLE
-- Tracks AI model performance over time
-- ============================================
CREATE TABLE model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    model_id UUID NOT NULL REFERENCES ai_models(model_id) ON DELETE CASCADE,
    
    -- Performance metrics
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    
    -- Confusion matrix elements
    true_positives INTEGER,
    true_negatives INTEGER,
    false_positives INTEGER,
    false_negatives INTEGER,
    
    -- Prediction counts
    total_predictions INTEGER DEFAULT 0,
    correct_predictions INTEGER DEFAULT 0,
    
    -- Time period for metrics
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    period_start TIMESTAMP WITH TIME ZONE,
    period_end TIMESTAMP WITH TIME ZONE,
    
    -- Dataset information
    dataset_version VARCHAR(50),
    test_set_size INTEGER,
    
    -- Additional metrics
    auc_roc DECIMAL(5,4),
    average_processing_time_ms INTEGER,
    
    -- Metadata
    notes TEXT
);

-- Indexes for model_metrics table
CREATE INDEX idx_model_metrics_model_id ON model_metrics(model_id);
CREATE INDEX idx_model_metrics_recorded_at ON model_metrics(recorded_at DESC);

-- ============================================
-- DOCTOR_REVIEWS TABLE
-- Stores doctor reviews and validation of AI analysis
-- ============================================
CREATE TABLE doctor_reviews (
    review_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- References
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    doctor_id UUID NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    
    -- Review details
    review_notes TEXT NOT NULL,
    clinical_findings TEXT,
    
    -- AI validation
    confirmed_ai_diagnosis BOOLEAN,
    ai_accuracy_rating INTEGER CHECK (ai_accuracy_rating BETWEEN 1 AND 5),
    
    -- Doctor's assessment
    doctor_risk_assessment VARCHAR(20) CHECK (doctor_risk_assessment IN ('low', 'medium', 'high')),
    urgency_level VARCHAR(20) CHECK (urgency_level IN ('routine', 'follow_up', 'urgent', 'critical')),
    
    -- Recommendations
    recommended_action TEXT,
    follow_up_required BOOLEAN DEFAULT false,
    follow_up_date DATE,
    
    -- Treatment plan
    treatment_plan TEXT,
    medication_prescribed JSONB,
    
    -- Timestamps
    reviewed_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Audit
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for doctor_reviews table
CREATE INDEX idx_doctor_reviews_report_id ON doctor_reviews(report_id);
CREATE INDEX idx_doctor_reviews_doctor_id ON doctor_reviews(doctor_id);
CREATE INDEX idx_doctor_reviews_urgency_level ON doctor_reviews(urgency_level);
CREATE INDEX idx_doctor_reviews_reviewed_at ON doctor_reviews(reviewed_at DESC);

-- ============================================
-- NOTIFICATIONS TABLE
-- Manages system notifications for users
-- ============================================
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Recipient
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Notification details
    type VARCHAR(50) NOT NULL, -- 'new_report', 'urgent_case', 'review_complete', 'system_alert', etc.
    priority VARCHAR(20) DEFAULT 'normal' CHECK (priority IN ('low', 'normal', 'high', 'urgent')),
    
    -- Content
    title VARCHAR(255) NOT NULL,
    message TEXT NOT NULL,
    action_url TEXT,
    
    -- Status
    is_read BOOLEAN DEFAULT false,
    read_at TIMESTAMP WITH TIME ZONE,
    
    -- Related entities
    related_report_id UUID REFERENCES medical_reports(report_id) ON DELETE SET NULL,
    
    -- Delivery channels
    sent_via_email BOOLEAN DEFAULT false,
    sent_via_sms BOOLEAN DEFAULT false,
    sent_via_push BOOLEAN DEFAULT false,
    
    -- Metadata
    metadata JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE
);

-- Indexes for notifications table
CREATE INDEX idx_notifications_user_id ON notifications(user_id);
CREATE INDEX idx_notifications_is_read ON notifications(is_read);
CREATE INDEX idx_notifications_type ON notifications(type);
CREATE INDEX idx_notifications_priority ON notifications(priority);
CREATE INDEX idx_notifications_created_at ON notifications(created_at DESC);

-- ============================================
-- AUDIT_LOGS TABLE
-- Comprehensive audit trail for compliance
-- ============================================
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- User who performed the action
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    user_email VARCHAR(255),
    user_role VARCHAR(20),
    
    -- Action details
    action VARCHAR(100) NOT NULL, -- 'login', 'upload_report', 'view_report', 'edit_profile', etc.
    entity_type VARCHAR(50), -- 'user', 'report', 'review', etc.
    entity_id UUID,
    
    -- Request details
    ip_address INET,
    user_agent TEXT,
    
    -- Changes (for update operations)
    old_values JSONB,
    new_values JSONB,
    
    -- Status
    status VARCHAR(20), -- 'success', 'failure', 'error'
    error_message TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for audit_logs table
CREATE INDEX idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX idx_audit_logs_action ON audit_logs(action);
CREATE INDEX idx_audit_logs_entity_type ON audit_logs(entity_type);
CREATE INDEX idx_audit_logs_created_at ON audit_logs(created_at DESC);

-- ============================================
-- PATIENT_MEDICAL_HISTORY TABLE
-- Stores comprehensive patient medical history
-- ============================================
CREATE TABLE patient_medical_history (
    history_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    patient_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    
    -- Medical conditions
    conditions JSONB, -- array of medical conditions
    allergies JSONB, -- array of allergies
    medications JSONB, -- current medications
    
    -- Family history
    family_history JSONB,
    
    -- Lifestyle factors
    smoking_status VARCHAR(50),
    alcohol_consumption VARCHAR(50),
    exercise_frequency VARCHAR(50),
    
    -- Previous procedures
    previous_surgeries JSONB,
    previous_hospitalizations JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id)
);

-- Indexes for patient_medical_history table
CREATE INDEX idx_patient_history_patient_id ON patient_medical_history(patient_id);

-- ============================================
-- SYSTEM_SETTINGS TABLE
-- Stores application-wide configuration
-- ============================================
CREATE TABLE system_settings (
    setting_id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Setting details
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    
    -- Metadata
    description TEXT,
    is_sensitive BOOLEAN DEFAULT false,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_by UUID REFERENCES users(id)
);

-- ============================================
-- FUNCTIONS AND TRIGGERS
-- ============================================

-- Function to update 'updated_at' timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updating 'updated_at' columns
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_medical_reports_updated_at BEFORE UPDATE ON medical_reports
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_doctor_reviews_updated_at BEFORE UPDATE ON doctor_reviews
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_patient_history_updated_at BEFORE UPDATE ON patient_medical_history
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================
-- VIEWS
-- ============================================

-- View for patient report summary
CREATE OR REPLACE VIEW v_patient_reports AS
SELECT 
    mr.report_id,
    mr.patient_id,
    mr.patient_name,
    mr.study_type,
    mr.risk_score,
    mr.confidence_score,
    mr.upload_date,
    mr.analysis_date,
    mr.status,
    aa.risk_probability,
    aa.processed_at,
    dr.review_id,
    dr.doctor_id,
    dr.urgency_level,
    dr.reviewed_at
FROM medical_reports mr
LEFT JOIN ai_analysis aa ON mr.report_id = aa.report_id
LEFT JOIN doctor_reviews dr ON mr.report_id = dr.report_id;

-- View for doctor priority queue
CREATE OR REPLACE VIEW v_doctor_priority_queue AS
SELECT 
    mr.report_id,
    mr.patient_id,
    mr.patient_name,
    mr.study_type,
    mr.risk_score,
    mr.confidence_score,
    mr.upload_date,
    mr.analysis_date,
    aa.risk_probability,
    CASE 
        WHEN mr.risk_score = 'high' THEN 1
        WHEN mr.risk_score = 'medium' THEN 2
        WHEN mr.risk_score = 'low' THEN 3
        ELSE 4
    END as priority_order
FROM medical_reports mr
LEFT JOIN ai_analysis aa ON mr.report_id = aa.report_id
WHERE mr.status = 'analyzed'
ORDER BY priority_order ASC, mr.analysis_date DESC;

-- ============================================
-- SAMPLE DATA (for development/testing)
-- ============================================

-- Insert sample users (passwords should be hashed in production)
INSERT INTO users (id, name, email, password_hash, role) VALUES
    ('a1111111-1111-1111-1111-111111111111', 'John Smith', 'patient@example.com', '$2a$10$hashedpassword', 'patient'),
    ('b2222222-2222-2222-2222-222222222222', 'Dr. Sarah Chen', 'radiologist@example.com', '$2a$10$hashedpassword', 'radiologist'),
    ('c3333333-3333-3333-3333-333333333333', 'Dr. Michael Johnson', 'doctor@example.com', '$2a$10$hashedpassword', 'doctor'),
    ('d4444444-4444-4444-4444-444444444444', 'Alex Rodriguez', 'tech@example.com', '$2a$10$hashedpassword', 'tech');

-- Insert sample AI model
INSERT INTO ai_models (model_id, model_name, version, architecture, framework, is_active, baseline_accuracy) VALUES
    ('e5555555-5555-5555-5555-555555555555', 'XRay Risk Classifier', 'v1.0.0', 'CNN', 'TensorFlow', true, 0.9250);

-- ============================================
-- COMMENTS
-- ============================================

COMMENT ON TABLE users IS 'Stores all system users including patients, radiologists, doctors, and tech team members';
COMMENT ON TABLE sessions IS 'Manages user authentication sessions with security tracking';
COMMENT ON TABLE medical_reports IS 'Central table for medical reports with AI analysis results';
COMMENT ON TABLE ai_analysis IS 'Detailed AI analysis results including risk scores and probabilities';
COMMENT ON TABLE ai_models IS 'Tracks different AI model versions and their configurations';
COMMENT ON TABLE model_metrics IS 'Performance metrics for AI models over time';
COMMENT ON TABLE doctor_reviews IS 'Doctor reviews and validation of AI-generated reports';
COMMENT ON TABLE notifications IS 'System notifications sent to users';
COMMENT ON TABLE audit_logs IS 'Comprehensive audit trail for compliance and security';

-- ============================================
-- GRANTS (adjust based on your user roles)
-- ============================================

-- Example: Grant permissions to application user
-- GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO medical_portal_app;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medical_portal_app;

-- ============================================
-- END OF SCHEMA
-- ============================================