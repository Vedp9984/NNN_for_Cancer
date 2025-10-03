 Database Schema (v1.0.0)
-- PostgreSQL 14+

-- ================= USERS & AUTH =================
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(20) CHECK (role IN ('patient','radiologist','doctor','tech')) NOT NULL,
    profile_picture_url TEXT,
    phone VARCHAR(20),
    date_of_birth DATE,
    is_active BOOLEAN DEFAULT TRUE,
    is_email_verified BOOLEAN DEFAULT FALSE,
    preferences JSONB DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    session_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    token VARCHAR(512) NOT NULL UNIQUE,
    refresh_token VARCHAR(512) NOT NULL UNIQUE,
    ip_address INET,
    user_agent TEXT,
    login_time TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    last_activity TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    is_active BOOLEAN DEFAULT TRUE,
    logout_time TIMESTAMPTZ
);

-- ================= MEDICAL REPORTS =================
CREATE TABLE medical_reports (
    report_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    radiologist_id UUID REFERENCES users(id),
    study_type VARCHAR(100),
    report_image_url TEXT,
    image_metadata JSONB,
    risk_score VARCHAR(10) CHECK (risk_score IN ('low','medium','high')),
    confidence_score DECIMAL(5,4),
    findings TEXT,
    recommendations JSONB,
    status VARCHAR(20) CHECK (status IN ('pending','analyzing','analyzed','reviewed','archived')) DEFAULT 'pending',
    upload_date TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    analysis_date TIMESTAMPTZ,
    patient_age INT,
    patient_gender VARCHAR(20),
    medical_history JSONB,
    symptoms JSONB
);

CREATE TABLE report_images (
    image_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    image_url TEXT NOT NULL,
    image_type VARCHAR(50),
    format VARCHAR(20),
    width INT,
    height INT,
    file_size BIGINT,
    dicom_metadata JSONB,
    display_order INT DEFAULT 0
);

-- ================= AI & ANALYSIS =================
CREATE TABLE ai_models (
    model_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    architecture VARCHAR(100),
    framework VARCHAR(50),
    model_file_url TEXT,
    trained_date TIMESTAMPTZ,
    deployed_date TIMESTAMPTZ,
    baseline_accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE ai_analysis (
    analysis_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    model_id UUID REFERENCES ai_models(model_id),
    risk_category VARCHAR(10) CHECK (risk_category IN ('low','medium','high')),
    risk_probability DECIMAL(5,4),
    findings JSONB,
    feature_maps JSONB,
    attention_maps JSONB,
    processed_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INT
);

CREATE TABLE model_metrics (
    metric_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_id UUID NOT NULL REFERENCES ai_models(model_id) ON DELETE CASCADE,
    accuracy DECIMAL(5,4),
    precision DECIMAL(5,4),
    recall DECIMAL(5,4),
    f1_score DECIMAL(5,4),
    confusion_matrix JSONB,
    total_predictions BIGINT DEFAULT 0,
    correct_predictions BIGINT DEFAULT 0,
    auc_roc DECIMAL(5,4),
    average_processing_time_ms INT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ================= DOCTOR REVIEWS =================
CREATE TABLE doctor_reviews (
    review_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    report_id UUID NOT NULL REFERENCES medical_reports(report_id) ON DELETE CASCADE,
    doctor_id UUID NOT NULL REFERENCES users(id),
    review_notes TEXT,
    clinical_findings TEXT,
    confirmed_ai_diagnosis BOOLEAN,
    ai_accuracy_rating INT CHECK (ai_accuracy_rating BETWEEN 1 AND 5),
    doctor_risk_assessment VARCHAR(10) CHECK (doctor_risk_assessment IN ('low','medium','high')),
    urgency_level VARCHAR(20) CHECK (urgency_level IN ('routine','follow_up','urgent','critical')),
    recommended_action TEXT,
    treatment_plan TEXT,
    medication_prescribed JSONB,
    follow_up_required BOOLEAN DEFAULT FALSE,
    follow_up_date DATE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ================= NOTIFICATIONS =================
CREATE TABLE notifications (
    notification_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    type VARCHAR(50),
    priority VARCHAR(10) CHECK (priority IN ('low','normal','high','urgent')) DEFAULT 'normal',
    title VARCHAR(255),
    message TEXT,
    action_url TEXT,
    is_read BOOLEAN DEFAULT FALSE,
    sent_via_email BOOLEAN DEFAULT FALSE,
    sent_via_sms BOOLEAN DEFAULT FALSE,
    sent_via_push BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ
);

-- ================= AUDIT & COMPLIANCE =================
CREATE TABLE audit_logs (
    log_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    action VARCHAR(100) NOT NULL,
    entity_type VARCHAR(50),
    entity_id UUID,
    ip_address INET,
    user_agent TEXT,
    old_values JSONB,
    new_values JSONB,
    status VARCHAR(20) CHECK (status IN ('success','failure','error')),
    error_message TEXT,
    created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ================= PATIENT HISTORY =================
CREATE TABLE patient_medical_history (
    history_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    patient_id UUID NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
    conditions JSONB,
    allergies JSONB,
    medications JSONB,
    family_history JSONB,
    lifestyle JSONB,
    previous_surgeries JSONB,
    hospitalizations JSONB,
    immunizations JSONB,
    last_updated TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);

-- ================= SYSTEM SETTINGS =================
CREATE TABLE system_settings (
    setting_id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    setting_key VARCHAR(100) UNIQUE NOT NULL,
    setting_value JSONB NOT NULL,
    description TEXT,
    is_sensitive BOOLEAN DEFAULT FALSE,
    updated_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
);
