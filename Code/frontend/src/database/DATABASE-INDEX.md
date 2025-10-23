# 🗄️ Medical Portal Database - Complete Index

Comprehensive index for all database documentation and files.

---

## 📁 Database Files

| File | Description | Use Case |
|------|-------------|----------|
| [**schema.sql**](./schema.sql) | Complete PostgreSQL database schema | Initial database setup |
| [**seed-data.sql**](./seed-data.sql) | Sample data for development/testing | Development & testing |
| [**README.md**](./README.md) | Database documentation & setup guide | Setup & maintenance |
| [**ER-Diagram-Visual.md**](./ER-Diagram-Visual.md) | Visual ER diagrams with Mermaid | Understanding schema |
| [**Quick-Reference.md**](./Quick-Reference.md) | Common queries & operations | Daily operations |
| [**DATABASE-INDEX.md**](./DATABASE-INDEX.md) | This file | Navigation |

---

## 🎯 Quick Start

### **New to the Database?**
1. Read [README.md](./README.md) - Setup instructions
2. Review [ER-Diagram-Visual.md](./ER-Diagram-Visual.md) - Visual schema
3. Run [schema.sql](./schema.sql) - Create database
4. Load [seed-data.sql](./seed-data.sql) - Sample data
5. Reference [Quick-Reference.md](./Quick-Reference.md) - Common queries

### **Need to Query the Database?**
→ Go to [Quick-Reference.md](./Quick-Reference.md)

### **Setting Up Database?**
→ Follow [README.md - Setup Instructions](./README.md#-setup-instructions)

### **Understanding Schema?**
→ Check [ER-Diagram-Visual.md](./ER-Diagram-Visual.md)

---

## 📊 Database Overview

### Technology Stack
- **Database**: PostgreSQL 14+
- **Extension**: uuid-ossp (UUID generation)
- **Data Types**: UUID, JSONB, TIMESTAMP WITH TIME ZONE, INET
- **Features**: Triggers, Views, Constraints, Indexes

### Database Statistics
- **Total Tables**: 12
- **Total Views**: 2
- **Total Triggers**: 4
- **Total Indexes**: 40+
- **Sample Users**: 9 (in seed data)
- **Sample Reports**: 4 (in seed data)

---

## 📋 Complete Table List

### Core Tables (12)

| # | Table Name | Rows (Est) | Primary Purpose |
|---|------------|------------|----------------|
| 1 | **users** | 10K | All system users (patients, radiologists, doctors, tech) |
| 2 | **sessions** | 50K | Authentication sessions |
| 3 | **medical_reports** | 500K | X-ray reports with AI analysis results |
| 4 | **report_images** | 500K | Multiple images per report |
| 5 | **ai_models** | 10 | AI model versions and configurations |
| 6 | **ai_analysis** | 500K | Detailed AI analysis for each report |
| 7 | **model_metrics** | 1K | AI model performance tracking |
| 8 | **doctor_reviews** | 300K | Doctor reviews and validations |
| 9 | **notifications** | 2M | System notifications for users |
| 10 | **audit_logs** | 5M | Comprehensive audit trail |
| 11 | **patient_medical_history** | 10K | Patient medical history |
| 12 | **system_settings** | 50 | Application configuration |

---

## 🔍 Table Details

### 1. USERS Table
**Purpose**: Store all system users across 4 roles

**Key Columns**:
```sql
id                  UUID PRIMARY KEY
email               VARCHAR(255) UNIQUE
role                VARCHAR(20) CHECK IN (patient, radiologist, doctor, tech)
password_hash       VARCHAR(255)
preferences         JSONB
is_active           BOOLEAN
```

**Relationships**:
- → sessions (1:N)
- → medical_reports (1:N as patient)
- → medical_reports (1:N as radiologist)
- → doctor_reviews (1:N)
- → notifications (1:N)
- → patient_medical_history (1:1)

**Indexes**: 4  
**Sample Query**: [Quick-Reference.md - User Management](./Quick-Reference.md#user-management)

---

### 2. SESSIONS Table
**Purpose**: Manage user authentication sessions

**Key Columns**:
```sql
session_id          UUID PRIMARY KEY
user_id             UUID FOREIGN KEY → users(id)
token               VARCHAR(512) UNIQUE
expires_at          TIMESTAMP WITH TIME ZONE
is_active           BOOLEAN
```

**Relationships**:
- ← users (N:1)

**Indexes**: 4  
**Sample Query**: [Quick-Reference.md - Get User Sessions](./Quick-Reference.md#get-users-active-sessions)

---

### 3. MEDICAL_REPORTS Table
**Purpose**: Central table for medical reports

**Key Columns**:
```sql
report_id           UUID PRIMARY KEY
patient_id          UUID FOREIGN KEY → users(id)
radiologist_id      UUID FOREIGN KEY → users(id)
risk_score          VARCHAR(20) CHECK IN (low, medium, high)
confidence_score    DECIMAL(5,4)
status              VARCHAR(20) CHECK IN (pending, analyzing, analyzed, reviewed)
```

**Relationships**:
- ← users (N:1 as patient)
- ← users (N:1 as radiologist)
- → report_images (1:N)
- → ai_analysis (1:1)
- → doctor_reviews (1:N)
- → notifications (1:N)

**Indexes**: 6  
**Sample Query**: [Quick-Reference.md - Medical Reports](./Quick-Reference.md#medical-reports)

---

### 4. AI_ANALYSIS Table
**Purpose**: Store AI model predictions and analysis

**Key Columns**:
```sql
analysis_id         UUID PRIMARY KEY
report_id           UUID FOREIGN KEY → medical_reports(report_id)
model_id            UUID FOREIGN KEY → ai_models(model_id)
risk_category       VARCHAR(20) CHECK IN (low, medium, high)
risk_probability    DECIMAL(5,4)
findings            JSONB
```

**Relationships**:
- ← medical_reports (1:1)
- ← ai_models (N:1)

**Indexes**: 4  
**Sample Query**: [Quick-Reference.md - AI Analysis](./Quick-Reference.md#ai-analysis)

---

### 5. DOCTOR_REVIEWS Table
**Purpose**: Doctor reviews and validation

**Key Columns**:
```sql
review_id           UUID PRIMARY KEY
report_id           UUID FOREIGN KEY → medical_reports(report_id)
doctor_id           UUID FOREIGN KEY → users(id)
urgency_level       VARCHAR(20) CHECK IN (routine, follow_up, urgent, critical)
confirmed_ai_diagnosis BOOLEAN
```

**Relationships**:
- ← medical_reports (N:1)
- ← users (N:1)

**Indexes**: 4  
**Sample Query**: [Quick-Reference.md - Doctor Reviews](./Quick-Reference.md#doctor-reviews)

---

### 6. NOTIFICATIONS Table
**Purpose**: System notifications for users

**Key Columns**:
```sql
notification_id     UUID PRIMARY KEY
user_id             UUID FOREIGN KEY → users(id)
type                VARCHAR(50)
priority            VARCHAR(20) CHECK IN (low, normal, high, urgent)
is_read             BOOLEAN
```

**Relationships**:
- ← users (N:1)
- ← medical_reports (N:1, optional)

**Indexes**: 5  
**Sample Query**: [Quick-Reference.md - Notifications](./Quick-Reference.md#notifications)

---

## 🔗 Database Relationships

### Primary Relationships

```
USERS
├── has many SESSIONS
├── has many MEDICAL_REPORTS (as patient)
├── uploads many MEDICAL_REPORTS (as radiologist)
├── creates many DOCTOR_REVIEWS (as doctor)
├── receives many NOTIFICATIONS
└── has one PATIENT_MEDICAL_HISTORY

MEDICAL_REPORTS
├── contains many REPORT_IMAGES
├── has one AI_ANALYSIS
├── reviewed by many DOCTOR_REVIEWS
└── triggers many NOTIFICATIONS

AI_MODELS
├── generates many AI_ANALYSIS
└── tracked by many MODEL_METRICS
```

### Complete ER Diagram
See [ER-Diagram-Visual.md](./ER-Diagram-Visual.md) for complete visual representation.

---

## 📖 Documentation by Topic

### Setup & Installation
- [README.md - Setup Instructions](./README.md#-setup-instructions)
- [README.md - Install PostgreSQL](./README.md#1-install-postgresql)
- [README.md - Create Database](./README.md#2-create-database)
- [README.md - Run Schema](./README.md#3-run-schema)

### Schema & Design
- [schema.sql](./schema.sql) - Full SQL schema
- [ER-Diagram-Visual.md](./ER-Diagram-Visual.md) - Visual diagrams
- [README.md - Database Schema](./README.md#-database-schema)

### Development & Testing
- [seed-data.sql](./seed-data.sql) - Sample data
- [README.md - Test Database Setup](./README.md#test-database-setup)

### Operations
- [Quick-Reference.md](./Quick-Reference.md) - All common queries
- [README.md - Performance Optimization](./README.md#-performance-optimization)
- [README.md - Backup and Recovery](./README.md#-backup-and-recovery)

### Security
- [README.md - Security Considerations](./README.md#-security-considerations)
- [ER-Diagram-Visual.md - Security Notes](./ER-Diagram-Visual.md#security-notes)

### Monitoring & Maintenance
- [README.md - Monitoring](./README.md#-monitoring)
- [README.md - Maintenance](./README.md#-maintenance)
- [Quick-Reference.md - Debugging Queries](./Quick-Reference.md#debugging-queries)

---

## 🚀 Common Tasks

### Task: Set up new development environment
1. Install PostgreSQL: [README.md - Install](./README.md#1-install-postgresql)
2. Create database: [README.md - Create DB](./README.md#2-create-database)
3. Run schema: `psql -d medical_portal -f schema.sql`
4. Load seed data: `psql -d medical_portal -f seed-data.sql`
5. Verify: `psql -d medical_portal -c "\dt"`

### Task: Query patient reports
```sql
-- See Quick-Reference.md for more examples
SELECT * FROM v_patient_reports 
WHERE patient_id = 'patient-uuid-here';
```

### Task: Create new user
```sql
-- See Quick-Reference.md - Insert Operations
INSERT INTO users (name, email, password_hash, role)
VALUES ('Name', 'email@example.com', 'hashed_password', 'patient');
```

### Task: Upload new report
See [Quick-Reference.md - Upload Report](./Quick-Reference.md#upload-a-new-medical-report)

### Task: Backup database
```bash
# See README.md - Backup and Recovery
pg_dump medical_portal > backup_$(date +%Y%m%d).sql
```

### Task: Check performance
```sql
-- See Quick-Reference.md - Debugging
SELECT tablename, pg_size_pretty(pg_total_relation_size(tablename::regclass))
FROM pg_tables WHERE schemaname = 'public';
```

---

## 📊 Database Views

### v_patient_reports
**Purpose**: Combined view of reports with AI analysis and doctor reviews

**Usage**:
```sql
SELECT * FROM v_patient_reports 
WHERE patient_id = 'patient-uuid-here';
```

**Columns**: report_id, patient_id, patient_name, study_type, risk_score, confidence_score, upload_date, analysis_date, status, risk_probability, processed_at, review_id, doctor_id, urgency_level, reviewed_at

---

### v_doctor_priority_queue
**Purpose**: Prioritized reports for doctor review (HIGH risk first)

**Usage**:
```sql
SELECT * FROM v_doctor_priority_queue LIMIT 20;
```

**Columns**: report_id, patient_id, patient_name, study_type, risk_score, confidence_score, upload_date, analysis_date, risk_probability, priority_order

---

## 🔧 Database Functions & Triggers

### update_updated_at_column()
**Purpose**: Automatically update `updated_at` timestamp on row updates

**Applied to**:
- users
- medical_reports
- doctor_reviews
- patient_medical_history

**Example**:
```sql
-- Automatically called on UPDATE
UPDATE users SET name = 'New Name' WHERE id = 'uuid';
-- updated_at is automatically set to CURRENT_TIMESTAMP
```

---

## 📈 Performance Tips

### Indexing
✅ **All foreign keys are indexed**  
✅ **Filter columns are indexed** (status, risk_score, is_read, etc.)  
✅ **Sort columns are indexed** (upload_date, created_at, etc.)

### Query Optimization
- Use `EXPLAIN ANALYZE` to check query plans
- Use `LIMIT` for large result sets
- Use `EXISTS` instead of `IN` for subqueries
- Batch inserts when possible

See [Quick-Reference.md - Performance Tips](./Quick-Reference.md#performance-tips)

---

## 🔐 Security Features

### Authentication
- ✅ Password hashing (bcrypt recommended)
- ✅ Session tokens with expiration
- ✅ IP tracking and user agent logging
- ✅ Active session management

### Data Protection
- ✅ Role-based access control (RBAC)
- ✅ Audit logging (all actions tracked)
- ✅ Encryption ready (at rest and in transit)
- ✅ HIPAA compliance ready

### Constraints
- ✅ CHECK constraints on enums (role, risk_score, status)
- ✅ UNIQUE constraints (email, token)
- ✅ NOT NULL constraints on critical fields
- ✅ Foreign key constraints with CASCADE/RESTRICT rules

---

## 📞 Support & Troubleshooting

### Common Issues

**Problem**: Connection refused  
**Solution**: Check if PostgreSQL is running: `sudo systemctl status postgresql`

**Problem**: Permission denied  
**Solution**: Grant permissions: [README.md - Grants](./README.md#grants)

**Problem**: Slow queries  
**Solution**: Check indexes and use EXPLAIN ANALYZE: [Quick-Reference.md - Debugging](./Quick-Reference.md#debugging-queries)

**Problem**: Database too large  
**Solution**: Archive old data, vacuum: [README.md - Maintenance](./README.md#-maintenance)

### Resources
- [PostgreSQL Documentation](https://www.postgresql.org/docs/14/)
- [README.md - Troubleshooting](./README.md#-troubleshooting)
- Email: tech-support@medicalportal.com

---

## 📝 Version History

| Version | Date | Changes | Migration |
|---------|------|---------|-----------|
| 1.0.0 | 2025-10-02 | Initial schema | Initial setup |

---

## 🎯 Next Steps

### For Developers
1. Clone repository
2. Set up PostgreSQL
3. Run schema.sql
4. Load seed-data.sql
5. Start building!

### For Database Admins
1. Review security settings
2. Set up backups
3. Configure monitoring
4. Set up replicas (if needed)
5. Optimize based on usage

### For Data Scientists
1. Review AI model tables
2. Understand metrics tracking
3. Plan training pipeline
4. Set up analytics queries
5. Monitor model performance

---

**Last Updated**: October 2, 2025  
**Database Version**: 1.0.0  
**PostgreSQL Version**: 14+  
**Status**: Production Ready ✅