# Medical Portal System - Database Documentation

## Overview

This directory contains all database-related files for the Medical Portal System, including schemas, migrations, seed data, and documentation.

## 📁 Files

- **schema.sql** - Complete PostgreSQL database schema
- **migrations/** - Database migration scripts
- **seeds/** - Sample data for development/testing
- **README.md** - This file

## 🗄️ Database Engine

**Primary Database**: PostgreSQL 14+

**Why PostgreSQL?**
- ✅ HIPAA compliance ready
- ✅ JSONB support for flexible data structures
- ✅ Advanced indexing capabilities
- ✅ Robust transaction support
- ✅ Excellent performance with large datasets
- ✅ Strong data integrity features

## 📊 Database Schema

### Core Tables

#### 1. **users** (13 columns)
Stores all system users across 4 roles.

**Key Fields**:
- `id` (UUID, PK)
- `email` (Unique)
- `role` (patient | radiologist | doctor | tech)
- `preferences` (JSONB)

**Indexes**: email, role, is_active, created_at

---

#### 2. **sessions** (11 columns)
Manages authentication sessions.

**Key Fields**:
- `session_id` (UUID, PK)
- `user_id` (FK → users)
- `token` (Unique)
- `expires_at`

**Indexes**: user_id, token, is_active, expires_at

---

#### 3. **medical_reports** (24 columns)
Central table for X-ray reports and AI analysis.

**Key Fields**:
- `report_id` (UUID, PK)
- `patient_id` (FK → users)
- `radiologist_id` (FK → users)
- `risk_score` (low | medium | high)
- `confidence_score` (0.0 - 1.0)
- `status` (pending | analyzing | analyzed | reviewed)

**Indexes**: patient_id, radiologist_id, status, risk_score, upload_date, analysis_date

---

#### 4. **report_images** (10 columns)
Stores multiple images per report.

**Key Fields**:
- `image_id` (UUID, PK)
- `report_id` (FK → medical_reports)
- `image_url`
- `image_type` (xray | ct_scan | mri)

**Indexes**: report_id, image_type

---

#### 5. **ai_models** (17 columns)
Tracks AI model versions and configurations.

**Key Fields**:
- `model_id` (UUID, PK)
- `model_name`
- `version`
- `is_active`
- `baseline_accuracy`

**Indexes**: is_active, version, deployed_date

---

#### 6. **ai_analysis** (18 columns)
Detailed AI analysis results for each report.

**Key Fields**:
- `analysis_id` (UUID, PK)
- `report_id` (FK → medical_reports)
- `model_id` (FK → ai_models)
- `risk_category` (low | medium | high)
- `risk_probability` (0.0 - 1.0)
- `findings` (JSONB)

**Indexes**: report_id, model_id, risk_category, processed_at

---

#### 7. **model_metrics** (16 columns)
Tracks AI model performance over time.

**Key Fields**:
- `metric_id` (UUID, PK)
- `model_id` (FK → ai_models)
- `accuracy`, `precision`, `recall`, `f1_score`
- `total_predictions`, `correct_predictions`

**Indexes**: model_id, recorded_at

---

#### 8. **doctor_reviews** (16 columns)
Doctor reviews and validation of AI analysis.

**Key Fields**:
- `review_id` (UUID, PK)
- `report_id` (FK → medical_reports)
- `doctor_id` (FK → users)
- `confirmed_ai_diagnosis` (Boolean)
- `urgency_level` (routine | follow_up | urgent | critical)

**Indexes**: report_id, doctor_id, urgency_level, reviewed_at

---

#### 9. **notifications** (16 columns)
System notifications for users.

**Key Fields**:
- `notification_id` (UUID, PK)
- `user_id` (FK → users)
- `type`, `priority`
- `is_read`
- `related_report_id` (FK → medical_reports)

**Indexes**: user_id, is_read, type, priority, created_at

---

#### 10. **audit_logs** (13 columns)
Comprehensive audit trail for compliance.

**Key Fields**:
- `log_id` (UUID, PK)
- `user_id` (FK → users)
- `action`, `entity_type`, `entity_id`
- `old_values`, `new_values` (JSONB)

**Indexes**: user_id, action, entity_type, created_at

---

#### 11. **patient_medical_history** (11 columns)
Patient medical history and conditions.

**Key Fields**:
- `history_id` (UUID, PK)
- `patient_id` (FK → users)
- `conditions`, `allergies`, `medications` (JSONB)

**Indexes**: patient_id

---

#### 12. **system_settings** (7 columns)
Application-wide configuration.

**Key Fields**:
- `setting_id` (UUID, PK)
- `setting_key` (Unique)
- `setting_value` (JSONB)

---

## 📈 Database Diagram

```
users (1) ----< (N) sessions
users (1) ----< (N) medical_reports [as patient]
users (1) ----< (N) medical_reports [as radiologist]
users (1) ----< (N) doctor_reviews
users (1) ----< (N) notifications
users (1) ----< (N) patient_medical_history

medical_reports (1) ----< (N) report_images
medical_reports (1) ---- (1) ai_analysis
medical_reports (1) ----< (N) doctor_reviews
medical_reports (1) ----< (N) notifications

ai_models (1) ----< (N) ai_analysis
ai_models (1) ----< (N) model_metrics
```

## 🚀 Setup Instructions

### 1. Install PostgreSQL

**macOS** (Homebrew):
```bash
brew install postgresql@14
brew services start postgresql@14
```

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install postgresql-14
sudo systemctl start postgresql
```

**Windows**:
Download from https://www.postgresql.org/download/windows/

### 2. Create Database

```bash
# Connect to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE medical_portal;

# Create application user
CREATE USER medical_portal_app WITH PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE medical_portal TO medical_portal_app;

# Exit psql
\q
```

### 3. Run Schema

```bash
# Connect to the database
psql -U postgres -d medical_portal

# Run schema file
\i database/schema.sql

# Verify tables were created
\dt

# Exit
\q
```

### 4. Verify Installation

```bash
# Check tables
psql -U postgres -d medical_portal -c "\dt"

# Check sample data
psql -U postgres -d medical_portal -c "SELECT * FROM users;"
```

## 🔐 Security Considerations

### Password Hashing
**Never store plain text passwords!**

Use bcrypt or similar:
```typescript
import bcrypt from 'bcrypt';

// Hash password
const hashedPassword = await bcrypt.hash(plainPassword, 10);

// Verify password
const isValid = await bcrypt.compare(plainPassword, hashedPassword);
```

### Connection Security
```javascript
// Use environment variables
const connectionString = process.env.DATABASE_URL;

// Enable SSL in production
const config = {
  connectionString,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false
};
```

### Data Encryption
- Use PostgreSQL's built-in encryption for sensitive fields
- Enable encryption at rest in production
- Use HTTPS/TLS for all database connections

## 📊 Performance Optimization

### Indexes
All critical columns have indexes:
- Foreign keys
- Frequently queried columns
- Sorting columns
- Filter columns

### Query Optimization Tips

```sql
-- Use EXPLAIN ANALYZE to check query performance
EXPLAIN ANALYZE SELECT * FROM medical_reports WHERE patient_id = 'xxx';

-- Create additional indexes if needed
CREATE INDEX idx_custom ON table_name(column_name);

-- Use partial indexes for filtered queries
CREATE INDEX idx_active_reports ON medical_reports(status) WHERE status = 'analyzed';
```

### Connection Pooling

```javascript
// Use pg-pool for connection pooling
import { Pool } from 'pg';

const pool = new Pool({
  user: 'medical_portal_app',
  host: 'localhost',
  database: 'medical_portal',
  password: process.env.DB_PASSWORD,
  port: 5432,
  max: 20, // maximum number of clients in pool
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 2000,
});
```

## 🔄 Backup and Recovery

### Daily Backup

```bash
# Backup entire database
pg_dump -U postgres medical_portal > backup_$(date +%Y%m%d).sql

# Backup with compression
pg_dump -U postgres medical_portal | gzip > backup_$(date +%Y%m%d).sql.gz
```

### Restore from Backup

```bash
# Restore database
psql -U postgres medical_portal < backup_20250101.sql

# Restore from compressed backup
gunzip -c backup_20250101.sql.gz | psql -U postgres medical_portal
```

### Automated Backups

```bash
# Add to crontab for daily backups at 2 AM
0 2 * * * pg_dump -U postgres medical_portal | gzip > /backups/medical_portal_$(date +\%Y\%m\%d).sql.gz
```

## 📝 Migration Strategy

### Version Control
- All schema changes go through migrations
- Never modify `schema.sql` directly in production
- Use numbered migration files: `001_initial.sql`, `002_add_field.sql`

### Running Migrations

```bash
# Create migration
touch database/migrations/003_add_new_field.sql

# Run migration
psql -U postgres -d medical_portal -f database/migrations/003_add_new_field.sql
```

## 🧪 Testing

### Test Database Setup

```bash
# Create test database
createdb medical_portal_test

# Run schema on test database
psql -U postgres -d medical_portal_test -f database/schema.sql

# Run tests
npm test
```

### Cleanup After Tests

```bash
# Drop test database
dropdb medical_portal_test
```

## 📊 Database Views

### Pre-created Views

1. **v_patient_reports** - Patient report summary with AI analysis
2. **v_doctor_priority_queue** - Prioritized reports for doctor review

### Usage

```sql
-- Get patient reports
SELECT * FROM v_patient_reports WHERE patient_id = 'xxx';

-- Get doctor priority queue
SELECT * FROM v_doctor_priority_queue LIMIT 10;
```

## 🔍 Monitoring

### Check Database Size

```sql
SELECT pg_size_pretty(pg_database_size('medical_portal'));
```

### Table Sizes

```sql
SELECT 
    tablename,
    pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size
FROM pg_tables
WHERE schemaname = 'public'
ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC;
```

### Active Connections

```sql
SELECT * FROM pg_stat_activity WHERE datname = 'medical_portal';
```

### Slow Queries

```sql
-- Enable slow query logging in postgresql.conf
log_min_duration_statement = 1000  -- Log queries taking > 1 second

-- View slow queries
SELECT query, mean_exec_time 
FROM pg_stat_statements 
ORDER BY mean_exec_time DESC 
LIMIT 10;
```

## 🛠️ Maintenance

### Vacuum and Analyze

```sql
-- Vacuum to reclaim storage
VACUUM medical_reports;

-- Analyze to update statistics
ANALYZE medical_reports;

-- Vacuum and analyze all tables
VACUUMDB -U postgres -d medical_portal --analyze
```

### Reindex

```sql
-- Reindex a table
REINDEX TABLE medical_reports;

-- Reindex entire database
REINDEX DATABASE medical_portal;
```

## 📞 Troubleshooting

### Connection Issues

```bash
# Check if PostgreSQL is running
sudo systemctl status postgresql

# Check PostgreSQL logs
sudo tail -f /var/log/postgresql/postgresql-14-main.log
```

### Permission Issues

```sql
-- Grant all permissions to user
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO medical_portal_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO medical_portal_app;
```

### Reset Database

```bash
# Drop and recreate database
dropdb medical_portal
createdb medical_portal
psql -U postgres -d medical_portal -f database/schema.sql
```

## 📚 Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/14/)
- [PostgreSQL Performance Tips](https://wiki.postgresql.org/wiki/Performance_Optimization)
- [HIPAA Compliance Guide](https://www.hhs.gov/hipaa/index.html)

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0.0 | 2025-10-02 | Initial schema with all core tables |

## 📧 Support

For database-related issues:
- Email: tech-support@medicalportal.com
- Documentation: https://docs.medicalportal.com/database