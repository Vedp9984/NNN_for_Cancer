# Overall System Architecture

This document provides comprehensive architecture diagrams for the Medical Portal System.

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph "Frontend Layer"
        WebApp[React Web Application]
        Mobile[Mobile Responsive UI]
        Theme[Theme System Light/Dark]
    end
    
    subgraph "Presentation Layer"
        PatientUI[Patient Dashboard]
        RadioUI[Radiologist Portal]
        DoctorUI[Doctor Dashboard]
        TechUI[Tech Team Dashboard]
        ProfileUI[Profile/Settings]
    end
    
    subgraph "Authentication & Authorization"
        Auth[Auth Service]
        Session[Session Manager]
        RBAC[Role-Based Access Control]
    end
    
    subgraph "Business Logic Layer"
        ReportMgmt[Report Management]
        UserMgmt[User Management]
        NotifMgmt[Notification Manager]
        WorkflowEngine[Workflow Engine]
    end
    
    subgraph "AI/ML Services"
        ImageProc[Image Processor]
        AIInference[AI Inference Engine]
        RiskAssess[Risk Assessment]
        ModelMgmt[Model Management]
    end
    
    subgraph "Data Layer"
        UserDB[(User Database)]
        ReportDB[(Report Database)]
        ImageStore[(Image Storage)]
        ModelStore[(Model Storage)]
        MetricsDB[(Metrics Database)]
    end
    
    subgraph "External Services"
        EmailSvc[Email Service]
        SMSSvc[SMS Service]
        CloudStorage[Cloud Storage S3]
        Analytics[Analytics Service]
    end
    
    WebApp --> PatientUI
    WebApp --> RadioUI
    WebApp --> DoctorUI
    WebApp --> TechUI
    WebApp --> ProfileUI
    Mobile --> WebApp
    Theme --> WebApp
    
    PatientUI --> Auth
    RadioUI --> Auth
    DoctorUI --> Auth
    TechUI --> Auth
    ProfileUI --> Auth
    
    Auth --> Session
    Auth --> RBAC
    
    PatientUI --> ReportMgmt
    RadioUI --> ReportMgmt
    DoctorUI --> ReportMgmt
    TechUI --> ModelMgmt
    
    ReportMgmt --> WorkflowEngine
    ReportMgmt --> NotifMgmt
    
    RadioUI --> ImageProc
    ImageProc --> AIInference
    AIInference --> RiskAssess
    RiskAssess --> ReportMgmt
    
    TechUI --> ModelMgmt
    ModelMgmt --> AIInference
    
    UserMgmt --> UserDB
    ReportMgmt --> ReportDB
    ImageProc --> ImageStore
    ModelMgmt --> ModelStore
    AIInference --> MetricsDB
    
    NotifMgmt --> EmailSvc
    NotifMgmt --> SMSSvc
    ImageStore --> CloudStorage
    TechUI --> Analytics
    
    style WebApp fill:#3b82f6,stroke:#1e40af,color:#fff
    style AIInference fill:#10b981,stroke:#059669,color:#fff
    style Auth fill:#f59e0b,stroke:#d97706,color:#fff
    style ReportDB fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

## 2. Detailed Component Architecture

```mermaid
graph TB
    subgraph "Client Side - React Application"
        subgraph "Components"
            Login[Login/SignUp]
            Dashboard[Role-Based Dashboards]
            Profile[Profile Management]
            Settings[Settings & Security]
            ErrorBoundary[Error Boundary]
        end
        
        subgraph "State Management"
            AppState[App State]
            ThemeContext[Theme Context]
            UserContext[User Context]
        end
        
        subgraph "Utilities"
            ReportStorage[Report Storage Utils]
            AIRiskAssessment[AI Risk Assessment]
            ThemeUtils[Theme Utilities]
            ResponsiveHooks[Responsive Hooks]
        end
        
        subgraph "UI Components"
            ShadcnUI[Shadcn/UI Components]
            Tailwind[Tailwind CSS v4]
            LucideIcons[Lucide Icons]
        end
    end
    
    subgraph "Backend Services API"
        subgraph "API Gateway"
            Gateway[API Gateway/Load Balancer]
            RateLimiter[Rate Limiter]
            APIAuth[API Authentication]
        end
        
        subgraph "Microservices"
            AuthService[Authentication Service]
            UserService[User Service]
            ReportService[Report Service]
            AIService[AI Service]
            NotificationService[Notification Service]
            MetricsService[Metrics Service]
        end
    end
    
    subgraph "AI/ML Infrastructure"
        subgraph "Model Serving"
            ModelServer[TensorFlow Serving / ONNX Runtime]
            LoadBalancer[Model Load Balancer]
        end
        
        subgraph "Model Storage"
            ModelRegistry[Model Registry]
            VersionControl[Model Version Control]
        end
        
        subgraph "Training Pipeline"
            DataPipeline[Data Pipeline]
            TrainingCluster[Training Cluster GPU]
            ValidationService[Validation Service]
        end
    end
    
    subgraph "Data Storage"
        PostgreSQL[(PostgreSQL<br/>User & Report Data)]
        MongoDB[(MongoDB<br/>Unstructured Data)]
        Redis[(Redis<br/>Cache & Sessions)]
        S3[(AWS S3<br/>Image Storage)]
    end
    
    subgraph "Monitoring & Logging"
        Prometheus[Prometheus<br/>Metrics]
        Grafana[Grafana<br/>Dashboards]
        ELK[ELK Stack<br/>Logging]
        Sentry[Sentry<br/>Error Tracking]
    end
    
    Login --> AppState
    Dashboard --> AppState
    Profile --> AppState
    Settings --> ThemeContext
    ThemeContext --> Tailwind
    
    Dashboard --> ReportStorage
    ReportStorage --> AIRiskAssessment
    ResponsiveHooks --> Dashboard
    
    AppState --> Gateway
    Gateway --> RateLimiter
    RateLimiter --> APIAuth
    APIAuth --> AuthService
    APIAuth --> UserService
    APIAuth --> ReportService
    APIAuth --> AIService
    
    ReportService --> AIService
    AIService --> ModelServer
    ModelServer --> ModelRegistry
    
    AuthService --> PostgreSQL
    UserService --> PostgreSQL
    ReportService --> PostgreSQL
    ReportService --> MongoDB
    AIService --> MongoDB
    ReportService --> S3
    
    AuthService --> Redis
    APIAuth --> Redis
    
    AIService --> Prometheus
    ReportService --> Prometheus
    Prometheus --> Grafana
    
    ErrorBoundary --> Sentry
    
    style AIService fill:#10b981,stroke:#059669,color:#fff
    style ModelServer fill:#10b981,stroke:#059669,color:#fff
    style Gateway fill:#f59e0b,stroke:#d97706,color:#fff
    style PostgreSQL fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

## 3. AI Model Architecture

```mermaid
graph LR
    subgraph "Input Layer"
        Input[X-Ray Image<br/>512x512x1]
    end
    
    subgraph "Preprocessing"
        Resize[Resize & Normalize]
        Augment[Augmentation<br/>rotation, flip, zoom]
        Denoise[Denoising Filter]
    end
    
    subgraph "Feature Extraction - CNN"
        Conv1[Conv2D 32 filters<br/>3x3, ReLU]
        Pool1[MaxPooling 2x2]
        Conv2[Conv2D 64 filters<br/>3x3, ReLU]
        Pool2[MaxPooling 2x2]
        Conv3[Conv2D 128 filters<br/>3x3, ReLU]
        Pool3[MaxPooling 2x2]
        Conv4[Conv2D 256 filters<br/>3x3, ReLU]
        Pool4[MaxPooling 2x2]
    end
    
    subgraph "Classification Head"
        Flatten[Flatten Layer]
        Dense1[Dense 512<br/>ReLU + Dropout 0.5]
        Dense2[Dense 256<br/>ReLU + Dropout 0.3]
        Output[Dense 3<br/>Softmax<br/>Low/Medium/High]
    end
    
    subgraph "Post-Processing"
        Threshold[Threshold Application]
        Confidence[Confidence Calculation]
        Explanation[Generate Explanation]
    end
    
    Input --> Resize
    Resize --> Augment
    Augment --> Denoise
    Denoise --> Conv1
    
    Conv1 --> Pool1
    Pool1 --> Conv2
    Conv2 --> Pool2
    Pool2 --> Conv3
    Conv3 --> Pool3
    Pool3 --> Conv4
    Conv4 --> Pool4
    
    Pool4 --> Flatten
    Flatten --> Dense1
    Dense1 --> Dense2
    Dense2 --> Output
    
    Output --> Threshold
    Threshold --> Confidence
    Confidence --> Explanation
    
    style Input fill:#3b82f6,stroke:#1e40af,color:#fff
    style Conv1 fill:#10b981,stroke:#059669,color:#fff
    style Conv2 fill:#10b981,stroke:#059669,color:#fff
    style Conv3 fill:#10b981,stroke:#059669,color:#fff
    style Conv4 fill:#10b981,stroke:#059669,color:#fff
    style Output fill:#f59e0b,stroke:#d97706,color:#fff
```

## 4. Data Flow Architecture

```mermaid
flowchart TD
    Start([User Uploads X-Ray]) --> Validate{Valid Image?}
    Validate -->|No| Error[Return Error]
    Validate -->|Yes| Store[Store in S3]
    
    Store --> Queue[Add to Processing Queue]
    Queue --> Preprocess[Preprocess Image]
    
    Preprocess --> LoadModel[Load AI Model]
    LoadModel --> Inference[Run Inference]
    
    Inference --> RiskCalc[Calculate Risk Score]
    RiskCalc --> SaveResults[Save to Database]
    
    SaveResults --> CheckRisk{Risk Level?}
    
    CheckRisk -->|HIGH| UrgentNotif[Send Urgent Notifications<br/>Doctor + Patient]
    CheckRisk -->|MEDIUM| StandardNotif[Send Standard Notifications<br/>Patient + Doctor Queue]
    CheckRisk -->|LOW| BasicNotif[Send Basic Notification<br/>Patient Only]
    
    UrgentNotif --> UpdateMetrics[Update Model Metrics]
    StandardNotif --> UpdateMetrics
    BasicNotif --> UpdateMetrics
    
    UpdateMetrics --> LogEvent[Log Analytics Event]
    LogEvent --> End([Complete])
    
    Error --> End
    
    style Start fill:#3b82f6,stroke:#1e40af,color:#fff
    style Inference fill:#10b981,stroke:#059669,color:#fff
    style CheckRisk fill:#f59e0b,stroke:#d97706,color:#fff
    style End fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

## 5. Deployment Architecture

```mermaid
graph TB
    subgraph "Client Devices"
        Desktop[Desktop Browsers]
        Tablet[Tablets]
        Mobile[Mobile Devices]
    end
    
    subgraph "CDN & Load Balancer"
        CDN[CloudFlare CDN]
        ALB[Application Load Balancer]
    end
    
    subgraph "Web Tier - Kubernetes Cluster"
        WebPod1[React App Pod 1]
        WebPod2[React App Pod 2]
        WebPod3[React App Pod 3]
    end
    
    subgraph "API Tier - Kubernetes Cluster"
        APIPod1[API Service Pod 1]
        APIPod2[API Service Pod 2]
        APIPod3[API Service Pod 3]
    end
    
    subgraph "AI/ML Tier - GPU Nodes"
        AIPod1[AI Service Pod 1<br/>Tesla V100]
        AIPod2[AI Service Pod 2<br/>Tesla V100]
    end
    
    subgraph "Data Tier"
        PrimaryDB[(Primary PostgreSQL<br/>RDS)]
        ReplicaDB[(Read Replica<br/>PostgreSQL)]
        RedisCluster[(Redis Cluster)]
        S3Bucket[(S3 Bucket<br/>Images & Models)]
    end
    
    subgraph "Monitoring"
        Monitor[CloudWatch<br/>Prometheus<br/>Grafana]
    end
    
    Desktop --> CDN
    Tablet --> CDN
    Mobile --> CDN
    
    CDN --> ALB
    ALB --> WebPod1
    ALB --> WebPod2
    ALB --> WebPod3
    
    WebPod1 --> APIPod1
    WebPod2 --> APIPod2
    WebPod3 --> APIPod3
    
    APIPod1 --> AIPod1
    APIPod2 --> AIPod2
    APIPod3 --> AIPod1
    
    APIPod1 --> PrimaryDB
    APIPod2 --> ReplicaDB
    APIPod3 --> ReplicaDB
    
    APIPod1 --> RedisCluster
    APIPod2 --> RedisCluster
    APIPod3 --> RedisCluster
    
    AIPod1 --> S3Bucket
    AIPod2 --> S3Bucket
    APIPod1 --> S3Bucket
    
    WebPod1 --> Monitor
    APIPod1 --> Monitor
    AIPod1 --> Monitor
    
    style AIPod1 fill:#10b981,stroke:#059669,color:#fff
    style AIPod2 fill:#10b981,stroke:#059669,color:#fff
    style PrimaryDB fill:#8b5cf6,stroke:#7c3aed,color:#fff
```

## 6. Security Architecture

```mermaid
graph TB
    subgraph "External"
        User[Users]
        Attacker[Potential Threats]
    end
    
    subgraph "Security Perimeter"
        WAF[Web Application Firewall]
        DDoS[DDoS Protection]
        SSL[SSL/TLS Encryption]
    end
    
    subgraph "Authentication Layer"
        OAuth[OAuth 2.0]
        JWT[JWT Tokens]
        MFA[Multi-Factor Auth]
        SessionMgmt[Session Management]
    end
    
    subgraph "Authorization Layer"
        RBAC2[Role-Based Access Control]
        PermCheck[Permission Checker]
        AuditLog[Audit Logging]
    end
    
    subgraph "Application Security"
        InputVal[Input Validation]
        XSSProt[XSS Protection]
        CSRFProt[CSRF Protection]
        SQLInject[SQL Injection Prevention]
    end
    
    subgraph "Data Security"
        Encryption[Data Encryption at Rest]
        Anonymize[Data Anonymization]
        Backup[Encrypted Backups]
        KeyMgmt[Key Management Service]
    end
    
    subgraph "Network Security"
        VPC[Virtual Private Cloud]
        PrivateSubnet[Private Subnets]
        SecurityGroups[Security Groups]
        NetworkACL[Network ACLs]
    end
    
    User --> WAF
    Attacker --> WAF
    WAF --> DDoS
    DDoS --> SSL
    
    SSL --> OAuth
    OAuth --> JWT
    JWT --> MFA
    MFA --> SessionMgmt
    
    SessionMgmt --> RBAC2
    RBAC2 --> PermCheck
    PermCheck --> AuditLog
    
    AuditLog --> InputVal
    InputVal --> XSSProt
    XSSProt --> CSRFProt
    CSRFProt --> SQLInject
    
    SQLInject --> Encryption
    Encryption --> Anonymize
    Anonymize --> Backup
    Backup --> KeyMgmt
    
    KeyMgmt --> VPC
    VPC --> PrivateSubnet
    PrivateSubnet --> SecurityGroups
    SecurityGroups --> NetworkACL
    
    style WAF fill:#f59e0b,stroke:#d97706,color:#fff
    style Encryption fill:#8b5cf6,stroke:#7c3aed,color:#fff
    style RBAC2 fill:#3b82f6,stroke:#1e40af,color:#fff
```

## Key Architectural Principles

### 1. **Scalability**
- Horizontal scaling through Kubernetes pods
- Load balancing across multiple instances
- Caching layer with Redis for performance
- CDN for static asset delivery

### 2. **Reliability**
- Database replication for high availability
- Auto-scaling based on load
- Health checks and automatic recovery
- Data backup and disaster recovery

### 3. **Security**
- Multi-layer security (WAF, SSL, Auth, RBAC)
- Data encryption at rest and in transit
- Regular security audits
- HIPAA compliance ready

### 4. **Performance**
- GPU acceleration for AI inference
- Caching strategies
- Optimized database queries
- Lazy loading and code splitting

### 5. **Maintainability**
- Microservices architecture
- Clear separation of concerns
- Comprehensive monitoring and logging
- CI/CD pipeline for deployments

## Technology Stack

### Frontend
- React 18+ with TypeScript
- Tailwind CSS v4
- Shadcn/UI components
- Lucide Icons
- Motion/React for animations

### Backend
- Node.js / Python FastAPI
- PostgreSQL (relational data)
- MongoDB (unstructured data)
- Redis (caching & sessions)
- AWS S3 (file storage)

### AI/ML
- TensorFlow / PyTorch
- ONNX Runtime for inference
- OpenCV for image processing
- NumPy / Pandas for data processing

### Infrastructure
- Kubernetes for orchestration
- Docker for containerization
- AWS / GCP / Azure for cloud
- Terraform for IaC
- GitHub Actions for CI/CD

### Monitoring
- Prometheus for metrics
- Grafana for dashboards
- ELK Stack for logging
- Sentry for error tracking