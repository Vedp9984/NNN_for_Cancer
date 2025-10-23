# Sequence Diagrams

This document contains sequence diagrams for key workflows in the Medical Portal System.

## 1. Patient Login and View Reports

```mermaid
sequenceDiagram
    actor Patient
    participant LoginUI
    participant AuthService
    participant Session
    participant PatientDashboard
    participant ReportStorage
    
    Patient->>LoginUI: Enter credentials
    LoginUI->>AuthService: authenticate(email, password)
    AuthService->>AuthService: validateCredentials()
    AuthService->>Session: createSession(userId)
    Session-->>AuthService: sessionToken
    AuthService-->>LoginUI: authSuccess + token
    LoginUI->>PatientDashboard: navigateToDashboard(user)
    
    PatientDashboard->>ReportStorage: getReportsByPatientId(patientId)
    ReportStorage-->>PatientDashboard: reports[]
    PatientDashboard->>PatientDashboard: calculateRiskSummary()
    PatientDashboard-->>Patient: Display reports and risk scores
    
    Patient->>PatientDashboard: Click "View Report"
    PatientDashboard->>ReportStorage: getReportDetails(reportId)
    ReportStorage-->>PatientDashboard: reportData + aiAnalysis
    PatientDashboard-->>Patient: Display full report with recommendations
```

## 2. Radiologist Upload X-Ray Report (AI-Powered)

```mermaid
sequenceDiagram
    actor Radiologist
    participant RadiologistUI
    participant ImageUploader
    participant ImageProcessor
    participant AIModel
    participant RiskEngine
    participant ReportStorage
    participant NotificationService
    participant Doctor
    participant Patient
    
    Radiologist->>RadiologistUI: Enter Patient ID
    Radiologist->>RadiologistUI: Upload X-Ray Image
    RadiologistUI->>ImageUploader: uploadImage(patientId, imageFile)
    
    ImageUploader->>ImageUploader: validateImage()
    ImageUploader->>ImageProcessor: preprocessImage(image)
    
    Note over ImageProcessor: Resize, normalize,<br/>enhance contrast
    ImageProcessor->>ImageProcessor: resize(image, 512x512)
    ImageProcessor->>ImageProcessor: normalize(image)
    ImageProcessor->>ImageProcessor: denoise(image)
    ImageProcessor-->>ImageUploader: processedImage
    
    ImageUploader->>AIModel: predict(processedImage)
    
    Note over AIModel: Deep learning inference<br/>Feature extraction<br/>Risk calculation
    AIModel->>AIModel: extractFeatures(image)
    AIModel->>AIModel: forwardPass(features)
    AIModel->>RiskEngine: calculateRisk(predictions)
    
    RiskEngine->>RiskEngine: categorizeRisk(score)
    RiskEngine->>RiskEngine: calculateConfidence()
    RiskEngine->>RiskEngine: generateRecommendations()
    RiskEngine-->>AIModel: riskAnalysis
    
    AIModel-->>ImageUploader: aiAnalysisResult
    
    ImageUploader->>ReportStorage: createReport(patientId, image, analysis)
    ReportStorage->>ReportStorage: saveReport()
    ReportStorage->>ReportStorage: saveAIAnalysis()
    ReportStorage-->>ImageUploader: reportId
    
    ImageUploader->>NotificationService: notifyStakeholders(report)
    
    alt Risk Score is HIGH
        NotificationService->>Doctor: sendUrgentAlert(report)
        NotificationService->>Patient: sendNotification(report)
        Doctor-->>NotificationService: alertReceived
    else Risk Score is MEDIUM or LOW
        NotificationService->>Patient: sendNotification(report)
    end
    
    ImageUploader-->>RadiologistUI: uploadSuccess(reportId, riskScore)
    RadiologistUI-->>Radiologist: Display success + risk score
```

## 3. Doctor Review AI-Generated Report

```mermaid
sequenceDiagram
    actor Doctor
    participant DoctorUI
    participant ReportStorage
    participant AIAnalysis
    participant DoctorReview
    participant NotificationService
    participant Patient
    
    Doctor->>DoctorUI: Login to Dashboard
    DoctorUI->>ReportStorage: getPriorityReports(doctorId)
    ReportStorage-->>DoctorUI: prioritizedReports[]
    
    Note over DoctorUI: Reports sorted by<br/>risk score (HIGH first)
    DoctorUI-->>Doctor: Display worklist
    
    Doctor->>DoctorUI: Select report to review
    DoctorUI->>ReportStorage: getReportDetails(reportId)
    ReportStorage->>AIAnalysis: getAnalysis(reportId)
    AIAnalysis-->>ReportStorage: aiAnalysis
    ReportStorage-->>DoctorUI: fullReportData
    
    DoctorUI-->>Doctor: Display X-Ray + AI Analysis
    
    Doctor->>Doctor: Review findings
    Doctor->>DoctorUI: Add clinical notes
    Doctor->>DoctorUI: Confirm/Modify AI diagnosis
    Doctor->>DoctorUI: Set urgency level
    Doctor->>DoctorUI: Add recommended actions
    
    DoctorUI->>DoctorReview: createReview(reviewData)
    DoctorReview->>DoctorReview: validate()
    DoctorReview->>ReportStorage: saveReview(review)
    ReportStorage->>ReportStorage: updateReportStatus("reviewed")
    ReportStorage-->>DoctorReview: reviewSaved
    
    DoctorReview->>NotificationService: notifyPatient(review)
    NotificationService->>Patient: sendNotification(review)
    Patient-->>NotificationService: notificationDelivered
    
    DoctorReview-->>DoctorUI: reviewComplete
    DoctorUI-->>Doctor: Display success message
```

## 4. Tech Team Monitor AI Model Performance

```mermaid
sequenceDiagram
    actor TechTeam
    participant TechUI
    participant ModelRegistry
    participant MetricsCollector
    participant ModelMetrics
    participant DataRepository
    participant AIModel
    
    TechTeam->>TechUI: Access Tech Dashboard
    TechUI->>ModelRegistry: getActiveModels()
    ModelRegistry-->>TechUI: models[]
    
    TechUI->>MetricsCollector: getLatestMetrics(modelId)
    MetricsCollector->>ModelMetrics: query(modelId, timeRange)
    ModelMetrics-->>MetricsCollector: metrics[]
    MetricsCollector-->>TechUI: performanceData
    
    Note over TechUI: Display accuracy,<br/>precision, recall,<br/>F1 score trends
    TechUI-->>TechTeam: Show performance dashboard
    
    TechTeam->>TechUI: View prediction history
    TechUI->>DataRepository: queryPredictions(filters)
    DataRepository-->>TechUI: predictions[]
    TechUI-->>TechTeam: Display predictions + outcomes
    
    alt Model performance degraded
        TechTeam->>TechUI: Trigger model retraining
        TechUI->>AIModel: initiateTraining(dataset)
        AIModel->>AIModel: loadTrainingData()
        AIModel->>AIModel: train(epochs=100)
        AIModel->>AIModel: validate(validationSet)
        AIModel->>ModelMetrics: saveTrainingMetrics()
        AIModel-->>TechUI: trainingComplete
        
        TechTeam->>TechUI: Deploy new model version
        TechUI->>ModelRegistry: deployModel(newModel)
        ModelRegistry->>ModelRegistry: validateModel()
        ModelRegistry->>ModelRegistry: updateActiveModel()
        ModelRegistry-->>TechUI: deploymentSuccess
        TechUI-->>TechTeam: New model deployed
    end
```

## 5. Complete User Journey: From Upload to Treatment

```mermaid
sequenceDiagram
    actor Radiologist
    actor Patient
    actor Doctor
    participant System
    participant AIModel
    participant NotificationService
    
    Note over Radiologist,NotificationService: Phase 1: Report Upload & AI Analysis
    Radiologist->>System: Upload X-Ray (PatientID + Image)
    System->>AIModel: Analyze Image
    AIModel-->>System: Risk Score + Analysis
    System->>NotificationService: Trigger Notifications
    
    alt HIGH Risk
        NotificationService->>Patient: Urgent notification
        NotificationService->>Doctor: Urgent alert
        NotificationService->>Radiologist: Upload confirmed (HIGH)
    else MEDIUM Risk
        NotificationService->>Patient: Standard notification
        NotificationService->>Doctor: Follow-up required
        NotificationService->>Radiologist: Upload confirmed (MEDIUM)
    else LOW Risk
        NotificationService->>Patient: Results available
        NotificationService->>Radiologist: Upload confirmed (LOW)
    end
    
    Note over Patient,Doctor: Phase 2: Patient Views Results
    Patient->>System: Login and view dashboard
    System-->>Patient: Display report + risk score
    Patient->>System: View detailed report
    System-->>Patient: Show AI analysis + recommendations
    
    Note over Doctor,NotificationService: Phase 3: Doctor Review
    Doctor->>System: Access priority queue
    System-->>Doctor: HIGH risk cases first
    Doctor->>System: Review AI analysis
    Doctor->>System: Add clinical review
    System->>NotificationService: Review complete
    NotificationService->>Patient: Doctor reviewed your report
    
    Note over Patient,Doctor: Phase 4: Follow-up Action
    Patient->>Doctor: Schedule appointment (if needed)
    Doctor-->>Patient: Appointment confirmed
    Doctor->>System: Update patient care plan
    System-->>Patient: Treatment plan available
```

## 6. AI Model Prediction Pipeline (Detailed)

```mermaid
sequenceDiagram
    participant Upload
    participant Validator
    participant Preprocessor
    participant FeatureExtractor
    participant ModelInference
    participant PostProcessor
    participant RiskCalculator
    participant Storage
    
    Upload->>Validator: validateImage(file)
    Validator->>Validator: checkFormat()
    Validator->>Validator: checkSize()
    Validator->>Validator: checkQuality()
    Validator-->>Upload: validationResult
    
    Upload->>Preprocessor: preprocess(image)
    
    Note over Preprocessor: Step 1: Resize to 512x512
    Preprocessor->>Preprocessor: resize(image, 512, 512)
    
    Note over Preprocessor: Step 2: Normalize pixel values
    Preprocessor->>Preprocessor: normalize(0-255 → 0-1)
    
    Note over Preprocessor: Step 3: Enhance contrast
    Preprocessor->>Preprocessor: contrastEnhancement()
    
    Note over Preprocessor: Step 4: Denoise
    Preprocessor->>Preprocessor: gaussianBlur()
    
    Preprocessor-->>Upload: processedImage
    
    Upload->>FeatureExtractor: extractFeatures(processedImage)
    FeatureExtractor->>FeatureExtractor: applyConvLayers()
    FeatureExtractor->>FeatureExtractor: pooling()
    FeatureExtractor->>FeatureExtractor: flatten()
    FeatureExtractor-->>Upload: featureVector
    
    Upload->>ModelInference: predict(features)
    ModelInference->>ModelInference: forwardPass()
    ModelInference->>ModelInference: applyActivation()
    ModelInference-->>Upload: rawPredictions
    
    Upload->>PostProcessor: postprocess(predictions)
    PostProcessor->>PostProcessor: applySoftmax()
    PostProcessor->>PostProcessor: getProbabilities()
    PostProcessor-->>Upload: probabilities
    
    Upload->>RiskCalculator: calculateRisk(probabilities)
    RiskCalculator->>RiskCalculator: categorize(thresholds)
    
    Note over RiskCalculator: HIGH: >0.7<br/>MEDIUM: 0.4-0.7<br/>LOW: <0.4
    
    RiskCalculator->>RiskCalculator: generateExplanation()
    RiskCalculator-->>Upload: riskAssessment
    
    Upload->>Storage: saveAnalysis(assessment)
    Storage-->>Upload: analysisId
```

## Key Workflows Summary

1. **Patient Flow**: Login → View Dashboard → Check Reports → Read AI Analysis → Contact Doctor
2. **Radiologist Flow**: Login → Enter Patient ID → Upload X-Ray → AI Auto-Analysis → Confirmation
3. **Doctor Flow**: Login → Priority Queue → Review AI Analysis → Add Clinical Notes → Approve/Modify
4. **Tech Team Flow**: Login → Monitor Metrics → Analyze Performance → Retrain/Deploy Models
5. **AI Pipeline**: Upload → Validate → Preprocess → Extract Features → Predict → Calculate Risk → Store

## Timing Considerations

- **Image Upload**: ~2-5 seconds
- **AI Analysis**: ~5-10 seconds
- **Risk Calculation**: ~1-2 seconds
- **Notification Delivery**: ~1-3 seconds
- **Total Time (Upload to Notification)**: ~10-20 seconds

## Error Handling

Each workflow includes error handling for:
- Authentication failures
- Invalid image formats
- AI model errors
- Database connection issues
- Network timeouts
- Insufficient permissions