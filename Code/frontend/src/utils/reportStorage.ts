export interface MedicalReport {
  id: string;
  patientId: string;
  reportImage: string;
  riskScore: 'low' | 'medium' | 'high';
  uploadDate: string;
  uploadedBy: string;
  studyType?: string;
}

const STORAGE_KEY = 'medical_reports';

export function saveReport(report: Omit<MedicalReport, 'id'>): void {
  try {
    const reports = getAllReports();
    const newReport: MedicalReport = {
      ...report,
      id: `R${Date.now()}`
    };
    reports.push(newReport);
    
    // Limit to last 50 reports to prevent localStorage overflow
    if (reports.length > 50) {
      reports.splice(0, reports.length - 50);
    }
    
    localStorage.setItem(STORAGE_KEY, JSON.stringify(reports));
  } catch (error) {
    console.error('Failed to save report:', error);
    throw new Error('Failed to save report to storage');
  }
}

export function getAllReports(): MedicalReport[] {
  try {
    const data = localStorage.getItem(STORAGE_KEY);
    if (!data) return [];
    const parsed = JSON.parse(data);
    return Array.isArray(parsed) ? parsed : [];
  } catch (error) {
    console.error('Failed to load reports:', error);
    return [];
  }
}

export function getReportsByPatientId(patientId: string): MedicalReport[] {
  const allReports = getAllReports();
  return allReports.filter(report => report.patientId === patientId);
}

export function deleteReport(reportId: string): void {
  const reports = getAllReports();
  const filtered = reports.filter(report => report.id !== reportId);
  localStorage.setItem(STORAGE_KEY, JSON.stringify(filtered));
}
