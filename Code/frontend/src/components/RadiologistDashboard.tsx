import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Badge } from './ui/badge';
import { Alert, AlertDescription } from './ui/alert';
import { Upload, Stethoscope, FileText, Check, Image as ImageIcon, Loader2, Brain, AlertCircle } from 'lucide-react';
import { User as UserType } from '../App';
import { saveReport, getAllReports, MedicalReport } from '../utils/reportStorage';
import { analyzeImageRisk, RiskAssessmentResult } from '../utils/aiRiskAssessment';
import { toast } from 'sonner@2.0.3';
import { UserDropdown } from './UserDropdown';
import { useTheme } from '../utils/themeContext';

interface RadiologistDashboardProps {
  user: UserType;
  onLogout: () => void;
  onNavigate: (destination: 'dashboard' | 'profile' | 'settings') => void;
}

export function RadiologistDashboard({ user, onLogout, onNavigate }: RadiologistDashboardProps) {
  const { actualTheme } = useTheme();
  const isDarkMode = actualTheme === 'dark';
  const [patientId, setPatientId] = useState('');
  const [studyType, setStudyType] = useState('');
  const [reportImage, setReportImage] = useState<string>('');
  const [imageFileName, setImageFileName] = useState('');
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [aiAssessment, setAiAssessment] = useState<RiskAssessmentResult | null>(null);
  const [uploadedReports, setUploadedReports] = useState<MedicalReport[]>(getAllReports());

  const handleImageUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      // Validate file size (max 5MB)
      if (file.size > 5 * 1024 * 1024) {
        toast.error('File size too large. Please select an image under 5MB.');
        return;
      }

      setImageFileName(file.name);
      setIsAnalyzing(true);
      setAiAssessment(null);

      const reader = new FileReader();
      reader.onloadend = async () => {
        const imageData = reader.result as string;
        setReportImage(imageData);

        try {
          // Analyze image with AI model with timeout
          toast.info('AI model analyzing image...');
          const assessment = await analyzeImageRisk(imageData);
          setAiAssessment(assessment);
          toast.success(`AI analysis complete: ${assessment.riskLevel} risk detected`);
        } catch (error) {
          console.error('AI analysis failed:', error);
          toast.error('AI analysis failed, using fallback assessment');
          
          // Fallback assessment
          setAiAssessment({
            riskLevel: 'low',
            confidence: 0.7,
            findings: ['Fallback assessment - manual review recommended'],
            recommendedAction: 'Manual radiologist review recommended due to AI analysis timeout'
          });
        } finally {
          setIsAnalyzing(false);
        }
      };

      reader.readAsDataURL(file);
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!patientId || !reportImage || !aiAssessment) {
      toast.error('Please fill in all required fields and upload an image');
      return;
    }

    try {
      const report: Omit<MedicalReport, 'id'> = {
        patientId,
        studyType: studyType || 'X-Ray',
        reportImage,
        riskScore: aiAssessment.riskLevel,
        uploadedBy: user.name,
        uploadDate: new Date().toISOString(),
        findings: aiAssessment.findings,
        confidence: aiAssessment.confidence
      };

      const savedReport = saveReport(report);
      setUploadedReports(prev => [savedReport, ...prev]);
      
      // Reset form
      setPatientId('');
      setStudyType('');
      setReportImage('');
      setImageFileName('');
      setAiAssessment(null);
      
      // Reset file input
      const fileInput = document.getElementById('report-image') as HTMLInputElement;
      if (fileInput) fileInput.value = '';

      toast.success('Report uploaded successfully!');
    } catch (error) {
      console.error('Failed to save report:', error);
      toast.error('Failed to save report. Please try again.');
    }
  };

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'}`}>
      {/* Header */}
      <header className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b px-4 sm:px-6 py-3`}>
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="h-8 w-8 bg-green-600 rounded-full flex items-center justify-center flex-shrink-0">
              <Stethoscope className="h-4 w-4 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className={`font-semibold truncate text-base sm:text-lg ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Radiologist Portal</h1>
              <p className={`text-sm truncate ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Welcome, {user.name} (ID: {user.id})</p>
            </div>
          </div>
          <UserDropdown user={user} onLogout={onLogout} onNavigate={onNavigate} isDarkMode={isDarkMode} />
        </div>
      </header>

      <div className="p-3 sm:p-4 max-w-7xl mx-auto">
        {/* Quick Stats for Radiologist */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 sm:gap-4 mb-6">
          <Card className={isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}>
            <CardContent className="p-3 sm:p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Reports Uploaded</p>
                  <p className={`text-lg sm:text-2xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>{uploadedReports.length}</p>
                </div>
                <FileText className="h-6 w-6 sm:h-8 sm:w-8 text-green-500" />
              </div>
            </CardContent>
          </Card>

          <Card className={isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}>
            <CardContent className="p-3 sm:p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Today's Reports</p>
                  <p className={`text-lg sm:text-2xl font-semibold ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                    {uploadedReports.filter(r => new Date(r.uploadDate).toDateString() === new Date().toDateString()).length}
                  </p>
                </div>
                <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-blue-400" />
              </div>
            </CardContent>
          </Card>

          <Card className={isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}>
            <CardContent className="p-3 sm:p-4">
              <div className="flex items-center justify-between">
                <div>
                  <p className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>High Risk Cases</p>
                  <p className="text-lg sm:text-2xl font-semibold text-red-400">
                    {uploadedReports.filter(r => r.riskScore === 'high').length}
                  </p>
                </div>
                <AlertCircle className="h-6 w-6 sm:h-8 sm:w-8 text-red-500" />
              </div>
            </CardContent>
          </Card>
        </div>

        <Tabs defaultValue="upload" className="space-y-3 sm:space-y-4">
          <TabsList className={`grid w-full grid-cols-2 h-auto ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-gray-100 border-gray-200'}`}>
            <TabsTrigger 
              value="upload" 
              className={`text-xs sm:text-sm py-2 px-2 sm:px-4 ${
                isDarkMode 
                  ? 'text-gray-300 data-[state=active]:bg-green-600 data-[state=active]:text-white' 
                  : 'text-gray-600 data-[state=active]:bg-green-600 data-[state=active]:text-white'
              }`}
            >
              <Upload className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">Upload Report</span>
              <span className="sm:hidden">Upload</span>
            </TabsTrigger>
            <TabsTrigger 
              value="reports" 
              className={`text-xs sm:text-sm py-2 px-2 sm:px-4 ${
                isDarkMode 
                  ? 'text-gray-300 data-[state=active]:bg-green-600 data-[state=active]:text-white' 
                  : 'text-gray-600 data-[state=active]:bg-green-600 data-[state=active]:text-white'
              }`}
            >
              <FileText className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">My Reports</span>
              <span className="sm:hidden">Reports</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="upload" className="space-y-4 sm:space-y-6">
            <Card className={isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}>
              <CardHeader className="pb-4">
                <CardTitle className={`text-lg sm:text-xl ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Upload New Report</CardTitle>
                <CardDescription className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Upload patient X-ray images for AI-powered risk assessment
                </CardDescription>
              </CardHeader>
              <CardContent>
                <form onSubmit={handleSubmit} className="space-y-4 sm:space-y-6">
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                    <div className="space-y-2">
                      <Label htmlFor="patient-id" className={`text-sm font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                        Patient ID <span className="text-red-400">*</span>
                      </Label>
                      <Input
                        id="patient-id"
                        placeholder="Enter patient ID"
                        value={patientId}
                        onChange={(e) => setPatientId(e.target.value)}
                        required
                        className={`w-full ${
                          isDarkMode 
                            ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                            : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                        }`}
                      />
                    </div>
                    <div className="space-y-2">
                      <Label htmlFor="study-type" className={`text-sm font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>Study Type</Label>
                      <Input
                        id="study-type"
                        placeholder="e.g., Chest X-Ray, Abdomen X-Ray"
                        value={studyType}
                        onChange={(e) => setStudyType(e.target.value)}
                        className={`w-full ${
                          isDarkMode 
                            ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                            : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
                        }`}
                      />
                    </div>
                  </div>

                  <div className="space-y-2">
                    <Label htmlFor="report-image" className={`text-sm font-medium ${isDarkMode ? 'text-gray-200' : 'text-gray-700'}`}>
                      Report Image <span className="text-red-400">*</span>
                    </Label>
                    <div className={`border-2 border-dashed rounded-lg p-4 sm:p-6 text-center transition-colors ${
                      isDarkMode 
                        ? 'border-gray-600 bg-gray-700 hover:border-gray-500' 
                        : 'border-gray-300 bg-gray-50 hover:border-gray-400'
                    }`}>
                      <input
                        id="report-image"
                        type="file"
                        accept="image/*"
                        onChange={handleImageUpload}
                        className="hidden"
                        required
                      />
                      <label
                        htmlFor="report-image"
                        className="cursor-pointer flex flex-col items-center gap-2"
                      >
                        {isAnalyzing ? (
                          <>
                            <Loader2 className="h-8 w-8 sm:h-12 sm:w-12 text-blue-500 animate-spin" />
                            <div className="text-sm sm:text-base font-medium">AI Analyzing Image...</div>
                            <div className="text-xs sm:text-sm text-muted-foreground">
                              This may take a few moments
                            </div>
                          </>
                        ) : reportImage ? (
                          <>
                            <Check className="h-8 w-8 sm:h-12 sm:w-12 text-green-500" />
                            <div className={`text-sm sm:text-base font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Image Uploaded</div>
                            <div className={`text-xs sm:text-sm truncate max-w-full ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              {imageFileName}
                            </div>
                            <Button 
                              type="button" 
                              variant="outline" 
                              size="sm" 
                              className={`mt-2 ${
                                isDarkMode 
                                  ? 'border-gray-600 text-gray-200 hover:bg-gray-600' 
                                  : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                              }`}
                            >
                              Change Image
                            </Button>
                          </>
                        ) : (
                          <>
                            <Upload className="h-8 w-8 sm:h-12 sm:w-12 text-gray-400" />
                            <div className={`text-sm sm:text-base font-medium ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Upload Report Image</div>
                            <div className={`text-xs sm:text-sm ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              Click to select or drag and drop
                            </div>
                            <div className={`text-xs mt-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              PNG, JPG up to 5MB
                            </div>
                          </>
                        )}
                      </label>
                    </div>
                  </div>

                  {/* AI Assessment Results */}
                  {aiAssessment && (
                    <Alert className={
                      isDarkMode 
                        ? 'border-blue-600 bg-blue-900/20 text-white' 
                        : 'border-blue-200 bg-blue-50 text-blue-900'
                    }>
                      <Brain className="h-4 w-4 text-blue-400" />
                      <AlertDescription>
                        <div className="space-y-2">
                          <div className="flex flex-col sm:flex-row sm:items-center gap-2">
                            <strong className={`text-sm ${isDarkMode ? 'text-white' : 'text-blue-900'}`}>AI Risk Assessment:</strong>
                            <div className="flex items-center gap-2 flex-wrap">
                              <Badge className={getRiskColor(aiAssessment.riskLevel)}>
                                {aiAssessment.riskLevel.toUpperCase()} RISK
                              </Badge>
                              <span className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-blue-700'}`}>
                                Confidence: {Math.round(aiAssessment.confidence * 100)}%
                              </span>
                            </div>
                          </div>
                          
                          {aiAssessment.findings.length > 0 && (
                            <div className="text-sm">
                              <strong className={isDarkMode ? 'text-white' : 'text-blue-900'}>Key Findings:</strong>
                              <ul className="list-disc list-inside mt-1 space-y-1 text-xs sm:text-sm">
                                {aiAssessment.findings.map((finding, index) => (
                                  <li key={index} className={isDarkMode ? 'text-gray-300' : 'text-blue-800'}>{finding}</li>
                                ))}
                              </ul>
                            </div>
                          )}
                          
                          <div className="text-xs sm:text-sm">
                            <strong className={isDarkMode ? 'text-white' : 'text-blue-900'}>Recommended Action:</strong>
                            <p className={`mt-1 ${isDarkMode ? 'text-gray-300' : 'text-blue-800'}`}>{aiAssessment.recommendedAction}</p>
                          </div>
                        </div>
                      </AlertDescription>
                    </Alert>
                  )}

                  <Button 
                    type="submit" 
                    className="w-full sm:w-auto bg-green-600 hover:bg-green-700"
                    disabled={isAnalyzing || !patientId || !reportImage || !aiAssessment}
                  >
                    {isAnalyzing ? (
                      <>
                        <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                        Processing...
                      </>
                    ) : (
                      <>
                        <Check className="h-4 w-4 mr-2" />
                        Submit Report
                      </>
                    )}
                  </Button>
                </form>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="reports" className="space-y-4 sm:space-y-6">
            <Card className={isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'}>
              <CardHeader className="pb-4">
                <CardTitle className={`text-lg sm:text-xl ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Uploaded Reports</CardTitle>
                <CardDescription className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Recent reports you've uploaded with AI assessments
                </CardDescription>
              </CardHeader>
              <CardContent>
                {uploadedReports.length === 0 ? (
                  <div className={`text-center py-8 sm:py-12 ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                    <FileText className="h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm sm:text-base">No reports uploaded yet</p>
                    <p className="text-xs sm:text-sm mt-1">Your uploaded reports will appear here</p>
                  </div>
                ) : (
                  <div className="space-y-4">
                    {uploadedReports.map((report) => (
                      <div key={report.id} className={`flex flex-col sm:flex-row items-start sm:items-center gap-4 p-3 sm:p-4 border rounded-lg ${
                        isDarkMode 
                          ? 'border-gray-700 bg-gray-700/50' 
                          : 'border-gray-200 bg-gray-50'
                      }`}>
                        <div className="flex items-center gap-3 sm:gap-4 flex-1 w-full">
                          <div className={`h-16 w-16 sm:h-20 sm:w-20 rounded flex items-center justify-center overflow-hidden flex-shrink-0 ${
                            isDarkMode ? 'bg-gray-600' : 'bg-gray-200'
                          }`}>
                            <img
                              src={report.reportImage}
                              alt="Report thumbnail"
                              className="h-full w-full object-cover"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 mb-2">
                              <h3 className={`font-medium truncate text-sm sm:text-base ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
                                Patient: {report.patientId}
                              </h3>
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge className={getRiskColor(report.riskScore)} size="sm">
                                  {report.riskScore.toUpperCase()} RISK
                                </Badge>
                                <Badge 
                                  variant="outline" 
                                  className={`gap-1 text-xs ${
                                    isDarkMode 
                                      ? 'border-gray-600 text-gray-300' 
                                      : 'border-gray-300 text-gray-600'
                                  }`}
                                >
                                  <Brain className="h-3 w-3" />
                                  AI Assessed
                                </Badge>
                              </div>
                            </div>
                            <div className={`text-xs sm:text-sm space-y-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                              <p className="truncate">
                                Study: {report.studyType || 'X-Ray'} • 
                                Confidence: {report.confidence ? Math.round(report.confidence * 100) : 'N/A'}%
                              </p>
                              <p className="truncate">
                                Uploaded on {new Date(report.uploadDate).toLocaleDateString()}{' '}
                                <span className="hidden sm:inline">
                                  at {new Date(report.uploadDate).toLocaleTimeString()}
                                </span>
                              </p>
                            </div>
                          </div>
                        </div>
                        <div className="flex items-center gap-2 w-full sm:w-auto">
                          <Button 
                            variant="outline" 
                            size="sm"
                            className={`flex-1 sm:flex-none ${
                              isDarkMode 
                                ? 'border-gray-600 text-gray-200 hover:bg-gray-600' 
                                : 'border-gray-300 text-gray-700 hover:bg-gray-100'
                            }`}
                          >
                            <ImageIcon className="h-4 w-4 sm:mr-2" />
                            <span className="text-xs sm:text-sm">View</span>
                          </Button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>
    </div>
  );
}