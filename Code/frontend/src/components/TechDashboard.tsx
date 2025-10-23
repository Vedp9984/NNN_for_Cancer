import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Progress } from './ui/progress';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { Database, Settings, BarChart3, Brain, Server, Activity, TrendingUp, Users, FileText, AlertTriangle, CheckCircle, X } from 'lucide-react';
import { User as UserType } from '../App';
import { getAllReports, MedicalReport } from '../utils/reportStorage';
import { UserDropdown } from './UserDropdown';
import { useTheme } from '../utils/themeContext';

interface TechDashboardProps {
  user: UserType;
  onLogout: () => void;
  onNavigate: (destination: 'dashboard' | 'profile' | 'settings') => void;
}

interface ModelMetrics {
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  totalPredictions: number;
  correctPredictions: number;
  lastUpdated: string;
}

export function TechDashboard({ user, onLogout, onNavigate }: TechDashboardProps) {
  const { actualTheme } = useTheme();
  const isDarkMode = actualTheme === 'dark';
  const [reports, setReports] = useState<MedicalReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<MedicalReport | null>(null);
  const [modelMetrics, setModelMetrics] = useState<ModelMetrics>({
    accuracy: 92.5,
    precision: 89.2,
    recall: 94.1,
    f1Score: 91.6,
    totalPredictions: 1247,
    correctPredictions: 1154,
    lastUpdated: new Date().toISOString()
  });

  useEffect(() => {
    const allReports = getAllReports();
    setReports(allReports.sort((a, b) => new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime()));
  }, []);

  const today = new Date().toDateString();
  const thisWeek = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000);
  
  const systemStats = {
    total: reports.length,
    todayReports: reports.filter(r => new Date(r.uploadDate).toDateString() === today).length,
    weekReports: reports.filter(r => new Date(r.uploadDate) >= thisWeek).length,
    riskDistribution: {
      high: reports.filter(r => r.riskScore === 'high').length,
      medium: reports.filter(r => r.riskScore === 'medium').length,
      low: reports.filter(r => r.riskScore === 'low').length
    },
    avgConfidence: reports.length > 0 
      ? Math.round((reports.reduce((sum, r) => sum + (r.confidence || 0), 0) / reports.length) * 100)
      : 0,
    systemHealth: 98.5,
    processingTime: 2.3
  };

  const getConfidenceColor = (confidence: number) => {
    if (confidence >= 90) return 'text-green-600';
    if (confidence >= 75) return 'text-yellow-600';
    return 'text-red-600';
  };

  const getSystemStatus = (health: number) => {
    if (health >= 95) return { status: 'Excellent', color: 'text-green-600', icon: CheckCircle };
    if (health >= 85) return { status: 'Good', color: 'text-yellow-600', icon: AlertTriangle };
    return { status: 'Needs Attention', color: 'text-red-600', icon: AlertTriangle };
  };

  const systemStatus = getSystemStatus(systemStats.systemHealth);

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white border-b px-4 sm:px-6 py-3">
        <div className="flex items-center justify-between max-w-7xl mx-auto">
          <div className="flex items-center gap-3 min-w-0 flex-1">
            <div className="h-8 w-8 bg-orange-600 rounded-full flex items-center justify-center flex-shrink-0">
              <Settings className="h-4 w-4 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className="font-semibold truncate text-base sm:text-lg">Tech Team Portal</h1>
              <p className="text-sm text-muted-foreground truncate">Welcome, {user.name} (ID: {user.id})</p>
            </div>
          </div>
          <UserDropdown user={user} onLogout={onLogout} onNavigate={onNavigate} />
        </div>
      </header>

      <div className="p-3 sm:p-4 max-w-7xl mx-auto">
        {/* Dashboard Quick Actions */}
        <div className="mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">System Dashboard</h2>
              <p className="text-sm text-muted-foreground">Monitor system performance and manage data</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <Server className="h-4 w-4 mr-2" />
                System Logs
              </Button>
              <Button variant="outline" size="sm">
                <Settings className="h-4 w-4 mr-2" />
                Settings
              </Button>
            </div>
          </div>
        </div>

        <Tabs defaultValue="overview" className="space-y-3 sm:space-y-4">
          <TabsList className="grid w-full grid-cols-3 h-auto">
            <TabsTrigger value="overview" className="text-xs sm:text-sm py-2 px-2 sm:px-4">
              <BarChart3 className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">Overview</span>
              <span className="sm:hidden">Stats</span>
            </TabsTrigger>
            <TabsTrigger value="ai-model" className="text-xs sm:text-sm py-2 px-2 sm:px-4">
              <Brain className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">AI Model</span>
              <span className="sm:hidden">AI</span>
            </TabsTrigger>
            <TabsTrigger value="data" className="text-xs sm:text-sm py-2 px-2 sm:px-4">
              <Database className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">Data Repository</span>
              <span className="sm:hidden">Data</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="overview" className="space-y-4 sm:space-y-6">
            {/* System Health Overview */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
              <Card className="border-l-4 border-l-blue-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">System Health</p>
                      <p className={`text-lg sm:text-2xl font-semibold ${systemStatus.color}`}>
                        {systemStats.systemHealth}%
                      </p>
                    </div>
                    <systemStatus.icon className={`h-6 w-6 sm:h-8 sm:w-8 ${systemStatus.color}`} />
                  </div>
                  <p className={`text-xs mt-1 ${systemStatus.color}`}>{systemStatus.status}</p>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-green-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Total Reports</p>
                      <p className="text-lg sm:text-2xl font-semibold">{systemStats.total}</p>
                    </div>
                    <FileText className="h-6 w-6 sm:h-8 sm:w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-purple-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Avg Processing</p>
                      <p className="text-lg sm:text-2xl font-semibold">{systemStats.processingTime}s</p>
                    </div>
                    <Activity className="h-6 w-6 sm:h-8 sm:w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-orange-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Avg Confidence</p>
                      <p className={`text-lg sm:text-2xl font-semibold ${getConfidenceColor(systemStats.avgConfidence)}`}>
                        {systemStats.avgConfidence}%
                      </p>
                    </div>
                    <Brain className="h-6 w-6 sm:h-8 sm:w-8 text-orange-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Usage Analytics */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
              <Card>
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg sm:text-xl">Usage Analytics</CardTitle>
                  <CardDescription>System usage over time</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <span className="text-sm font-medium">Today</span>
                    <div className="flex items-center gap-2">
                      <div className="text-lg sm:text-xl font-semibold">{systemStats.todayReports}</div>
                      <span className="text-xs sm:text-sm text-muted-foreground">reports</span>
                    </div>
                  </div>
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <span className="text-sm font-medium">This Week</span>
                    <div className="flex items-center gap-2">
                      <div className="text-lg sm:text-xl font-semibold">{systemStats.weekReports}</div>
                      <span className="text-xs sm:text-sm text-muted-foreground">reports</span>
                    </div>
                  </div>
                  <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2">
                    <span className="text-sm font-medium">All Time</span>
                    <div className="flex items-center gap-2">
                      <div className="text-lg sm:text-xl font-semibold">{systemStats.total}</div>
                      <span className="text-xs sm:text-sm text-muted-foreground">reports</span>
                    </div>
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader className="pb-4">
                  <CardTitle className="text-lg sm:text-xl">Risk Distribution</CardTitle>
                  <CardDescription>Breakdown of AI risk assessments</CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-red-500 rounded-full"></div>
                        <span className="text-sm">High Risk</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{systemStats.riskDistribution.high}</span>
                        <span className="text-xs text-muted-foreground">
                          ({((systemStats.riskDistribution.high / systemStats.total) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <Progress 
                      value={(systemStats.riskDistribution.high / systemStats.total) * 100} 
                      className="h-2"
                    />
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-yellow-500 rounded-full"></div>
                        <span className="text-sm">Medium Risk</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{systemStats.riskDistribution.medium}</span>
                        <span className="text-xs text-muted-foreground">
                          ({((systemStats.riskDistribution.medium / systemStats.total) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <Progress 
                      value={(systemStats.riskDistribution.medium / systemStats.total) * 100} 
                      className="h-2"
                    />
                  </div>
                  
                  <div className="space-y-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center gap-2">
                        <div className="w-3 h-3 bg-green-500 rounded-full"></div>
                        <span className="text-sm">Low Risk</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-medium">{systemStats.riskDistribution.low}</span>
                        <span className="text-xs text-muted-foreground">
                          ({((systemStats.riskDistribution.low / systemStats.total) * 100).toFixed(1)}%)
                        </span>
                      </div>
                    </div>
                    <Progress 
                      value={(systemStats.riskDistribution.low / systemStats.total) * 100} 
                      className="h-2"
                    />
                  </div>
                </CardContent>
              </Card>
            </div>
          </TabsContent>

          <TabsContent value="ai-model" className="space-y-4 sm:space-y-6">
            {/* Model Performance */}
            <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3 sm:gap-4">
              <Card>
                <CardContent className="p-3 sm:p-4 text-center">
                  <div className="text-xl sm:text-2xl font-semibold text-blue-600">{modelMetrics.accuracy}%</div>
                  <div className="text-xs sm:text-sm text-muted-foreground">Accuracy</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 sm:p-4 text-center">
                  <div className="text-xl sm:text-2xl font-semibold text-green-600">{modelMetrics.precision}%</div>
                  <div className="text-xs sm:text-sm text-muted-foreground">Precision</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 sm:p-4 text-center">
                  <div className="text-xl sm:text-2xl font-semibold text-purple-600">{modelMetrics.recall}%</div>
                  <div className="text-xs sm:text-sm text-muted-foreground">Recall</div>
                </CardContent>
              </Card>
              <Card>
                <CardContent className="p-3 sm:p-4 text-center">
                  <div className="text-xl sm:text-2xl font-semibold text-orange-600">{modelMetrics.f1Score}%</div>
                  <div className="text-xs sm:text-sm text-muted-foreground">F1 Score</div>
                </CardContent>
              </Card>
            </div>

            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg sm:text-xl">Model Performance Details</CardTitle>
                <CardDescription>
                  Detailed metrics for the AI risk assessment model
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4 sm:space-y-6">
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 sm:gap-6">
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Total Predictions</span>
                      <span className="text-lg font-semibold">{modelMetrics.totalPredictions.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Correct Predictions</span>
                      <span className="text-lg font-semibold text-green-600">{modelMetrics.correctPredictions.toLocaleString()}</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Model Version</span>
                      <Badge variant="outline">v2.1.3</Badge>
                    </div>
                  </div>
                  <div className="space-y-4">
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Last Updated</span>
                      <span className="text-sm text-muted-foreground">
                        {new Date(modelMetrics.lastUpdated).toLocaleDateString()}
                      </span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Training Dataset</span>
                      <span className="text-sm text-muted-foreground">45,000 images</span>
                    </div>
                    <div className="flex justify-between items-center">
                      <span className="text-sm font-medium">Model Status</span>
                      <Badge className="bg-green-100 text-green-800">Active</Badge>
                    </div>
                  </div>
                </div>
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="data" className="space-y-4 sm:space-y-6">
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg sm:text-xl">Data Repository</CardTitle>
                <CardDescription>
                  Browse and manage uploaded medical reports and images
                </CardDescription>
              </CardHeader>
              <CardContent>
                {reports.length === 0 ? (
                  <div className="text-center py-8 sm:py-12 text-muted-foreground">
                    <Database className="h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm sm:text-base">No data available</p>
                    <p className="text-xs sm:text-sm mt-1">Reports will appear here once uploaded</p>
                  </div>
                ) : (
                  <div className="space-y-3 sm:space-y-4">
                    {reports.slice(0, 10).map((report) => (
                      <div key={report.id} className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-3 sm:p-4 border rounded-lg">
                        <div className="flex items-center gap-3 sm:gap-4 flex-1 w-full">
                          <div className="h-12 w-12 sm:h-16 sm:w-16 bg-gray-100 rounded flex items-center justify-center overflow-hidden flex-shrink-0">
                            <img
                              src={report.reportImage}
                              alt="Report thumbnail"
                              className="h-full w-full object-cover"
                            />
                          </div>
                          <div className="flex-1 min-w-0">
                            <div className="flex flex-col sm:flex-row sm:items-center gap-2 sm:gap-3 mb-2">
                              <h3 className="font-medium truncate text-sm sm:text-base">
                                ID: {report.id}
                              </h3>
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge variant="outline" size="sm">
                                  Patient: {report.patientId}
                                </Badge>
                                <Badge 
                                  className={
                                    report.riskScore === 'high' ? 'bg-red-100 text-red-800' :
                                    report.riskScore === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                                    'bg-green-100 text-green-800'
                                  }
                                  size="sm"
                                >
                                  {report.riskScore.toUpperCase()}
                                </Badge>
                              </div>
                            </div>
                            <div className="text-xs sm:text-sm text-muted-foreground space-y-1">
                              <p className="truncate">
                                Study: {report.studyType || 'X-Ray'} • 
                                Confidence: {report.confidence ? Math.round(report.confidence * 100) : 'N/A'}%
                              </p>
                              <p className="truncate">
                                Uploaded: {new Date(report.uploadDate).toLocaleDateString()}{' '}
                                <span className="hidden sm:inline">
                                  by Dr. {report.uploadedBy}
                                </span>
                              </p>
                            </div>
                          </div>
                        </div>
                        <Button
                          variant="outline"
                          size="sm"
                          onClick={() => setSelectedReport(report)}
                          className="w-full sm:w-auto flex-shrink-0"
                        >
                          <Database className="h-4 w-4 sm:mr-2" />
                          <span className="text-xs sm:text-sm">View</span>
                        </Button>
                      </div>
                    ))}
                    
                    {reports.length > 10 && (
                      <div className="text-center pt-4">
                        <p className="text-sm text-muted-foreground">
                          Showing 10 of {reports.length} reports
                        </p>
                      </div>
                    )}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Data Detail Dialog */}
      <Dialog open={!!selectedReport} onOpenChange={() => setSelectedReport(null)}>
        <DialogContent className="max-w-lg sm:max-w-3xl lg:max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <DialogTitle className="text-base sm:text-lg">
                Data Record: {selectedReport?.id}
              </DialogTitle>
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setSelectedReport(null)}
                className="h-6 w-6 p-0"
              >
                <X className="h-4 w-4" />
              </Button>
            </div>
          </DialogHeader>
          {selectedReport && (
            <div className="space-y-4">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 pb-4 border-b">
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">Patient ID</div>
                  <div className="font-medium">{selectedReport.patientId}</div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">Study Type</div>
                  <div className="font-medium">{selectedReport.studyType || 'X-Ray'}</div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">Risk Score</div>
                  <Badge 
                    className={
                      selectedReport.riskScore === 'high' ? 'bg-red-100 text-red-800' :
                      selectedReport.riskScore === 'medium' ? 'bg-yellow-100 text-yellow-800' :
                      'bg-green-100 text-green-800'
                    }
                  >
                    {selectedReport.riskScore.toUpperCase()}
                  </Badge>
                </div>
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">AI Confidence</div>
                  <div className="font-medium">
                    {selectedReport.confidence ? Math.round(selectedReport.confidence * 100) : 'N/A'}%
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">Uploaded By</div>
                  <div className="font-medium">Dr. {selectedReport.uploadedBy}</div>
                </div>
                <div className="space-y-2">
                  <div className="text-xs sm:text-sm text-muted-foreground">Upload Date</div>
                  <div className="font-medium">
                    {new Date(selectedReport.uploadDate).toLocaleString()}
                  </div>
                </div>
              </div>
              
              <div className="space-y-4">
                {selectedReport.findings && selectedReport.findings.length > 0 && (
                  <div className="bg-blue-50 p-3 sm:p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-sm sm:text-base">AI Findings</h4>
                    <ul className="list-disc list-inside space-y-1">
                      {selectedReport.findings.map((finding, index) => (
                        <li key={index} className="text-xs sm:text-sm text-muted-foreground">
                          {finding}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
                
                <div className="space-y-2">
                  <h4 className="font-medium text-sm sm:text-base">Report Image</h4>
                  <div className="relative bg-gray-100 rounded-lg overflow-hidden">
                    <img
                      src={selectedReport.reportImage}
                      alt="Medical report"
                      className="w-full h-auto max-h-[50vh] object-contain"
                    />
                  </div>
                </div>
              </div>
            </div>
          )}
        </DialogContent>
      </Dialog>
    </div>
  );
}