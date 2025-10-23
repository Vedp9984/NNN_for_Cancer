import { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Badge } from './ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from './ui/tabs';
import { Dialog, DialogContent, DialogHeader, DialogTitle } from './ui/dialog';
import { AlertTriangle, FileText, Shield, Users, Calendar, Brain, TrendingUp, Activity, X, MessageSquare } from 'lucide-react';
import { User as UserType } from '../App';
import { getAllReports, MedicalReport } from '../utils/reportStorage';
import { UserDropdown } from './UserDropdown';
import { useTheme } from '../utils/themeContext';

interface DoctorDashboardProps {
  user: UserType;
  onLogout: () => void;
  onNavigate: (destination: 'dashboard' | 'profile' | 'settings') => void;
}

export function DoctorDashboard({ user, onLogout, onNavigate }: DoctorDashboardProps) {
  const { actualTheme } = useTheme();
  const isDarkMode = actualTheme === 'dark';
  const [reports, setReports] = useState<MedicalReport[]>([]);
  const [selectedReport, setSelectedReport] = useState<MedicalReport | null>(null);
  const [selectedFilter, setSelectedFilter] = useState<'all' | 'high' | 'medium' | 'low'>('all');

  useEffect(() => {
    const allReports = getAllReports();
    // Sort by risk priority and date
    const sortedReports = allReports.sort((a, b) => {
      const riskOrder = { high: 3, medium: 2, low: 1 };
      const riskDiff = (riskOrder[b.riskScore as keyof typeof riskOrder] || 0) - 
                      (riskOrder[a.riskScore as keyof typeof riskOrder] || 0);
      if (riskDiff !== 0) return riskDiff;
      return new Date(b.uploadDate).getTime() - new Date(a.uploadDate).getTime();
    });
    setReports(sortedReports);
  }, []);

  const filteredReports = selectedFilter === 'all' 
    ? reports 
    : reports.filter(report => report.riskScore === selectedFilter);

  const getRiskColor = (risk: string) => {
    switch (risk) {
      case 'low': return 'bg-green-100 text-green-800 border-green-200';
      case 'medium': return 'bg-yellow-100 text-yellow-800 border-yellow-200';
      case 'high': return 'bg-red-100 text-red-800 border-red-200';
      default: return 'bg-gray-100 text-gray-800 border-gray-200';
    }
  };

  const getRiskIcon = (risk: string) => {
    if (risk === 'medium' || risk === 'high') {
      return <AlertTriangle className="h-4 w-4" />;
    }
    return null;
  };

  const getUrgencyLevel = (risk: string) => {
    switch (risk) {
      case 'high': return 'URGENT';
      case 'medium': return 'FOLLOW-UP';
      case 'low': return 'ROUTINE';
      default: return 'REVIEW';
    }
  };

  const getNextSteps = (risk: string) => {
    switch (risk) {
      case 'high':
        return 'Immediate consultation recommended. Contact patient within 24 hours to schedule urgent follow-up.';
      case 'medium':
        return 'Schedule follow-up appointment within 2-4 weeks. Monitor patient symptoms and discuss findings.';
      case 'low':
        return 'Continue routine care. No immediate action required unless patient symptoms change.';
      default:
        return 'Review findings and determine appropriate follow-up based on clinical judgment.';
    }
  };

  const stats = {
    total: reports.length,
    high: reports.filter(r => r.riskScore === 'high').length,
    medium: reports.filter(r => r.riskScore === 'medium').length,
    low: reports.filter(r => r.riskScore === 'low').length,
    todayReports: reports.filter(r => new Date(r.uploadDate).toDateString() === new Date().toDateString()).length
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'} safe-area-inset`}>
      {/* Header */}
      <header className={`${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200'} border-b mobile-header`}>
        <div className="flex-responsive max-w-7xl mx-auto">
          <div className="flex items-center space-responsive min-w-0 flex-1">
            <div className="avatar-responsive h-8 w-8 bg-purple-600 rounded-full flex items-center justify-center flex-shrink-0 touch-target">
              <Shield className="h-4 w-4 text-white" />
            </div>
            <div className="min-w-0">
              <h1 className={`font-semibold truncate text-fluid-lg ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>Doctor Portal</h1>
              <p className={`text-fluid-sm truncate ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>Welcome, {user.name} (ID: {user.id})</p>
            </div>
          </div>
          <UserDropdown user={user} onLogout={onLogout} onNavigate={onNavigate} isDarkMode={isDarkMode} />
        </div>
      </header>

      <div className="p-3 sm:p-4 max-w-7xl mx-auto">
        {/* Quick Actions Bar */}
        <div className="mb-6">
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4 mb-4">
            <div>
              <h2 className="text-xl font-semibold text-gray-900">Clinical Dashboard</h2>
              <p className="text-sm text-muted-foreground">Patient reports and clinical insights</p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" size="sm">
                <MessageSquare className="h-4 w-4 mr-2" />
                Messages
              </Button>
              <Button variant="outline" size="sm">
                <Calendar className="h-4 w-4 mr-2" />
                Schedule
              </Button>
            </div>
          </div>
        </div>

        <Tabs defaultValue="dashboard" className="space-y-3 sm:space-y-4">
          <TabsList className="grid w-full grid-cols-2 h-auto">
            <TabsTrigger value="dashboard" className="text-xs sm:text-sm py-2 px-2 sm:px-4">
              <Activity className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">Dashboard</span>
              <span className="sm:hidden">Home</span>
            </TabsTrigger>
            <TabsTrigger value="reports" className="text-xs sm:text-sm py-2 px-2 sm:px-4">
              <FileText className="h-4 w-4 sm:mr-2" />
              <span className="hidden sm:inline">All Reports</span>
              <span className="sm:hidden">Reports</span>
            </TabsTrigger>
          </TabsList>

          <TabsContent value="dashboard" className="space-y-4 sm:space-y-6">
            {/* Stats Overview */}
            <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3 sm:gap-4">
              <Card className="border-l-4 border-l-blue-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Total Reports</p>
                      <p className="text-lg sm:text-2xl font-semibold">{stats.total}</p>
                    </div>
                    <FileText className="h-6 w-6 sm:h-8 sm:w-8 text-blue-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-red-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">High Risk</p>
                      <p className="text-lg sm:text-2xl font-semibold text-red-600">{stats.high}</p>
                    </div>
                    <AlertTriangle className="h-6 w-6 sm:h-8 sm:w-8 text-red-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-yellow-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Medium Risk</p>
                      <p className="text-lg sm:text-2xl font-semibold text-yellow-600">{stats.medium}</p>
                    </div>
                    <TrendingUp className="h-6 w-6 sm:h-8 sm:w-8 text-yellow-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-green-500">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Low Risk</p>
                      <p className="text-lg sm:text-2xl font-semibold text-green-600">{stats.low}</p>
                    </div>
                    <Users className="h-6 w-6 sm:h-8 sm:w-8 text-green-500" />
                  </div>
                </CardContent>
              </Card>

              <Card className="border-l-4 border-l-purple-500 col-span-2 sm:col-span-1">
                <CardContent className="p-3 sm:p-4">
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="text-xs sm:text-sm text-muted-foreground">Today</p>
                      <p className="text-lg sm:text-2xl font-semibold text-purple-600">{stats.todayReports}</p>
                    </div>
                    <Calendar className="h-6 w-6 sm:h-8 sm:w-8 text-purple-500" />
                  </div>
                </CardContent>
              </Card>
            </div>

            {/* Priority Queue */}
            <Card>
              <CardHeader className="pb-4">
                <CardTitle className="text-lg sm:text-xl">Priority Queue</CardTitle>
                <CardDescription>
                  High and medium risk cases requiring immediate attention
                </CardDescription>
              </CardHeader>
              <CardContent>
                {reports.filter(r => r.riskScore === 'high' || r.riskScore === 'medium').length === 0 ? (
                  <div className="text-center py-6 sm:py-8 text-muted-foreground">
                    <AlertTriangle className="h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm sm:text-base">No priority cases</p>
                    <p className="text-xs sm:text-sm mt-1">All current reports are low risk</p>
                  </div>
                ) : (
                  <div className="space-y-3 sm:space-y-4">
                    {reports.filter(r => r.riskScore === 'high' || r.riskScore === 'medium').slice(0, 5).map((report) => (
                      <div key={report.id} className="flex flex-col sm:flex-row items-start sm:items-center gap-4 p-3 sm:p-4 border rounded-lg bg-white">
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
                                Patient: {report.patientId}
                              </h3>
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge className={getRiskColor(report.riskScore)} size="sm">
                                  {getRiskIcon(report.riskScore)}
                                  <span className="ml-1">{getUrgencyLevel(report.riskScore)}</span>
                                </Badge>
                                <Badge variant="outline" className="gap-1 text-xs">
                                  <Brain className="h-3 w-3" />
                                  AI
                                </Badge>
                              </div>
                            </div>
                            <div className="text-xs sm:text-sm text-muted-foreground space-y-1">
                              <p className="truncate">
                                Study: {report.studyType || 'X-Ray'} • 
                                Dr. {report.uploadedBy}
                              </p>
                              <p className="truncate">
                                {new Date(report.uploadDate).toLocaleDateString()}{' '}
                                <span className="hidden sm:inline">
                                  at {new Date(report.uploadDate).toLocaleTimeString()}
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
                          <FileText className="h-4 w-4 sm:mr-2" />
                          <span className="text-xs sm:text-sm">Review</span>
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>

          <TabsContent value="reports" className="space-y-4 sm:space-y-6">
            {/* Filter Tabs */}
            <Card>
              <CardHeader className="pb-4">
                <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                  <div>
                    <CardTitle className="text-lg sm:text-xl">All Reports</CardTitle>
                    <CardDescription>Complete list of patient reports with AI assessments</CardDescription>
                  </div>
                  <div className="flex gap-2 flex-wrap">
                    {(['all', 'high', 'medium', 'low'] as const).map((filter) => (
                      <Button
                        key={filter}
                        variant={selectedFilter === filter ? 'default' : 'outline'}
                        size="sm"
                        onClick={() => setSelectedFilter(filter)}
                        className="text-xs sm:text-sm"
                      >
                        {filter === 'all' ? 'All' : `${filter.charAt(0).toUpperCase() + filter.slice(1)} Risk`}
                        {filter !== 'all' && (
                          <Badge variant="secondary" className="ml-2" size="sm">
                            {filter === 'high' ? stats.high : filter === 'medium' ? stats.medium : stats.low}
                          </Badge>
                        )}
                      </Button>
                    ))}
                  </div>
                </div>
              </CardHeader>
              <CardContent>
                {filteredReports.length === 0 ? (
                  <div className="text-center py-8 sm:py-12 text-muted-foreground">
                    <FileText className="h-8 w-8 sm:h-12 sm:w-12 mx-auto mb-3 opacity-50" />
                    <p className="text-sm sm:text-base">
                      {selectedFilter === 'all' ? 'No reports available' : `No ${selectedFilter} risk reports`}
                    </p>
                  </div>
                ) : (
                  <div className="space-y-3 sm:space-y-4">
                    {filteredReports.map((report) => (
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
                                Patient: {report.patientId}
                              </h3>
                              <div className="flex items-center gap-2 flex-wrap">
                                <Badge className={getRiskColor(report.riskScore)} size="sm">
                                  {getRiskIcon(report.riskScore)}
                                  <span className="ml-1">{report.riskScore.toUpperCase()}</span>
                                </Badge>
                                <Badge variant="outline" className="gap-1 text-xs">
                                  <Brain className="h-3 w-3" />
                                  {report.confidence ? `${Math.round(report.confidence * 100)}%` : 'AI'}
                                </Badge>
                              </div>
                            </div>
                            <div className="text-xs sm:text-sm text-muted-foreground space-y-1">
                              <p className="truncate">
                                Study: {report.studyType || 'X-Ray'} • Dr. {report.uploadedBy}
                              </p>
                              <p className="truncate">
                                {new Date(report.uploadDate).toLocaleDateString()}{' '}
                                <span className="hidden sm:inline">
                                  at {new Date(report.uploadDate).toLocaleTimeString()}
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
                          <FileText className="h-4 w-4 sm:mr-2" />
                          <span className="text-xs sm:text-sm">Review</span>
                        </Button>
                      </div>
                    ))}
                  </div>
                )}
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </div>

      {/* Report Detail Dialog */}
      <Dialog open={!!selectedReport} onOpenChange={() => setSelectedReport(null)}>
        <DialogContent className="max-w-lg sm:max-w-3xl lg:max-w-4xl max-h-[90vh] overflow-y-auto">
          <DialogHeader>
            <div className="flex items-center justify-between">
              <DialogTitle className="text-base sm:text-lg">
                Patient Report: {selectedReport?.patientId}
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
              <div className="flex flex-col sm:flex-row sm:items-center gap-3 sm:gap-4 pb-4 border-b">
                <Badge className={getRiskColor(selectedReport.riskScore)}>
                  {getRiskIcon(selectedReport.riskScore)}
                  <span className="ml-1">{getUrgencyLevel(selectedReport.riskScore)}</span>
                </Badge>
                <Badge variant="outline" className="gap-1">
                  <Brain className="h-3 w-3" />
                  AI Confidence: {selectedReport.confidence ? Math.round(selectedReport.confidence * 100) : 'N/A'}%
                </Badge>
                <div className="text-xs sm:text-sm text-muted-foreground">
                  {new Date(selectedReport.uploadDate).toLocaleDateString()} • Dr. {selectedReport.uploadedBy}
                </div>
              </div>
              
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-4 sm:gap-6">
                <div className="space-y-4">
                  <div className="bg-gray-50 p-3 sm:p-4 rounded-lg">
                    <h4 className="font-medium mb-2 text-sm sm:text-base">Clinical Recommendations</h4>
                    <p className="text-xs sm:text-sm text-muted-foreground">
                      {getNextSteps(selectedReport.riskScore)}
                    </p>
                  </div>
                  
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

                  {/* Communication Tools */}
                  <div className="bg-purple-50 p-3 sm:p-4 rounded-lg">
                    <h4 className="font-medium mb-3 text-sm sm:text-base">Communication</h4>
                    <div className="space-y-2">
                      <Button 
                        onClick={() => {
                          // In a real app, this would open a messaging interface
                          alert(`Opening message interface for Patient ${selectedReport.patientId}`);
                        }}
                        className="w-full sm:w-auto bg-purple-600 hover:bg-purple-700"
                        size="sm"
                      >
                        <MessageSquare className="h-4 w-4 mr-2" />
                        New Message to Patient
                      </Button>
                      <p className="text-xs text-muted-foreground">
                        Send secure messages regarding this case
                      </p>
                    </div>
                  </div>
                </div>
                
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