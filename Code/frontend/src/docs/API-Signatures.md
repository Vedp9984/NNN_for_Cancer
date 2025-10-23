# API Signatures - Medical Portal System

This document provides the complete API signatures for all components, utilities, and interfaces in the Medical Portal System.

## Mermaid Class Diagram - API Signatures

```mermaid
classDiagram
    %% ===== CORE APP TYPES =====
    class User {
        <<interface>>
        +string id
        +string name
        +string email
        +UserRole role
    }
    
    class UserRole {
        <<type>>
        'patient' | 'radiologist' | 'doctor' | 'tech' | null
    }
    
    class AppView {
        <<type>>
        'login' | 'signup' | 'dashboard' | 'profile' | 'settings'
    }
    
    %% ===== AUTHENTICATION COMPONENTS =====
    class LoginProps {
        <<interface>>
        +onLogin(user: User) void
        +onSignUpClick() void
    }
    
    class SignUpProps {
        <<interface>>
        +onSignUp(user: User) void
        +onBackToLogin() void
    }
    
    class Login {
        <<component>>
        +render(props: LoginProps) JSX.Element
    }
    
    class SignUp {
        <<component>>
        +render(props: SignUpProps) JSX.Element
    }
    
    %% ===== DASHBOARD COMPONENTS =====
    class DashboardPropsBase {
        <<interface>>
        +user: User
        +onLogout() void
        +onNavigate(destination: 'dashboard' | 'profile' | 'settings') void
    }
    
    class PatientDashboard {
        <<component>>
        +render(props: DashboardPropsBase) JSX.Element
    }
    
    class RadiologistDashboard {
        <<component>>
        +render(props: DashboardPropsBase) JSX.Element
    }
    
    class DoctorDashboard {
        <<component>>
        +render(props: DashboardPropsBase) JSX.Element
    }
    
    class TechDashboard {
        <<component>>
        +render(props: DashboardPropsBase) JSX.Element
    }
    
    %% ===== PROFILE & SETTINGS =====
    class ProfileProps {
        <<interface>>
        +user: User
        +onBack() void
        +onUpdateProfile(updatedUser: User) void
    }
    
    class SettingsProps {
        <<interface>>
        +user: User
        +onBack() void
    }
    
    class Profile {
        <<component>>
        +render(props: ProfileProps) JSX.Element
    }
    
    class Settings {
        <<component>>
        +render(props: SettingsProps) JSX.Element
    }
    
    %% ===== SHARED UI COMPONENTS =====
    class UserDropdownProps {
        <<interface>>
        +user: User
        +onNavigate(destination: 'dashboard' | 'profile' | 'settings') void
        +onLogout() void
    }
    
    class UserDropdown {
        <<component>>
        +render(props: UserDropdownProps) JSX.Element
    }
    
    class ThemeToggleProps {
        <<interface>>
        +className?: string
    }
    
    class ThemeToggle {
        <<component>>
        +render(props: ThemeToggleProps) JSX.Element
    }
    
    class ThemeCardProps {
        <<interface>>
        +children: ReactNode
        +className?: string
        +variant?: 'default' | 'bordered' | 'elevated'
    }
    
    class ThemeCard {
        <<component>>
        +render(props: ThemeCardProps) JSX.Element
    }
    
    class ErrorBoundaryProps {
        <<interface>>
        +children: ReactNode
    }
    
    class ErrorBoundaryState {
        <<interface>>
        +hasError: boolean
        +error?: Error
    }
    
    class ErrorBoundary {
        <<component>>
        +render(props: ErrorBoundaryProps) JSX.Element
        +componentDidCatch(error: Error, info: ErrorInfo) void
    }
    
    %% ===== MEDICAL REPORT TYPES =====
    class MedicalReport {
        <<interface>>
        +reportId: string
        +patientId: string
        +patientName: string
        +radiologistId: string
        +radiologistName: string
        +reportImageUrl: string
        +studyType: string
        +riskScore: RiskScore
        +confidenceScore: number
        +uploadDate: Date
        +analysisDate: Date
        +findings: string
        +recommendations: string[]
        +status: 'pending' | 'analyzed' | 'reviewed'
    }
    
    class RiskScore {
        <<type>>
        'low' | 'medium' | 'high'
    }
    
    %% ===== AI RISK ASSESSMENT =====
    class RiskAssessmentResult {
        <<interface>>
        +riskScore: RiskScore
        +confidenceScore: number
        +findings: string
        +recommendations: string[]
        +processingTime: number
    }
    
    class AnalyzeImageRisk {
        <<function>>
        +analyzeImageRisk(imageUrl: string, patientId: string) Promise~RiskAssessmentResult~
    }
    
    %% ===== REPORT STORAGE =====
    class ReportStorage {
        <<utility>>
        +saveReport(report: MedicalReport) void
        +getReportsByPatientId(patientId: string) MedicalReport[]
        +getAllReports() MedicalReport[]
        +getReportById(reportId: string) MedicalReport | undefined
    }
    
    %% ===== THEME CONTEXT =====
    class ThemeContextType {
        <<interface>>
        +theme: Theme
        +setTheme(theme: Theme) void
        +actualTheme: 'light' | 'dark'
    }
    
    class Theme {
        <<type>>
        'light' | 'dark' | 'system'
    }
    
    class ThemeProviderProps {
        <<interface>>
        +children: ReactNode
        +defaultTheme?: Theme
    }
    
    class ThemeProvider {
        <<component>>
        +render(props: ThemeProviderProps) JSX.Element
    }
    
    class UseThemeHook {
        <<hook>>
        +useTheme() ThemeContextType
    }
    
    %% ===== RESPONSIVE UTILITIES =====
    class Breakpoint {
        <<type>>
        'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl'
    }
    
    class DeviceType {
        <<type>>
        'mobile' | 'tablet' | 'desktop'
    }
    
    class ResponsiveHookResult {
        <<interface>>
        +isMobile: boolean
        +isTablet: boolean
        +isDesktop: boolean
        +breakpoint: Breakpoint
        +deviceType: DeviceType
        +isTouch: boolean
        +orientation: 'portrait' | 'landscape'
        +width: number
        +height: number
    }
    
    class UseResponsiveHook {
        <<hook>>
        +useResponsive() ResponsiveHookResult
        +useMediaQuery(query: string) boolean
        +useResponsiveValue~T~(values: ResponsiveValues~T~) T
    }
    
    class ResponsiveValues {
        <<interface>>
        +mobile?: T
        +tablet?: T
        +desktop?: T
    }
    
    %% ===== SHADCN UI COMPONENTS =====
    class ButtonProps {
        <<interface>>
        +variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link'
        +size?: 'default' | 'sm' | 'lg' | 'icon'
        +asChild?: boolean
        +children: ReactNode
        +className?: string
        +onClick?: () => void
        +disabled?: boolean
        +type?: 'button' | 'submit' | 'reset'
    }
    
    class Button {
        <<component>>
        +render(props: ButtonProps) JSX.Element
    }
    
    class InputProps {
        <<interface>>
        +type?: string
        +placeholder?: string
        +value?: string
        +onChange?: (e: ChangeEvent) => void
        +onFocus?: () => void
        +onBlur?: () => void
        +className?: string
        +disabled?: boolean
        +id?: string
    }
    
    class Input {
        <<component>>
        +render(props: InputProps) JSX.Element
    }
    
    class CardProps {
        <<interface>>
        +children: ReactNode
        +className?: string
    }
    
    class Card {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class CardHeader {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class CardTitle {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class CardDescription {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class CardContent {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class CardFooter {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class LabelProps {
        <<interface>>
        +htmlFor?: string
        +children: ReactNode
        +className?: string
    }
    
    class Label {
        <<component>>
        +render(props: LabelProps) JSX.Element
    }
    
    class BadgeProps {
        <<interface>>
        +variant?: 'default' | 'secondary' | 'destructive' | 'outline'
        +children: ReactNode
        +className?: string
    }
    
    class Badge {
        <<component>>
        +render(props: BadgeProps) JSX.Element
    }
    
    class AlertProps {
        <<interface>>
        +variant?: 'default' | 'destructive'
        +children: ReactNode
        +className?: string
    }
    
    class Alert {
        <<component>>
        +render(props: AlertProps) JSX.Element
    }
    
    class AlertDescription {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class TabsProps {
        <<interface>>
        +defaultValue: string
        +value?: string
        +onValueChange?: (value: string) => void
        +children: ReactNode
        +className?: string
    }
    
    class Tabs {
        <<component>>
        +render(props: TabsProps) JSX.Element
    }
    
    class TabsList {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class TabsTrigger {
        <<component>>
        +render(props: { value: string, children: ReactNode }) JSX.Element
    }
    
    class TabsContent {
        <<component>>
        +render(props: { value: string, children: ReactNode }) JSX.Element
    }
    
    class DialogProps {
        <<interface>>
        +open?: boolean
        +onOpenChange?: (open: boolean) => void
        +children: ReactNode
    }
    
    class Dialog {
        <<component>>
        +render(props: DialogProps) JSX.Element
    }
    
    class DialogTrigger {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DialogContent {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DialogHeader {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DialogTitle {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DialogDescription {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DialogFooter {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class SelectProps {
        <<interface>>
        +value?: string
        +onValueChange?: (value: string) => void
        +children: ReactNode
        +defaultValue?: string
        +disabled?: boolean
    }
    
    class Select {
        <<component>>
        +render(props: SelectProps) JSX.Element
    }
    
    class SelectTrigger {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class SelectValue {
        <<component>>
        +render(props: { placeholder?: string }) JSX.Element
    }
    
    class SelectContent {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class SelectItem {
        <<component>>
        +render(props: { value: string, children: ReactNode }) JSX.Element
    }
    
    class TableProps {
        <<interface>>
        +children: ReactNode
        +className?: string
    }
    
    class Table {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class TableHeader {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class TableBody {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class TableRow {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class TableHead {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class TableCell {
        <<component>>
        +render(props: TableProps) JSX.Element
    }
    
    class ProgressProps {
        <<interface>>
        +value?: number
        +className?: string
    }
    
    class Progress {
        <<component>>
        +render(props: ProgressProps) JSX.Element
    }
    
    class SeparatorProps {
        <<interface>>
        +orientation?: 'horizontal' | 'vertical'
        +decorative?: boolean
        +className?: string
    }
    
    class Separator {
        <<component>>
        +render(props: SeparatorProps) JSX.Element
    }
    
    class AvatarProps {
        <<interface>>
        +children: ReactNode
        +className?: string
    }
    
    class Avatar {
        <<component>>
        +render(props: AvatarProps) JSX.Element
    }
    
    class AvatarImage {
        <<component>>
        +render(props: { src: string, alt: string }) JSX.Element
    }
    
    class AvatarFallback {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DropdownMenuProps {
        <<interface>>
        +open?: boolean
        +onOpenChange?: (open: boolean) => void
        +children: ReactNode
    }
    
    class DropdownMenu {
        <<component>>
        +render(props: DropdownMenuProps) JSX.Element
    }
    
    class DropdownMenuTrigger {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DropdownMenuContent {
        <<component>>
        +render(props: { align?: 'start' | 'center' | 'end', children: ReactNode }) JSX.Element
    }
    
    class DropdownMenuItem {
        <<component>>
        +render(props: { onSelect?: () => void, children: ReactNode }) JSX.Element
    }
    
    class DropdownMenuLabel {
        <<component>>
        +render(props: CardProps) JSX.Element
    }
    
    class DropdownMenuSeparator {
        <<component>>
        +render(props: {}) JSX.Element
    }
    
    %% ===== TOAST NOTIFICATIONS =====
    class ToastFunction {
        <<function>>
        +toast(message: string, options?: ToastOptions) void
        +toast.success(message: string, options?: ToastOptions) void
        +toast.error(message: string, options?: ToastOptions) void
        +toast.info(message: string, options?: ToastOptions) void
        +toast.warning(message: string, options?: ToastOptions) void
    }
    
    class ToastOptions {
        <<interface>>
        +description?: string
        +duration?: number
        +action?: ToastAction
    }
    
    class ToastAction {
        <<interface>>
        +label: string
        +onClick: () => void
    }
    
    class Toaster {
        <<component>>
        +render() JSX.Element
    }
    
    %% ===== APP COMPONENT =====
    class App {
        <<component>>
        +render() JSX.Element
        -handleLogin(user: User) void
        -handleSignUp(user: User) void
        -handleLogout() void
        -handleNavigation(destination: AppView) void
        -handleUpdateProfile(updatedUser: User) void
    }
    
    %% ===== RELATIONSHIPS =====
    App --> User : uses
    App --> AppView : uses
    App --> Login : renders
    App --> SignUp : renders
    App --> PatientDashboard : renders
    App --> RadiologistDashboard : renders
    App --> DoctorDashboard : renders
    App --> TechDashboard : renders
    App --> Profile : renders
    App --> Settings : renders
    App --> ThemeProvider : wraps
    App --> ErrorBoundary : wraps
    
    Login --> LoginProps : implements
    SignUp --> SignUpProps : implements
    PatientDashboard --> DashboardPropsBase : implements
    RadiologistDashboard --> DashboardPropsBase : implements
    DoctorDashboard --> DashboardPropsBase : implements
    TechDashboard --> DashboardPropsBase : implements
    Profile --> ProfileProps : implements
    Settings --> SettingsProps : implements
    
    UserDropdown --> UserDropdownProps : implements
    ThemeToggle --> ThemeToggleProps : implements
    ThemeCard --> ThemeCardProps : implements
    ErrorBoundary --> ErrorBoundaryProps : implements
    
    ThemeProvider --> ThemeProviderProps : implements
    UseThemeHook --> ThemeContextType : returns
    
    UseResponsiveHook --> ResponsiveHookResult : returns
    UseResponsiveHook --> ResponsiveValues : accepts
    
    ReportStorage --> MedicalReport : manages
    AnalyzeImageRisk --> RiskAssessmentResult : returns
    
    MedicalReport --> RiskScore : uses
    RiskAssessmentResult --> RiskScore : uses
    
    Button --> ButtonProps : implements
    Input --> InputProps : implements
    Card --> CardProps : implements
    Label --> LabelProps : implements
    Badge --> BadgeProps : implements
    Alert --> AlertProps : implements
    Tabs --> TabsProps : implements
    Dialog --> DialogProps : implements
    Select --> SelectProps : implements
    Table --> TableProps : implements
    Progress --> ProgressProps : implements
    Separator --> SeparatorProps : implements
    Avatar --> AvatarProps : implements
    DropdownMenu --> DropdownMenuProps : implements
    
    ToastFunction --> ToastOptions : accepts
    ToastOptions --> ToastAction : contains
```

## API Reference by Category

### 1. Core Types & Interfaces

#### User
```typescript
interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}
```

#### UserRole
```typescript
type UserRole = 'patient' | 'radiologist' | 'doctor' | 'tech' | null;
```

#### AppView
```typescript
type AppView = 'login' | 'signup' | 'dashboard' | 'profile' | 'settings';
```

---

### 2. Authentication Components

#### Login
```typescript
interface LoginProps {
  onLogin: (user: User) => void;
  onSignUpClick: () => void;
}

function Login(props: LoginProps): JSX.Element;
```

#### SignUp
```typescript
interface SignUpProps {
  onSignUp: (user: User) => void;
  onBackToLogin: () => void;
}

function SignUp(props: SignUpProps): JSX.Element;
```

---

### 3. Dashboard Components

#### Base Dashboard Props
```typescript
interface DashboardPropsBase {
  user: User;
  onLogout: () => void;
  onNavigate: (destination: 'dashboard' | 'profile' | 'settings') => void;
}
```

#### PatientDashboard
```typescript
function PatientDashboard(props: DashboardPropsBase): JSX.Element;
```

#### RadiologistDashboard
```typescript
function RadiologistDashboard(props: DashboardPropsBase): JSX.Element;
```

#### DoctorDashboard
```typescript
function DoctorDashboard(props: DashboardPropsBase): JSX.Element;
```

#### TechDashboard
```typescript
function TechDashboard(props: DashboardPropsBase): JSX.Element;
```

---

### 4. Profile & Settings

#### Profile
```typescript
interface ProfileProps {
  user: User;
  onBack: () => void;
  onUpdateProfile: (updatedUser: User) => void;
}

function Profile(props: ProfileProps): JSX.Element;
```

#### Settings
```typescript
interface SettingsProps {
  user: User;
  onBack: () => void;
}

function Settings(props: SettingsProps): JSX.Element;
```

---

### 5. Medical Report Types

#### MedicalReport
```typescript
interface MedicalReport {
  reportId: string;
  patientId: string;
  patientName: string;
  radiologistId: string;
  radiologistName: string;
  reportImageUrl: string;
  studyType: string;
  riskScore: RiskScore;
  confidenceScore: number;
  uploadDate: Date;
  analysisDate: Date;
  findings: string;
  recommendations: string[];
  status: 'pending' | 'analyzed' | 'reviewed';
}
```

#### RiskScore
```typescript
type RiskScore = 'low' | 'medium' | 'high';
```

---

### 6. AI Risk Assessment

#### RiskAssessmentResult
```typescript
interface RiskAssessmentResult {
  riskScore: RiskScore;
  confidenceScore: number;
  findings: string;
  recommendations: string[];
  processingTime: number;
}
```

#### analyzeImageRisk Function
```typescript
function analyzeImageRisk(
  imageUrl: string,
  patientId: string
): Promise<RiskAssessmentResult>;
```

---

### 7. Report Storage API

```typescript
// Save a medical report
function saveReport(report: MedicalReport): void;

// Get reports by patient ID
function getReportsByPatientId(patientId: string): MedicalReport[];

// Get all reports
function getAllReports(): MedicalReport[];

// Get report by ID
function getReportById(reportId: string): MedicalReport | undefined;
```

---

### 8. Theme Context

#### Theme Types
```typescript
type Theme = 'light' | 'dark' | 'system';
```

#### ThemeContext
```typescript
interface ThemeContextType {
  theme: Theme;
  setTheme: (theme: Theme) => void;
  actualTheme: 'light' | 'dark';
}
```

#### ThemeProvider
```typescript
interface ThemeProviderProps {
  children: ReactNode;
  defaultTheme?: Theme;
}

function ThemeProvider(props: ThemeProviderProps): JSX.Element;
```

#### useTheme Hook
```typescript
function useTheme(): ThemeContextType;
```

---

### 9. Responsive Utilities

#### Responsive Hook
```typescript
interface ResponsiveHookResult {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  breakpoint: Breakpoint;
  deviceType: DeviceType;
  isTouch: boolean;
  orientation: 'portrait' | 'landscape';
  width: number;
  height: number;
}

function useResponsive(): ResponsiveHookResult;
```

#### Media Query Hook
```typescript
function useMediaQuery(query: string): boolean;
```

#### Responsive Value Hook
```typescript
interface ResponsiveValues<T> {
  mobile?: T;
  tablet?: T;
  desktop?: T;
}

function useResponsiveValue<T>(values: ResponsiveValues<T>): T;
```

---

### 10. Toast Notifications

```typescript
interface ToastOptions {
  description?: string;
  duration?: number;
  action?: {
    label: string;
    onClick: () => void;
  };
}

// Toast functions
function toast(message: string, options?: ToastOptions): void;
toast.success(message: string, options?: ToastOptions): void;
toast.error(message: string, options?: ToastOptions): void;
toast.info(message: string, options?: ToastOptions): void;
toast.warning(message: string, options?: ToastOptions): void;
```

---

### 11. UI Component Props

#### Button
```typescript
interface ButtonProps {
  variant?: 'default' | 'destructive' | 'outline' | 'secondary' | 'ghost' | 'link';
  size?: 'default' | 'sm' | 'lg' | 'icon';
  asChild?: boolean;
  children: ReactNode;
  className?: string;
  onClick?: () => void;
  disabled?: boolean;
  type?: 'button' | 'submit' | 'reset';
}
```

#### Input
```typescript
interface InputProps {
  type?: string;
  placeholder?: string;
  value?: string;
  onChange?: (e: ChangeEvent<HTMLInputElement>) => void;
  onFocus?: () => void;
  onBlur?: () => void;
  className?: string;
  disabled?: boolean;
  id?: string;
}
```

#### Card Components
```typescript
interface CardProps {
  children: ReactNode;
  className?: string;
}

// Available components:
// Card, CardHeader, CardTitle, CardDescription, CardContent, CardFooter
```

#### Badge
```typescript
interface BadgeProps {
  variant?: 'default' | 'secondary' | 'destructive' | 'outline';
  children: ReactNode;
  className?: string;
}
```

#### Alert
```typescript
interface AlertProps {
  variant?: 'default' | 'destructive';
  children: ReactNode;
  className?: string;
}
```

#### Tabs
```typescript
interface TabsProps {
  defaultValue: string;
  value?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
  className?: string;
}
```

#### Dialog
```typescript
interface DialogProps {
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
}
```

#### Select
```typescript
interface SelectProps {
  value?: string;
  onValueChange?: (value: string) => void;
  children: ReactNode;
  defaultValue?: string;
  disabled?: boolean;
}
```

#### Progress
```typescript
interface ProgressProps {
  value?: number;
  className?: string;
}
```

---

## Usage Examples

### Authentication Flow
```typescript
// Login component usage
<Login 
  onLogin={(user: User) => {
    setCurrentUser(user);
    setCurrentView('dashboard');
  }}
  onSignUpClick={() => setCurrentView('signup')}
/>

// SignUp component usage
<SignUp
  onSignUp={(user: User) => {
    setCurrentUser(user);
    setCurrentView('dashboard');
  }}
  onBackToLogin={() => setCurrentView('login')}
/>
```

### Dashboard Routing
```typescript
// Render appropriate dashboard based on role
switch (user.role) {
  case 'patient':
    return <PatientDashboard user={user} onLogout={handleLogout} onNavigate={handleNavigation} />;
  case 'radiologist':
    return <RadiologistDashboard user={user} onLogout={handleLogout} onNavigate={handleNavigation} />;
  case 'doctor':
    return <DoctorDashboard user={user} onLogout={handleLogout} onNavigate={handleNavigation} />;
  case 'tech':
    return <TechDashboard user={user} onLogout={handleLogout} onNavigate={handleNavigation} />;
}
```

### AI Risk Assessment
```typescript
// Analyze uploaded X-ray image
const result = await analyzeImageRisk(imageUrl, patientId);

// Result contains:
// {
//   riskScore: 'high',
//   confidenceScore: 0.92,
//   findings: 'Abnormality detected...',
//   recommendations: ['Immediate consultation', ...],
//   processingTime: 8500
// }
```

### Report Storage
```typescript
// Save a new report
saveReport({
  reportId: 'R123',
  patientId: 'P001',
  patientName: 'John Doe',
  radiologistId: 'R001',
  radiologistName: 'Dr. Smith',
  reportImageUrl: 'https://...',
  studyType: 'Chest X-Ray',
  riskScore: 'high',
  confidenceScore: 0.92,
  uploadDate: new Date(),
  analysisDate: new Date(),
  findings: 'Abnormality detected',
  recommendations: ['Immediate consultation'],
  status: 'analyzed'
});

// Retrieve reports
const patientReports = getReportsByPatientId('P001');
const allReports = getAllReports();
const specificReport = getReportById('R123');
```

### Theme Management
```typescript
// Use theme context
const { theme, setTheme, actualTheme } = useTheme();

// Change theme
setTheme('dark');  // 'light' | 'dark' | 'system'

// Check actual theme (resolved from system if 'system' is selected)
if (actualTheme === 'dark') {
  // Apply dark mode styles
}
```

### Responsive Hooks
```typescript
// Use responsive hook
const { isMobile, isTablet, isDesktop, breakpoint, deviceType, isTouch, orientation } = useResponsive();

// Responsive values
const columns = useResponsiveValue({
  mobile: 1,
  tablet: 2,
  desktop: 3
});

// Media query
const isLargeScreen = useMediaQuery('(min-width: 1024px)');
```

### Toast Notifications
```typescript
// Show notifications
toast('Report uploaded successfully');
toast.success('Analysis complete', { duration: 5000 });
toast.error('Upload failed', { 
  description: 'Please try again',
  action: {
    label: 'Retry',
    onClick: () => handleRetry()
  }
});
```

---

## Type Safety

All components and functions are fully typed with TypeScript for:
- ✅ Type checking at compile time
- ✅ IntelliSense support in IDEs
- ✅ Better refactoring capabilities
- ✅ Self-documenting code
- ✅ Reduced runtime errors

## API Versioning

**Current Version**: 1.0.0
**Last Updated**: October 2, 2025

All API signatures are stable and follow semantic versioning principles.