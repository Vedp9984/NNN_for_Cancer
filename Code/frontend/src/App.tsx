import { useState } from 'react';
import { Login } from './components/Login';
import { SignUp } from './components/SignUp';
import { Profile } from './components/Profile';
import { Settings } from './components/Settings';
import { PatientDashboard } from './components/PatientDashboard';
import { RadiologistDashboard } from './components/RadiologistDashboard';
import { DoctorDashboard } from './components/DoctorDashboard';
import { TechDashboard } from './components/TechDashboard';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Toaster } from './components/ui/sonner';
import { ThemeProvider } from './utils/themeContext';

export type UserRole = 'patient' | 'radiologist' | 'doctor' | 'tech' | null;
export type AppView = 'login' | 'signup' | 'dashboard' | 'profile' | 'settings';

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
}

export default function App() {
  const [currentUser, setCurrentUser] = useState<User | null>(null);
  const [currentView, setCurrentView] = useState<AppView>('login');
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = (user: User) => {
    setIsLoading(true);
    try {
      setCurrentUser(user);
      setCurrentView('dashboard'); // Navigate to dashboard after login
    } catch (error) {
      console.error('Login error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleSignUp = (user: User) => {
    setIsLoading(true);
    try {
      setCurrentUser(user);
      setCurrentView('dashboard'); // Navigate to dashboard after signup
    } catch (error) {
      console.error('SignUp error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleLogout = () => {
    setIsLoading(true);
    try {
      setCurrentUser(null);
      setCurrentView('login'); // Navigate back to login
    } catch (error) {
      console.error('Logout error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleNavigation = (destination: 'dashboard' | 'profile' | 'settings') => {
    setCurrentView(destination);
  };

  const handleUpdateProfile = (updatedUser: User) => {
    setCurrentUser(updatedUser);
  };

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-background">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto mb-4"></div>
          <p className="text-muted-foreground">Loading...</p>
        </div>
      </div>
    );
  }

  // Handle different views based on current state
  const renderCurrentView = () => {
    if (!currentUser) {
      // User not logged in - show login or signup
      if (currentView === 'signup') {
        return (
          <SignUp 
            onSignUp={handleSignUp} 
            onBackToLogin={() => setCurrentView('login')} 
          />
        );
      }
      return (
        <Login 
          onLogin={handleLogin} 
          onSignUpClick={() => setCurrentView('signup')} 
        />
      );
    }

    // User logged in - show appropriate view
    switch (currentView) {
      case 'profile':
        return (
          <Profile 
            user={currentUser} 
            onBack={() => setCurrentView('dashboard')}
            onUpdateProfile={handleUpdateProfile}
          />
        );
      
      case 'settings':
        return (
          <Settings 
            user={currentUser} 
            onBack={() => setCurrentView('dashboard')}
          />
        );
      
      case 'dashboard':
      default:
        // Role-based dashboard routing
        switch (currentUser.role) {
          case 'patient':
            return (
              <PatientDashboard 
                user={currentUser} 
                onLogout={handleLogout}
                onNavigate={handleNavigation}
              />
            );
          case 'radiologist':
            return (
              <RadiologistDashboard 
                user={currentUser} 
                onLogout={handleLogout}
                onNavigate={handleNavigation}
              />
            );
          case 'doctor':
            return (
              <DoctorDashboard 
                user={currentUser} 
                onLogout={handleLogout}
                onNavigate={handleNavigation}
              />
            );
          case 'tech':
            return (
              <TechDashboard 
                user={currentUser} 
                onLogout={handleLogout}
                onNavigate={handleNavigation}
              />
            );
          default:
            return <div>Unknown role</div>;
        }
    }
  };

  return (
    <ThemeProvider>
      <ErrorBoundary>
        <div className="min-h-screen bg-background">
          {renderCurrentView()}
        </div>
        <Toaster />
      </ErrorBoundary>
    </ThemeProvider>
  );
}