import { useState } from 'react';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Separator } from './ui/separator';
import { Badge } from './ui/badge';
import { Eye, EyeOff, Loader2, Stethoscope, Shield, Users, Settings } from 'lucide-react';
import { User, UserRole } from '../App';

interface LoginProps {
  onLogin: (user: User) => void;
  onSignUpClick: () => void;
}

const roleInfo = {
  patient: {
    icon: Users,
    label: 'Patient Portal',
    description: 'View your medical reports and health information',
    color: 'bg-blue-50 text-blue-700 border-blue-200'
  },
  radiologist: {
    icon: Stethoscope,
    label: 'Radiologist Portal', 
    description: 'AI-powered image analysis and reporting',
    color: 'bg-green-50 text-green-700 border-green-200'
  },
  doctor: {
    icon: Shield,
    label: 'Doctor Portal',
    description: 'Review prioritized reports and patient charts',
    color: 'bg-purple-50 text-purple-700 border-purple-200'
  },
  tech: {
    icon: Settings,
    label: 'Tech Team Portal',
    description: 'Data management and AI model monitoring',
    color: 'bg-orange-50 text-orange-700 border-orange-200'
  }
};

export function Login({ onLogin, onSignUpClick }: LoginProps) {
  const [selectedRole, setSelectedRole] = useState<UserRole>(null);
  const [showRoleSelection, setShowRoleSelection] = useState(false);
  const [formData, setFormData] = useState({
    email: '',
    password: ''
  });
  const [showPassword, setShowPassword] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState('');
  const [focusedField, setFocusedField] = useState<string | null>(null);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({ ...prev, [field]: value }));
    if (error) setError(''); // Clear error when user starts typing
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    
    if (!formData.email || !formData.password) {
      setError('Please fill in all required fields');
      return;
    }

    if (!selectedRole) {
      setError('Please select your role');
      return;
    }

    setIsLoading(true);
    setError('');

    // Simulate API call
    try {
      await new Promise(resolve => setTimeout(resolve, 1200));
      
      // Mock user data based on role
      const mockUsers = {
        patient: { id: 'P001', name: 'John Smith', email: formData.email, role: 'patient' as UserRole },
        radiologist: { id: 'R001', name: 'Dr. Sarah Chen', email: formData.email, role: 'radiologist' as UserRole },
        doctor: { id: 'D001', name: 'Dr. Michael Johnson', email: formData.email, role: 'doctor' as UserRole },
        tech: { id: 'T001', name: 'Alex Rodriguez', email: formData.email, role: 'tech' as UserRole }
      };

      onLogin(mockUsers[selectedRole as keyof typeof mockUsers]);
    } catch (err) {
      setError('Login failed. Please check your credentials and try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleRoleSelect = (role: UserRole) => {
    setSelectedRole(role);
    setShowRoleSelection(false);
    setError('');
  };

  return (
    <div 
      className="min-h-screen flex items-center justify-center p-responsive safe-area-inset relative overflow-hidden"
      style={{
        backgroundImage: `linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(147, 51, 234, 0.08) 100%), url('https://images.unsplash.com/photo-1642844613096-7b743b7d9915?crop=entropy&cs=tinysrgb&fit=max&fm=jpg&ixid=M3w3Nzg4Nzd8MHwxfHNlYXJjaHwxfHxtb2Rlcm4lMjBtZWRpY2FsJTIwY2xpbmljJTIwaW50ZXJpb3IlMjBzb2Z0JTIwZm9jdXN8ZW58MXx8fHwxNzU5MzkwMTM0fDA&ixlib=rb-4.1.0&q=80&w=1080')`,
        backgroundSize: 'cover',
        backgroundPosition: 'center',
        backgroundRepeat: 'no-repeat'
      }}
    >
      {/* Background overlay */}
      <div className="absolute inset-0 bg-white/70 backdrop-blur-sm" />
      
      {/* Floating elements for visual interest - responsive visibility */}
      <div className="hidden sm:block absolute top-20 left-20 w-2 h-2 bg-blue-400/30 rounded-full animate-pulse" />
      <div className="hidden md:block absolute top-40 right-32 w-3 h-3 bg-purple-400/20 rounded-full animate-pulse delay-1000" />
      <div className="hidden lg:block absolute bottom-32 left-40 w-2 h-2 bg-green-400/25 rounded-full animate-pulse delay-2000" />
      
      <div className="container-narrow w-full">
        <Card className="modal-responsive shadow-2xl border-0 backdrop-blur-sm bg-white/95 transform transition-all duration-300 hover:shadow-3xl relative z-10">
          <CardHeader className="space-y-1 text-center p-responsive">
            <div className="mx-auto avatar-responsive w-16 h-16 bg-gradient-to-br from-blue-500 to-purple-600 rounded-full flex items-center justify-center shadow-lg mb-4 transform transition-transform duration-300 hover:scale-105 touch-target">
              <Stethoscope className="h-8 w-8 text-white" />
            </div>
            <CardTitle className="text-fluid-2xl font-semibold bg-gradient-to-r from-gray-900 to-gray-600 bg-clip-text text-transparent">
              Medical Portal
            </CardTitle>
            <CardDescription className="text-fluid-base text-gray-600">
              Secure access to your healthcare platform
            </CardDescription>
        </CardHeader>

          <CardContent className="space-responsive p-responsive">
          {!showRoleSelection ? (
            <form onSubmit={handleLogin} className="space-y-5">
              {/* Email Field */}
              <div className="space-y-2">
                <Label htmlFor="email" className="text-sm font-medium text-black">
                  Email Address
                </Label>
                <div className={`relative transition-all duration-200 ${
                  focusedField === 'email' ? 'transform scale-[1.02]' : ''
                }`}>
                  <Input
                    id="email"
                    type="email"
                    placeholder="Enter your email"
                    value={formData.email}
                    onChange={(e) => handleInputChange('email', e.target.value)}
                    onFocus={() => setFocusedField('email')}
                    onBlur={() => setFocusedField(null)}
                    className={`transition-all duration-200 ${
                      focusedField === 'email' 
                        ? 'border-blue-500 ring-4 ring-blue-500/10 shadow-md' 
                        : 'border-gray-200 hover:border-gray-300'
                    } ${error && !formData.email ? 'border-red-300 ring-red-500/10' : ''}`}
                    disabled={isLoading}
                  />
                </div>
              </div>

              {/* Password Field */}
              <div className="space-y-2">
                <Label htmlFor="password" className="text-sm font-medium text-black">
                  Password
                </Label>
                <div className={`relative transition-all duration-200 ${
                  focusedField === 'password' ? 'transform scale-[1.02]' : ''
                }`}>
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={formData.password}
                    onChange={(e) => handleInputChange('password', e.target.value)}
                    onFocus={() => setFocusedField('password')}
                    onBlur={() => setFocusedField(null)}
                    className={`pr-12 transition-all duration-200 ${
                      focusedField === 'password' 
                        ? 'border-blue-500 ring-4 ring-blue-500/10 shadow-md' 
                        : 'border-gray-200 hover:border-gray-300'
                    } ${error && !formData.password ? 'border-red-300 ring-red-500/10' : ''}`}
                    disabled={isLoading}
                  />
                  <button
                    type="button"
                    onClick={() => setShowPassword(!showPassword)}
                    className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 transition-colors duration-200 p-1 rounded-full hover:bg-gray-100"
                    disabled={isLoading}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </button>
                </div>
              </div>

              {/* Role Selection */}
              {selectedRole && (
                <div className="space-y-2">
                  <Label className="text-sm font-medium text-gray-700">Selected Role</Label>
                  <div className={`p-3 rounded-lg border-2 ${roleInfo[selectedRole].color} transition-all duration-200`}>
                    <div className="flex items-center gap-3">
                      {(() => {
                        const IconComponent = roleInfo[selectedRole].icon;
                        return IconComponent ? <IconComponent className="h-5 w-5" /> : null;
                      })()}
                      <div>
                        <div className="font-medium">{roleInfo[selectedRole].label}</div>
                        <div className="text-xs opacity-80">{roleInfo[selectedRole].description}</div>
                      </div>
                    </div>
                  </div>
                  <button
                    type="button"
                    onClick={() => setShowRoleSelection(true)}
                    className="text-sm text-blue-600 hover:text-blue-700 transition-colors duration-200"
                  >
                    Change role
                  </button>
                </div>
              )}

              {/* Error Message */}
              {error && (
                <div className="p-3 rounded-lg bg-red-50 border border-red-200 text-red-700 text-sm animate-in slide-in-from-top-2 duration-200">
                  {error}
                </div>
              )}

              {/* Action Buttons */}
              <div className="space-y-4 pt-2">
                {!selectedRole ? (
                  <Button
                    type="button"
                    onClick={() => setShowRoleSelection(true)}
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg transition-all duration-200 transform hover:scale-[1.02] hover:shadow-xl"
                    disabled={isLoading}
                  >
                    Select Your Role
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    className="w-full bg-gradient-to-r from-blue-500 to-purple-600 hover:from-blue-600 hover:to-purple-700 text-white shadow-lg transition-all duration-200 transform hover:scale-[1.02] hover:shadow-xl disabled:transform-none disabled:scale-100"
                    disabled={isLoading}
                  >
                    {isLoading ? (
                      <div className="flex items-center gap-2">
                        <Loader2 className="h-4 w-4 animate-spin" />
                        Signing In...
                      </div>
                    ) : (
                      'Sign In'
                    )}
                  </Button>
                )}

                <button
                  type="button"
                  className="w-full text-sm text-gray-600 hover:text-gray-800 transition-colors duration-200 p-2 rounded-lg hover:bg-gray-50"
                  disabled={isLoading}
                >
                  Forgot your password?
                </button>

                {/* Sign Up Link */}
                <div className="text-center pt-2">
                  <p className="text-sm text-gray-600">
                    Don't have an account?{' '}
                    <button
                      type="button"
                      onClick={onSignUpClick}
                      className="text-blue-600 hover:text-blue-700 font-medium transition-colors hover:underline"
                      disabled={isLoading}
                    >
                      Sign Up
                    </button>
                  </p>
                </div>
              </div>
            </form>
          ) : (
            /* Role Selection Screen */
            <div className="space-y-4">
              <div className="text-center">
                <h3 className="text-lg font-semibold text-gray-900 mb-2">Select Your Role</h3>
                <p className="text-sm text-gray-600">Choose the portal that matches your access level</p>
              </div>

              <div className="grid gap-3">
                {(Object.keys(roleInfo) as UserRole[]).filter(Boolean).map((role) => {
                  if (!role) return null;
                  const info = roleInfo[role];
                  return (
                    <button
                      key={role}
                      onClick={() => handleRoleSelect(role)}
                      className={`p-4 rounded-xl border-2 text-left transition-all duration-200 transform hover:scale-[1.02] hover:shadow-lg ${info.color} hover:shadow-lg group`}
                    >
                      <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-white/50 group-hover:bg-white/80 transition-colors duration-200">
                          {(() => {
                            const IconComponent = info.icon;
                            return <IconComponent className="h-5 w-5" />;
                          })()}
                        </div>
                        <div className="flex-1">
                          <div className="font-medium">{info.label}</div>
                          <div className="text-sm opacity-80">{info.description}</div>
                        </div>
                      </div>
                    </button>
                  );
                })}
              </div>

              <Button
                type="button"
                variant="outline"
                onClick={() => setShowRoleSelection(false)}
                className="w-full border-gray-200 hover:bg-gray-50 transition-all duration-200"
              >
                Back to Login
              </Button>
            </div>
          )}

          {/* Demo Credentials */}
          {!showRoleSelection && (
            <>
              <Separator className="my-6" />
              <div className="text-center space-y-3">
                <Badge variant="secondary" className="text-xs px-3 py-1">
                  Demo Credentials
                </Badge>
                <div className="text-xs text-gray-500 space-y-1">
                  <p><strong>Email:</strong> demo@hospital.com</p>
                  <p><strong>Password:</strong> demo123</p>
                  <p className="italic">Use any credentials to access the demo</p>
                </div>
              </div>
            </>
          )}
          </CardContent>
        </Card>
      </div>
    </div>
  );
}