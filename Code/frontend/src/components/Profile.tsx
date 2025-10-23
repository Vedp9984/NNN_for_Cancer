import { useState } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from './ui/card';
import { Button } from './ui/button';
import { Input } from './ui/input';
import { Label } from './ui/label';
import { Badge } from './ui/badge';
import { Avatar, AvatarFallback } from './ui/avatar';
import { Alert, AlertDescription } from './ui/alert';
import { ArrowLeft, User, Mail, Save, Edit3, CheckCircle } from 'lucide-react';
import { User as UserType } from '../App';
import { useTheme } from '../utils/themeContext';

interface ProfileProps {
  user: UserType;
  onBack: () => void;
  onUpdateProfile: (updatedUser: UserType) => void;
}

export function Profile({ user, onBack, onUpdateProfile }: ProfileProps) {
  const { actualTheme } = useTheme();
  const isDarkMode = actualTheme === 'dark';
  const [isEditing, setIsEditing] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [success, setSuccess] = useState(false);
  const [formData, setFormData] = useState({
    name: user.name,
    email: user.email
  });

  const getUserInitials = (name: string) => {
    return name
      .split(' ')
      .map(word => word[0])
      .join('')
      .toUpperCase()
      .slice(0, 2);
  };

  const getBadgeColor = (role: string | null) => {
    switch (role) {
      case 'patient': return 'bg-blue-600';
      case 'radiologist': return 'bg-green-600';
      case 'doctor': return 'bg-purple-600';
      case 'tech': return 'bg-orange-600';
      default: return 'bg-gray-600';
    }
  };

  const getRoleDisplayName = (role: string | null) => {
    switch (role) {
      case 'patient': return 'Patient';
      case 'radiologist': return 'Radiologist';
      case 'doctor': return 'Doctor';
      case 'tech': return 'Tech Team';
      default: return 'Unknown';
    }
  };

  const handleSave = async () => {
    setIsLoading(true);
    setSuccess(false);

    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      const updatedUser: UserType = {
        ...user,
        name: formData.name,
        email: formData.email
      };

      onUpdateProfile(updatedUser);
      setIsEditing(false);
      setSuccess(true);
      
      // Hide success message after 3 seconds
      setTimeout(() => setSuccess(false), 3000);
    } catch (error) {
      console.error('Failed to update profile:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleCancel = () => {
    setFormData({
      name: user.name,
      email: user.email
    });
    setIsEditing(false);
  };

  return (
    <div className={`min-h-screen ${isDarkMode ? 'bg-gray-900' : 'bg-gray-50'} p-responsive safe-area-inset`}>
      <div className="container-narrow mx-auto">
        {/* Back Button */}
        <div className="mb-6">
          <Button
            variant="ghost"
            onClick={onBack}
            className={`flex items-center space-responsive transition-colors touch-target ${
              isDarkMode ? 'text-gray-300 hover:text-white' : 'text-gray-600 hover:text-gray-900'
            }`}
          >
            <ArrowLeft className="h-4 w-4" />
            Back to Dashboard
          </Button>
        </div>

        {/* Success Alert */}
        {success && (
          <Alert className={`mb-6 animate-in slide-in-from-top-2 ${
            isDarkMode 
              ? 'border-green-700 bg-green-900/20 text-green-400' 
              : 'border-green-200 bg-green-50'
          }`}>
            <CheckCircle className="h-4 w-4 text-green-600" />
            <AlertDescription className={isDarkMode ? 'text-green-400' : 'text-green-700'}>
              Profile updated successfully!
            </AlertDescription>
          </Alert>
        )}

        {/* Profile Card */}
        <Card>
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle className="text-2xl">My Profile</CardTitle>
                <CardDescription>
                  View and edit your personal information
                </CardDescription>
              </div>
              {!isEditing ? (
                <Button
                  onClick={() => setIsEditing(true)}
                  variant="outline"
                  className="flex items-center gap-2"
                >
                  <Edit3 className="h-4 w-4" />
                  Edit Profile
                </Button>
              ) : (
                <div className="flex gap-2">
                  <Button
                    onClick={handleCancel}
                    variant="outline"
                    disabled={isLoading}
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleSave}
                    disabled={isLoading}
                    className="flex items-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <div className="h-4 w-4 border-2 border-current border-t-transparent rounded-full animate-spin" />
                        Saving...
                      </>
                    ) : (
                      <>
                        <Save className="h-4 w-4" />
                        Save Changes
                      </>
                    )}
                  </Button>
                </div>
              )}
            </div>
          </CardHeader>
          <CardContent className="space-y-6">
            {/* Profile Picture Section */}
            <div className="flex items-center gap-6">
              <Avatar className="h-20 w-20">
                <AvatarFallback className={`${getBadgeColor(user.role)} text-white text-xl font-medium`}>
                  {getUserInitials(user.name)}
                </AvatarFallback>
              </Avatar>
              <div>
                <h3 className="text-lg font-medium">{user.name}</h3>
                <p className="text-muted-foreground mb-2">{user.email}</p>
                <Badge className={`${getBadgeColor(user.role)} text-white`}>
                  {getRoleDisplayName(user.role)}
                </Badge>
              </div>
            </div>

            {/* Form Fields */}
            <div className="grid gap-6 pt-6 border-t">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
                <div className="space-y-2">
                  <Label htmlFor="userId">User ID</Label>
                  <Input
                    id="userId"
                    value={user.id}
                    disabled
                    className="bg-muted"
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="role">Role</Label>
                  <Input
                    id="role"
                    value={getRoleDisplayName(user.role)}
                    disabled
                    className="bg-muted"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="name">Full Name</Label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="name"
                    value={formData.name}
                    onChange={(e) => setFormData(prev => ({ ...prev, name: e.target.value }))}
                    disabled={!isEditing}
                    className={`pl-10 ${!isEditing ? 'bg-muted' : ''}`}
                    placeholder="Enter your full name"
                  />
                </div>
              </div>

              <div className="space-y-2">
                <Label htmlFor="email">Email Address</Label>
                <div className="relative">
                  <Mail className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    id="email"
                    type="email"
                    value={formData.email}
                    onChange={(e) => setFormData(prev => ({ ...prev, email: e.target.value }))}
                    disabled={!isEditing}
                    className={`pl-10 ${!isEditing ? 'bg-muted' : ''}`}
                    placeholder="Enter your email address"
                  />
                </div>
              </div>
            </div>

            {/* Additional Information */}
            <div className="pt-6 border-t">
              <h4 className="font-medium mb-4">Account Information</h4>
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-muted-foreground">Account Created</p>
                  <p className="font-medium">January 15, 2024</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Last Login</p>
                  <p className="font-medium">Today, 2:30 PM</p>
                </div>
                <div>
                  <p className="text-muted-foreground">Account Status</p>
                  <Badge variant="outline" className="text-green-600 border-green-200">
                    Active
                  </Badge>
                </div>
                <div>
                  <p className="text-muted-foreground">Portal Access</p>
                  <p className="font-medium capitalize">{user.role} Portal</p>
                </div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}