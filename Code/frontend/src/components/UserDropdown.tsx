import { useState } from 'react';
import { Button } from './ui/button';
import { DropdownMenu, DropdownMenuContent, DropdownMenuItem, DropdownMenuSeparator, DropdownMenuTrigger } from './ui/dropdown-menu';
import { Avatar, AvatarFallback } from './ui/avatar';
import { User, Settings, LogOut, LayoutDashboard } from 'lucide-react';
import { User as UserType } from '../App';
import { ThemeToggle } from './ThemeToggle';

interface UserDropdownProps {
  user: UserType;
  onLogout: () => void;
  onNavigate: (destination: 'dashboard' | 'profile' | 'settings') => void;
  isDarkMode?: boolean;
}

export function UserDropdown({ user, onLogout, onNavigate, isDarkMode = false }: UserDropdownProps) {
  const [currentTab, setCurrentTab] = useState('dashboard');

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

  const handleTabClick = (tab: string) => {
    setCurrentTab(tab);
    // In a real app, this would navigate to different sections
  };

  return (
    <DropdownMenu>
      <DropdownMenuTrigger asChild>
        <Button 
          variant="ghost" 
          className="relative h-10 w-10 rounded-full p-0"
          aria-label={`${user.name} user menu`}
        >
          <Avatar className="h-10 w-10">
            <AvatarFallback className={`${getBadgeColor(user.role)} text-white font-medium text-sm`}>
              {getUserInitials(user.name)}
            </AvatarFallback>
          </Avatar>
        </Button>
      </DropdownMenuTrigger>
      <DropdownMenuContent 
        className={`w-56 ${isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white'}`} 
        align="end" 
        forceMount
      >
        <div className={`px-3 py-2 ${isDarkMode ? 'text-white' : 'text-gray-900'}`}>
          <div className="flex flex-col space-y-1">
            <p className="text-sm font-medium">{user.name}</p>
            <p className={`text-xs ${isDarkMode ? 'text-gray-300' : 'text-gray-500'}`}>
              {user.email || `${user.id}@medportal.com`}
            </p>
            <p className={`text-xs capitalize ${isDarkMode ? 'text-gray-400' : 'text-gray-500'}`}>
              {user.role} Portal • ID: {user.id}
            </p>
          </div>
        </div>
        <DropdownMenuSeparator className={isDarkMode ? 'bg-gray-700' : ''} />
        
        <DropdownMenuItem 
          onClick={() => onNavigate('dashboard')}
          className={`cursor-pointer ${isDarkMode ? 'text-white hover:bg-gray-700' : ''}`}
        >
          <LayoutDashboard className="mr-2 h-4 w-4" />
          <span>Dashboard</span>
        </DropdownMenuItem>
        
        <DropdownMenuItem 
          onClick={() => onNavigate('profile')}
          className={`cursor-pointer ${isDarkMode ? 'text-white hover:bg-gray-700' : ''}`}
        >
          <User className="mr-2 h-4 w-4" />
          <span>My Profile</span>
        </DropdownMenuItem>
        
        <DropdownMenuItem 
          onClick={() => onNavigate('settings')}
          className={`cursor-pointer ${isDarkMode ? 'text-white hover:bg-gray-700' : ''}`}
        >
          <Settings className="mr-2 h-4 w-4" />
          <span>Settings</span>
        </DropdownMenuItem>
        
        <DropdownMenuSeparator className={isDarkMode ? 'bg-gray-700' : ''} />
        
        {/* Theme Toggle Section */}
        <ThemeToggle inDropdown={true} isDarkMode={isDarkMode} />
        
        <DropdownMenuSeparator className={isDarkMode ? 'bg-gray-700' : ''} />
        
        <DropdownMenuItem 
          onClick={onLogout}
          className={`cursor-pointer ${isDarkMode ? 'text-red-400 hover:bg-gray-700 hover:text-red-300' : 'text-red-600 hover:text-red-700'}`}
        >
          <LogOut className="mr-2 h-4 w-4" />
          <span>Logout</span>
        </DropdownMenuItem>
      </DropdownMenuContent>
    </DropdownMenu>
  );
}