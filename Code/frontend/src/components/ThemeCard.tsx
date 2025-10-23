import { Card, CardProps } from './ui/card';
import { useTheme } from '../utils/themeContext';

interface ThemeCardProps extends CardProps {
  variant?: 'default' | 'bordered' | 'elevated';
}

export function ThemeCard({ variant = 'default', className = '', children, ...props }: ThemeCardProps) {
  const { actualTheme } = useTheme();
  const isDarkMode = actualTheme === 'dark';

  const variantClasses = {
    default: isDarkMode ? 'bg-gray-800 border-gray-700' : 'bg-white border-gray-200',
    bordered: isDarkMode ? 'bg-gray-800 border-gray-600 border-2' : 'bg-white border-gray-300 border-2',
    elevated: isDarkMode ? 'bg-gray-800 border-gray-700 shadow-xl' : 'bg-white border-gray-200 shadow-xl',
  };

  return (
    <Card 
      className={`${variantClasses[variant]} ${className}`}
      {...props}
    >
      {children}
    </Card>
  );
}
