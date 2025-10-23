import { useTheme } from '../utils/themeContext';
import { Button } from './ui/button';
import { DropdownMenuItem } from './ui/dropdown-menu';
import { Sun, Moon, Monitor, Check } from 'lucide-react';

interface ThemeToggleProps {
  inDropdown?: boolean;
  isDarkMode?: boolean;
}

export function ThemeToggle({ inDropdown = false, isDarkMode = false }: ThemeToggleProps) {
  const { theme, setTheme } = useTheme();

  const themes = [
    {
      value: 'light' as const,
      label: 'Light',
      icon: Sun,
      description: 'Light theme'
    },
    {
      value: 'dark' as const,
      label: 'Dark',
      icon: Moon,
      description: 'Dark theme'
    },
    {
      value: 'system' as const,
      label: 'System',
      icon: Monitor,
      description: 'Follow system preference'
    }
  ];

  if (inDropdown) {
    return (
      <div className="px-2 py-1">
        <div className={`text-xs font-medium mb-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
          Appearance
        </div>
        {themes.map((themeOption) => {
          const IconComponent = themeOption.icon;
          return (
            <DropdownMenuItem
              key={themeOption.value}
              onClick={() => setTheme(themeOption.value)}
              className={`cursor-pointer flex items-center justify-between ${
                isDarkMode ? 'text-white hover:bg-gray-700' : ''
              }`}
            >
              <div className="flex items-center">
                <IconComponent className="mr-2 h-4 w-4" />
                <span>{themeOption.label}</span>
              </div>
              {theme === themeOption.value && (
                <Check className="h-4 w-4" />
              )}
            </DropdownMenuItem>
          );
        })}
      </div>
    );
  }

  // Standalone theme toggle button (for use outside dropdown)
  const currentTheme = themes.find(t => t.value === theme);
  const CurrentIcon = currentTheme?.icon || Monitor;

  return (
    <Button
      variant="ghost"
      size="sm"
      onClick={() => {
        const currentIndex = themes.findIndex(t => t.value === theme);
        const nextIndex = (currentIndex + 1) % themes.length;
        setTheme(themes[nextIndex].value);
      }}
      className="h-9 w-9 px-0"
      title={`Current: ${currentTheme?.label}. Click to cycle themes.`}
    >
      <CurrentIcon className="h-4 w-4" />
      <span className="sr-only">Toggle theme</span>
    </Button>
  );
}