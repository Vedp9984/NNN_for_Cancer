import { useState, useEffect } from 'react';

export interface ResponsiveBreakpoints {
  xs: boolean;  // < 640px
  sm: boolean;  // >= 640px
  md: boolean;  // >= 768px
  lg: boolean;  // >= 1024px
  xl: boolean;  // >= 1280px
  '2xl': boolean; // >= 1536px
}

export interface DeviceInfo {
  isMobile: boolean;
  isTablet: boolean;
  isDesktop: boolean;
  isTouchDevice: boolean;
  orientation: 'portrait' | 'landscape';
  screenSize: 'xs' | 'sm' | 'md' | 'lg' | 'xl' | '2xl';
}

export function useResponsive(): ResponsiveBreakpoints & DeviceInfo {
  const [breakpoints, setBreakpoints] = useState<ResponsiveBreakpoints>({
    xs: false,
    sm: false,
    md: false,
    lg: false,
    xl: false,
    '2xl': false,
  });

  const [deviceInfo, setDeviceInfo] = useState<DeviceInfo>({
    isMobile: false,
    isTablet: false,
    isDesktop: false,
    isTouchDevice: false,
    orientation: 'portrait',
    screenSize: 'md',
  });

  useEffect(() => {
    const updateBreakpoints = () => {
      const width = window.innerWidth;
      const height = window.innerHeight;
      
      const newBreakpoints: ResponsiveBreakpoints = {
        xs: width < 640,
        sm: width >= 640,
        md: width >= 768,
        lg: width >= 1024,
        xl: width >= 1280,
        '2xl': width >= 1536,
      };

      // Determine screen size
      let screenSize: DeviceInfo['screenSize'] = 'md';
      if (width < 640) screenSize = 'xs';
      else if (width >= 640 && width < 768) screenSize = 'sm';
      else if (width >= 768 && width < 1024) screenSize = 'md';
      else if (width >= 1024 && width < 1280) screenSize = 'lg';
      else if (width >= 1280 && width < 1536) screenSize = 'xl';
      else screenSize = '2xl';

      // Device type detection
      const isMobile = width < 768;
      const isTablet = width >= 768 && width < 1024;
      const isDesktop = width >= 1024;
      
      // Touch device detection
      const isTouchDevice = 'ontouchstart' in window || navigator.maxTouchPoints > 0;
      
      // Orientation detection
      const orientation = height > width ? 'portrait' : 'landscape';

      const newDeviceInfo: DeviceInfo = {
        isMobile,
        isTablet,
        isDesktop,
        isTouchDevice,
        orientation,
        screenSize,
      };

      setBreakpoints(newBreakpoints);
      setDeviceInfo(newDeviceInfo);
    };

    // Update on mount
    updateBreakpoints();

    // Listen for resize events
    const handleResize = () => {
      updateBreakpoints();
    };

    // Listen for orientation changes
    const handleOrientationChange = () => {
      // Timeout to ensure correct dimensions after orientation change
      setTimeout(updateBreakpoints, 100);
    };

    window.addEventListener('resize', handleResize);
    window.addEventListener('orientationchange', handleOrientationChange);
    
    // Clean up event listeners
    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('orientationchange', handleOrientationChange);
    };
  }, []);

  return {
    ...breakpoints,
    ...deviceInfo,
  };
}

// Utility function to get responsive values
export function useResponsiveValue<T>(values: Partial<Record<keyof ResponsiveBreakpoints | 'base', T>>): T {
  const responsive = useResponsive();
  
  // Priority order: base < xs < sm < md < lg < xl < 2xl
  if (responsive['2xl'] && values['2xl'] !== undefined) return values['2xl'];
  if (responsive.xl && values.xl !== undefined) return values.xl;
  if (responsive.lg && values.lg !== undefined) return values.lg;
  if (responsive.md && values.md !== undefined) return values.md;
  if (responsive.sm && values.sm !== undefined) return values.sm;
  if (responsive.xs && values.xs !== undefined) return values.xs;
  
  // Fallback to base value
  return values.base as T;
}

// Hook for conditional rendering based on screen size
export function useMediaQuery(query: string): boolean {
  const [matches, setMatches] = useState(false);

  useEffect(() => {
    const media = window.matchMedia(query);
    
    const updateMatch = () => {
      setMatches(media.matches);
    };
    
    updateMatch();
    
    // Use the newer addEventListener if available
    if (media.addEventListener) {
      media.addEventListener('change', updateMatch);
      return () => media.removeEventListener('change', updateMatch);
    } else {
      // Fallback for older browsers
      media.addListener(updateMatch);
      return () => media.removeListener(updateMatch);
    }
  }, [query]);

  return matches;
}

// Predefined media queries
export const mediaQueries = {
  mobile: '(max-width: 767px)',
  tablet: '(min-width: 768px) and (max-width: 1023px)',
  desktop: '(min-width: 1024px)',
  touchDevice: '(hover: none) and (pointer: coarse)',
  landscape: '(orientation: landscape)',
  portrait: '(orientation: portrait)',
  reducedMotion: '(prefers-reduced-motion: reduce)',
  highContrast: '(prefers-contrast: high)',
  darkMode: '(prefers-color-scheme: dark)',
} as const;