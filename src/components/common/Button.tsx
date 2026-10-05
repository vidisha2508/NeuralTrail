import React from 'react';

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'danger' | 'vaporwave';
  size?: 'sm' | 'md' | 'lg';
  icon?: React.ReactNode;
  children: React.ReactNode;
}

export const Button: React.FC<ButtonProps> = ({
  variant = 'secondary',
  size = 'md',
  icon,
  children,
  className = '',
  disabled,
  ...props
}) => {
  const baseStyles = 'inline-flex items-center justify-center font-mono font-medium transition-all duration-200 focus:outline-none disabled:opacity-50 disabled:cursor-not-allowed select-none rounded';

  const sizeStyles = {
    sm: 'text-xs px-2.5 py-1.5 gap-1.5',
    md: 'text-xs px-3.5 py-2 gap-2',
    lg: 'text-sm px-4 py-2.5 gap-2.5',
  };

  const variantStyles = {
    primary: 'bg-accent-cyan text-[#07090D] hover:bg-accent-cyan/90 font-semibold shadow-glow-cyan active:translate-y-px',
    secondary: 'bg-bg-card hover:bg-bg-subtle text-text-primary border border-border-subtle hover:border-border-active active:translate-y-px',
    outline: 'bg-transparent text-text-secondary hover:text-text-primary border border-border-subtle hover:border-accent-cyan/40',
    danger: 'bg-failure-dark/60 text-failure border border-failure/40 hover:bg-failure-dark/90 shadow-glow-failure',
    vaporwave: 'bg-gradient-to-r from-vaporwave-purple to-vaporwave-pink text-white font-semibold shadow-glow-purple hover:opacity-95 active:translate-y-px',
  };

  return (
    <button
      className={`${baseStyles} ${sizeStyles[size]} ${variantStyles[variant]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
    </button>
  );
};
