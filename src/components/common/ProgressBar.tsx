import React from 'react';

export type ProgressAccent = 'cyan' | 'green' | 'magenta' | 'amber' | 'purple';

interface ProgressBarProps {
  value: number; // 0 to 100
  accent?: ProgressAccent;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  label?: string;
  className?: string;
}

export const ProgressBar: React.FC<ProgressBarProps> = ({
  value,
  accent = 'cyan',
  size = 'md',
  showLabel = false,
  label,
  className = '',
}) => {
  const clamped = Math.max(0, Math.min(100, isNaN(value) ? 0 : value));

  const heightClasses = {
    sm: 'h-1.5',
    md: 'h-2.5',
    lg: 'h-4',
  };

  const barColors: Record<ProgressAccent, string> = {
    cyan: 'bg-[#00f0ff]',
    green: 'bg-[#00ff88]',
    magenta: 'bg-[#ff007f]',
    amber: 'bg-[#ffb300]',
    purple: 'bg-[#b388ff]',
  };

  return (
    <div className={`w-full ${className}`}>
      {(showLabel || label) && (
        <div className="flex justify-between items-center mb-1 text-xs font-mono">
          <span className="text-white/60">{label}</span>
          <span className="text-white font-medium">{clamped.toFixed(0)}%</span>
        </div>
      )}
      <div className={`w-full bg-white/5 rounded-full overflow-hidden border border-white/10 ${heightClasses[size]}`}>
        <div
          className={`h-full rounded-full transition-all duration-300 ${barColors[accent]}`}
          style={{ width: `${clamped}%` }}
        />
      </div>
    </div>
  );
};
