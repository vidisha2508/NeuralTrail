import React from 'react';

interface MetricProps {
  label: string;
  value: string | number;
  subValue?: string;
  badge?: React.ReactNode;
  trend?: 'up' | 'down' | 'neutral';
  accent?: 'green' | 'red' | 'cyan' | 'yellow' | 'purple';
  icon?: React.ReactNode;
}

export const Metric: React.FC<MetricProps> = ({
  label,
  value,
  subValue,
  badge,
  trend,
  accent = 'cyan',
  icon,
}) => {
  const accentColors = {
    green: 'text-stable glow-text-green',
    red: 'text-failure glow-text-red',
    cyan: 'text-accent-cyan glow-text-cyan',
    yellow: 'text-uncertain',
    purple: 'text-vaporwave-purple glow-text-purple',
  };

  return (
    <div className="flex flex-col py-1 px-3 border-r last:border-r-0 border-border-subtle/80">
      <div className="flex items-center gap-1.5 text-text-dim text-[11px] font-mono uppercase tracking-wider">
        {icon && <span className="opacity-75">{icon}</span>}
        <span>{label}</span>
      </div>
      <div className="flex items-baseline gap-2 mt-0.5">
        <span className={`text-xl font-heading font-semibold tracking-tight ${accentColors[accent]}`}>
          {value}
        </span>
        {subValue && (
          <span className="text-xs font-mono text-text-secondary flex items-center gap-0.5">
            {trend === 'up' && <span className="text-stable">↑</span>}
            {trend === 'down' && <span className="text-failure">↓</span>}
            {subValue}
          </span>
        )}
        {badge && <div className="ml-auto">{badge}</div>}
      </div>
    </div>
  );
};
