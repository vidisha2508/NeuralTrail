import React from 'react';

export type MetricAccent = 'cyan' | 'green' | 'magenta' | 'amber' | 'purple' | 'neutral';

interface MetricCardProps {
  label: string;
  value: string | number;
  subValue?: string;
  change?: string;
  trend?: 'up' | 'down' | 'neutral';
  accent?: MetricAccent;
  icon?: React.ReactNode;
  badge?: React.ReactNode;
  className?: string;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  label,
  value,
  subValue,
  change,
  trend,
  accent = 'cyan',
  icon,
  badge,
  className = '',
}) => {
  const accentTextColors: Record<MetricAccent, string> = {
    cyan: 'text-[#00f0ff]',
    green: 'text-[#00ff88]',
    magenta: 'text-[#ff007f]',
    amber: 'text-[#ffb300]',
    purple: 'text-[#b388ff]',
    neutral: 'text-white',
  };

  const accentBorderColors: Record<MetricAccent, string> = {
    cyan: 'border-l-[#00f0ff]/70',
    green: 'border-l-[#00ff88]/70',
    magenta: 'border-l-[#ff007f]/70',
    amber: 'border-l-[#ffb300]/70',
    purple: 'border-l-[#b388ff]/70',
    neutral: 'border-l-white/20',
  };

  return (
    <div
      className={`rounded-lg bg-[#150a29]/90 border border-white/10 p-4 border-l-2 ${accentBorderColors[accent]} backdrop-blur-md transition-all hover:border-white/20 ${className}`}
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5 text-white/50 text-xs font-sans font-medium tracking-wide">
          {icon && <span className="text-white/60">{icon}</span>}
          <span>{label}</span>
        </div>
        {badge}
      </div>

      <div className="flex items-baseline justify-between gap-2">
        <span className={`text-2xl lg:text-3xl font-mono font-semibold tracking-tight ${accentTextColors[accent]}`}>
          {value}
        </span>
        {(change || subValue) && (
          <div className="text-right">
            {change && (
              <span
                className={`text-xs font-mono font-medium ${
                  trend === 'up'
                    ? 'text-[#00ff88]'
                    : trend === 'down'
                    ? 'text-[#ff007f]'
                    : 'text-white/60'
                }`}
              >
                {trend === 'up' && '↑ '}
                {trend === 'down' && '↓ '}
                {change}
              </span>
            )}
            {subValue && (
              <p className="text-[11px] text-white/40 font-sans mt-0.5">
                {subValue}
              </p>
            )}
          </div>
        )}
      </div>
    </div>
  );
};
