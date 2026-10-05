import React from 'react';

interface StatusBadgeProps {
  status: 'stable' | 'uncertain' | 'failure' | 'HIGH' | 'MEDIUM' | 'LOW';
  size?: 'sm' | 'md';
  pulse?: boolean;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, size = 'sm', pulse = false }) => {
  const configs = {
    stable: {
      bg: 'bg-stable-dark/60',
      border: 'border-stable/40',
      text: 'text-stable',
      dot: 'bg-stable shadow-glow-stable',
      label: 'Stable',
    },
    uncertain: {
      bg: 'bg-uncertain-dark/60',
      border: 'border-uncertain/40',
      text: 'text-uncertain',
      dot: 'bg-uncertain',
      label: 'Uncertain',
    },
    failure: {
      bg: 'bg-failure-dark/60',
      border: 'border-failure/40',
      text: 'text-failure',
      dot: 'bg-failure shadow-glow-failure',
      label: 'Failure',
    },
    HIGH: {
      bg: 'bg-failure-dark/60',
      border: 'border-failure/50',
      text: 'text-failure',
      dot: 'bg-failure shadow-glow-failure',
      label: 'HIGH',
    },
    MEDIUM: {
      bg: 'bg-uncertain-dark/60',
      border: 'border-uncertain/50',
      text: 'text-uncertain',
      dot: 'bg-uncertain',
      label: 'MEDIUM',
    },
    LOW: {
      bg: 'bg-accent-cyan/10',
      border: 'border-accent-cyan/30',
      text: 'text-accent-cyan',
      dot: 'bg-accent-cyan',
      label: 'LOW',
    },
  };

  const config = configs[status] || configs.stable;
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-2.5 py-1 text-xs';

  return (
    <span
      className={`inline-flex items-center gap-1.5 rounded font-mono font-medium border ${config.bg} ${config.border} ${config.text} ${sizeClasses}`}
    >
      <span
        className={`w-1.5 h-1.5 rounded-full ${config.dot} ${pulse ? 'animate-ping' : ''}`}
      />
      {config.label}
    </span>
  );
};
