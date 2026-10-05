import React from 'react';

interface PageHeaderProps {
  stepNumber?: string;
  stepCode?: string;
  title: string;
  description: string;
  badge?: React.ReactNode;
  actions?: React.ReactNode;
  className?: string;
}

export const PageHeader: React.FC<PageHeaderProps> = ({
  stepNumber,
  stepCode,
  title,
  description,
  badge,
  actions,
  className = '',
}) => {
  return (
    <div className={`flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-6 mb-6 border-b border-white/10 ${className}`}>
      <div className="min-w-0">
        <div className="flex items-center gap-2.5 mb-1.5 flex-wrap">
          {stepNumber && (
            <span className="font-mono text-xs font-semibold text-[#00f0ff] px-2 py-0.5 rounded bg-[#00f0ff]/10 border border-[#00f0ff]/30">
              {stepNumber} {stepCode}
            </span>
          )}
          <h1 className="font-display font-bold text-2xl lg:text-3xl text-white tracking-tight">
            {title}
          </h1>
          {badge}
        </div>
        <p className="text-sm text-white/60 font-sans max-w-3xl leading-relaxed">
          {description}
        </p>
      </div>

      {actions && (
        <div className="flex items-center gap-3 shrink-0">
          {actions}
        </div>
      )}
    </div>
  );
};
