import React from 'react';

export type CardVariant = 'default' | 'highlight' | 'warning' | 'error' | 'success';

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  variant?: CardVariant;
  headerTitle?: string;
  headerSubtitle?: string;
  headerAction?: React.ReactNode;
  headerBadge?: React.ReactNode;
  headerIcon?: React.ReactNode;
  noPadding?: boolean;
}

export const Card: React.FC<CardProps> = ({
  variant = 'default',
  headerTitle,
  headerSubtitle,
  headerAction,
  headerBadge,
  headerIcon,
  noPadding = false,
  className = '',
  children,
  ...props
}) => {
  const variantStyles: Record<CardVariant, string> = {
    default: 'bg-[#150a29]/90 border-white/10 hover:border-white/15',
    highlight: 'bg-[#180d32]/95 border-[#00f0ff]/30 shadow-[0_0_20px_rgba(0,240,255,0.06)]',
    warning: 'bg-[#1f132b]/95 border-[#ffb300]/30 shadow-[0_0_20px_rgba(255,179,0,0.06)]',
    error: 'bg-[#220d2a]/95 border-[#ff007f]/35 shadow-[0_0_20px_rgba(255,0,127,0.08)]',
    success: 'bg-[#0f1f22]/95 border-[#00ff88]/30 shadow-[0_0_20px_rgba(0,255,136,0.06)]',
  };

  return (
    <div
      className={`rounded-lg border backdrop-blur-md transition-all duration-200 ${variantStyles[variant]} ${className}`}
      {...props}
    >
      {(headerTitle || headerAction || headerBadge || headerIcon) && (
        <div className="flex items-center justify-between px-5 py-3.5 border-b border-white/5">
          <div className="flex items-center gap-2.5 min-w-0">
            {headerIcon && <span className="shrink-0 text-white/70">{headerIcon}</span>}
            <div className="min-w-0">
              <div className="flex items-center gap-2">
                {headerTitle && (
                  <h3 className="font-sans font-semibold text-sm text-white tracking-wide truncate">
                    {headerTitle}
                  </h3>
                )}
                {headerBadge}
              </div>
              {headerSubtitle && (
                <p className="text-xs text-white/50 font-sans mt-0.5 truncate">
                  {headerSubtitle}
                </p>
              )}
            </div>
          </div>
          {headerAction && <div className="shrink-0 ml-3">{headerAction}</div>}
        </div>
      )}
      <div className={noPadding ? '' : 'p-5'}>{children}</div>
    </div>
  );
};
