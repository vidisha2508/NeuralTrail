import React from 'react';
import { LucideIcon, Hammer, ArrowLeft, Terminal } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from './Button';

interface LabPlaceholderProps {
  moduleName: string;
  category: string;
  description: string;
  icon: LucideIcon;
  estimatedFeatures: string[];
}

export const LabPlaceholder: React.FC<LabPlaceholderProps> = ({
  moduleName,
  category,
  description,
  icon: Icon,
  estimatedFeatures,
}) => {
  return (
    <div className="p-6 md:p-10 max-w-4xl mx-auto space-y-6">
      {/* Breadcrumb back */}
      <div className="flex items-center gap-2 text-xs font-mono text-text-dim">
        <Link to="/" className="text-accent-cyan hover:underline flex items-center gap-1">
          <ArrowLeft className="w-3.5 h-3.5" /> Return to Dashboard
        </Link>
        <span>/</span>
        <span>{category}</span>
        <span>/</span>
        <span className="text-text-primary">{moduleName}</span>
      </div>

      {/* Main Polished Card */}
      <div className="lab-panel rounded-xl border border-border-subtle p-8 md:p-10 relative overflow-hidden">
        {/* Subtle background glow */}
        <div className="absolute top-0 right-0 w-80 h-80 bg-accent-cyan/5 rounded-full blur-3xl pointer-events-none" />
        <div className="absolute bottom-0 left-0 w-80 h-80 bg-vaporwave-purple/5 rounded-full blur-3xl pointer-events-none" />

        <div className="relative z-10 space-y-6">
          {/* Header */}
          <div className="flex items-start justify-between flex-wrap gap-4">
            <div className="flex items-center gap-4">
              <div className="w-14 h-14 rounded-lg bg-bg-card border border-accent-cyan/30 flex items-center justify-center shadow-glow-cyan">
                <Icon className="w-7 h-7 text-accent-cyan" />
              </div>
              <div>
                <span className="text-xs font-mono tracking-wider text-text-dim uppercase">
                  {category} MODULE
                </span>
                <h1 className="text-2xl font-heading font-bold text-text-primary tracking-tight">
                  {moduleName}
                </h1>
              </div>
            </div>

            {/* Status Pill */}
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-uncertain-dark/50 border border-uncertain/40 text-uncertain font-mono text-xs">
              <span className="w-2 h-2 rounded-full bg-uncertain animate-ping" />
              <span>MODULE IN DEVELOPMENT</span>
            </div>
          </div>

          {/* Description */}
          <p className="text-sm text-text-secondary leading-relaxed font-sans max-w-2xl">
            {description}
          </p>

          {/* Feature Specs */}
          <div className="space-y-3 pt-2">
            <h4 className="text-xs font-mono text-accent-cyan uppercase tracking-wider flex items-center gap-1.5">
              <Terminal className="w-3.5 h-3.5" />
              <span>Target Capabilities in Next Engineering Sprint</span>
            </h4>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-2.5">
              {estimatedFeatures.map((feat, idx) => (
                <div
                  key={idx}
                  className="p-3 rounded bg-bg-card/60 border border-border-subtle text-xs font-mono text-text-secondary flex items-start gap-2.5"
                >
                  <span className="text-accent-cyan mt-0.5">›</span>
                  <span>{feat}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Footer Action */}
          <div className="pt-4 border-t border-border-subtle flex items-center justify-between flex-wrap gap-3">
            <span className="text-xs font-mono text-text-dim">
              Telemetry hooked to Active Model runtime daemon
            </span>
            <Link to="/">
              <Button variant="outline" size="sm">
                Switch to Live Dashboard Map
              </Button>
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};
