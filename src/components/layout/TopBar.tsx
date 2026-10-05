import React, { useState } from 'react';
import { useLocation } from 'react-router-dom';
import { Play, CheckCircle2, Loader2, Sparkles, Layers, Sliders } from 'lucide-react';
import { AnalysisStage } from '../../types/neuralTrail';

interface TopBarProps {
  onRunAnalysis?: (stage: AnalysisStage) => void;
}

export const TopBar: React.FC<TopBarProps> = ({ onRunAnalysis }) => {
  const location = useLocation();
  const [analysisStage, setAnalysisStage] = useState<AnalysisStage>('IDLE');

  // Map path to breadcrumb
  const pathMap: Record<string, string> = {
    '/': 'Overview / Dashboard',
    '/blind-spots': 'Investigation / Blind Spot Explorer',
    '/neural-xray': 'Investigation / Neural X-Ray',
    '/robustness': 'Investigation / Robustness Lab',
    '/what-if': 'Investigation / What-If Lab',
    '/intervention': 'Investigation / Intervention Lab',
    '/comparison': 'Analysis / Model Comparison',
    '/dataset': 'Analysis / Dataset Analysis',
    '/research': 'Research / AI Assistant',
  };

  const breadcrumb = pathMap[location.pathname] || 'Dashboard';

  const triggerAnalysis = () => {
    if (analysisStage !== 'IDLE' && analysisStage !== 'COMPLETE') return;

    setAnalysisStage('ANALYZING');
    onRunAnalysis?.('ANALYZING');

    setTimeout(() => {
      setAnalysisStage('MAPPING');
      onRunAnalysis?.('MAPPING');
    }, 1100);

    setTimeout(() => {
      setAnalysisStage('SCANNING');
      onRunAnalysis?.('SCANNING');
    }, 2200);

    setTimeout(() => {
      setAnalysisStage('COMPLETE');
      onRunAnalysis?.('COMPLETE');
      setTimeout(() => {
        setAnalysisStage('IDLE');
        onRunAnalysis?.('IDLE');
      }, 3000);
    }, 3400);
  };

  return (
    <header className="h-14 border-b border-border-subtle bg-bg-panel px-6 flex items-center justify-between z-20">
      {/* Left: Breadcrumbs */}
      <div className="flex items-center gap-2 text-xs font-mono">
        <span className="text-text-dim uppercase tracking-wider">LAB ENVIRONMENT</span>
        <span className="text-border-active">/</span>
        <span className="text-accent-cyan font-medium">{breadcrumb}</span>
      </div>

      {/* Center / Right: Badges & Run Analysis */}
      <div className="flex items-center gap-3">
        {/* Model & Dataset Badge */}
        <div className="hidden md:flex items-center gap-2 bg-bg-card border border-border-subtle rounded px-2.5 py-1 text-xs font-mono">
          <div className="flex items-center gap-1.5 text-text-secondary">
            <Layers className="w-3.5 h-3.5 text-accent-cyan" />
            <span className="text-text-primary font-medium">Active Vision Core</span>
            <span className="text-border-active">|</span>
            <span className="text-text-dim text-[11px]">Demo Vision Set</span>
          </div>
          <span className="w-1.5 h-1.5 rounded-full bg-stable shadow-glow-stable" />
        </div>

        {/* Run Analysis Button with animated state machine */}
        <button
          onClick={triggerAnalysis}
          disabled={analysisStage !== 'IDLE' && analysisStage !== 'COMPLETE'}
          className={`relative px-4 py-1.5 rounded font-mono text-xs font-semibold flex items-center gap-2 transition-all duration-300 overflow-hidden ${
            analysisStage === 'IDLE'
              ? 'bg-gradient-to-r from-accent-cyan via-accent-cyan to-vaporwave-purple text-[#07090D] shadow-glow-cyan hover:brightness-110 active:scale-95'
              : analysisStage === 'COMPLETE'
              ? 'bg-stable-dark border border-stable text-stable shadow-glow-stable'
              : 'bg-bg-subtle text-accent-cyan border border-accent-cyan/50 shadow-glow-cyan'
          }`}
        >
          {/* Animated scanline bar during analysis */}
          {analysisStage !== 'IDLE' && analysisStage !== 'COMPLETE' && (
            <div className="absolute inset-0 bg-accent-cyan/15 animate-pulse" />
          )}

          {analysisStage === 'IDLE' && (
            <>
              <Play className="w-3.5 h-3.5 fill-current" />
              <span>RUN ANALYSIS</span>
            </>
          )}

          {analysisStage === 'ANALYZING' && (
            <>
              <Loader2 className="w-3.5 h-3.5 animate-spin" />
              <span className="tracking-wider">ANALYZING...</span>
            </>
          )}

          {analysisStage === 'MAPPING' && (
            <>
              <Sliders className="w-3.5 h-3.5 animate-spin" />
              <span className="tracking-wider">MAPPING TOPOLOGY...</span>
            </>
          )}

          {analysisStage === 'SCANNING' && (
            <>
              <Sparkles className="w-3.5 h-3.5 animate-pulse text-vaporwave-pink" />
              <span className="tracking-wider">SCANNING FAILURES...</span>
            </>
          )}

          {analysisStage === 'COMPLETE' && (
            <>
              <CheckCircle2 className="w-3.5 h-3.5 text-stable" />
              <span className="tracking-wider text-stable">COMPLETE</span>
            </>
          )}
        </button>
      </div>
    </header>
  );
};
