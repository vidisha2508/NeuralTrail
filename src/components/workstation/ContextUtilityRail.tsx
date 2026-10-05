import React, { useState } from 'react';
import { 
  BlindSpot, 
  Sample, 
  Experiment, 
  GemmaInterpretationResponse 
} from '../../types/neuralTrail';
import { ContextualDiagnosticPanel } from './ContextualDiagnosticPanel';
import { GemmaResearcherConsole } from './GemmaResearcherConsole';
import { ChevronRight, ChevronLeft, Sliders, Brain, PanelRight } from 'lucide-react';

interface ContextUtilityRailProps {
  selectedBlindSpot: BlindSpot | null;
  selectedSample: Sample | null;
  selectedClusterId: string | null;
  onSelectCluster: (clusterId: string | null) => void;
  onSwitchToTab: (tab: any) => void;
  onTriggerExperiment: (name: string, exp?: Experiment, interp?: GemmaInterpretationResponse) => void;
  activeExperiment?: Experiment | null;
  interpretation?: GemmaInterpretationResponse | null;
}

export const ContextUtilityRail: React.FC<ContextUtilityRailProps> = ({
  selectedBlindSpot,
  selectedSample,
  selectedClusterId,
  onSelectCluster,
  onSwitchToTab,
  onTriggerExperiment,
  activeExperiment,
  interpretation,
}) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);
  const [mode, setMode] = useState<'CONTEXT' | 'RESEARCH'>('CONTEXT');

  return (
    <div className="flex shrink-0 h-full border-l border-white/10 z-20 select-none bg-[#0e051c]">
      {/* If Closed: Thin vertical tab rail */}
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="w-10 h-full bg-[#120624] hover:bg-[#180933] text-white/50 hover:text-white flex flex-col items-center py-5 transition-colors cursor-pointer group"
          title="Open Diagnostic Inspector"
        >
          <PanelRight className="w-4 h-4 text-white/40 group-hover:text-[#00f0ff] mb-4 transition-colors" />
          <span className="font-mono text-[11px] font-semibold tracking-widest uppercase text-white/40 group-hover:text-white [writing-mode:vertical-lr] rotate-180 transition-colors">
            Inspector & Context
          </span>
          {selectedBlindSpot && (
            <span className="mt-4 w-2 h-2 rounded-full bg-[#ff007f] animate-pulse" />
          )}
        </button>
      )}

      {/* If Open: Full right-side utility panel */}
      {isOpen && (
        <div className="w-[320px] h-full flex flex-col bg-[#120624]/95 backdrop-blur-md">
          {/* Top Rail Control Bar */}
          <div className="h-12 px-4 border-b border-white/10 flex items-center justify-between shrink-0">
            {/* View Mode Toggle */}
            <div className="flex items-center gap-1 bg-[#180933] p-1 rounded-md border border-white/10">
              <button
                onClick={() => setMode('CONTEXT')}
                className={`px-2.5 py-1 rounded text-xs font-sans font-medium transition-colors ${
                  mode === 'CONTEXT'
                    ? 'bg-[#29134d] text-white'
                    : 'text-white/45 hover:text-white'
                }`}
              >
                Context
              </button>
              <button
                onClick={() => setMode('RESEARCH')}
                className={`px-2.5 py-1 rounded text-xs font-sans font-medium transition-colors ${
                  mode === 'RESEARCH'
                    ? 'bg-[#29134d] text-white'
                    : 'text-white/45 hover:text-white'
                }`}
              >
                Assistant
              </button>
            </div>

            {/* Collapse Toggle */}
            <button
              onClick={() => setIsOpen(false)}
              className="p-1.5 rounded-md hover:bg-white/10 text-white/50 hover:text-white transition-colors"
              title="Collapse Inspector"
            >
              <ChevronRight className="w-4 h-4" />
            </button>
          </div>

          {/* Panel Content */}
          <div className="flex-1 overflow-hidden flex flex-col">
            {mode === 'CONTEXT' ? (
              <ContextualDiagnosticPanel
                selectedBlindSpot={selectedBlindSpot}
                selectedSample={selectedSample}
                onSelectCluster={onSelectCluster}
                onSwitchToTab={onSwitchToTab}
                onTriggerExperiment={onTriggerExperiment}
                activeExperiment={activeExperiment}
                interpretation={interpretation}
              />
            ) : (
              <GemmaResearcherConsole
                selectedClusterId={selectedClusterId}
                onTriggerExperiment={onTriggerExperiment}
                onNavigateToBeforeAfter={() => onSwitchToTab('IMPROVE')}
                collapsed={false}
                onToggleCollapse={() => setIsOpen(false)}
              />
            )}
          </div>
        </div>
      )}
    </div>
  );
};
