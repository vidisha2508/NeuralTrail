import React from 'react';
import { 
  BlindSpot, 
  Sample, 
  Experiment, 
  GemmaInterpretationResponse 
} from '../../types/neuralTrail';
import { 
  Activity, 
  Layers, 
  Sliders, 
  Zap, 
  ChevronRight, 
  Brain,
  Sparkles
} from 'lucide-react';
import { formatAccuracy } from '../../utils/formatters';

interface ContextualDiagnosticPanelProps {
  selectedBlindSpot: BlindSpot | null;
  selectedSample: Sample | null;
  onSelectCluster: (clusterId: string | null) => void;
  onSwitchToTab: (tab: any) => void;
  onTriggerExperiment?: (name: string) => void;
  activeExperiment?: Experiment | null;
  interpretation?: GemmaInterpretationResponse | null;
}

export const ContextualDiagnosticPanel: React.FC<ContextualDiagnosticPanelProps> = ({
  selectedBlindSpot,
  selectedSample,
  onSelectCluster,
  onSwitchToTab,
  onTriggerExperiment,
  activeExperiment,
  interpretation,
}) => {
  // If nothing is selected, display calm minimal state
  if (!selectedBlindSpot && !selectedSample) {
    return (
      <div className="w-full h-full p-5 flex flex-col justify-between font-mono">
        <div>
          <div className="text-[11px] font-sans font-semibold text-white/40 uppercase tracking-wider mb-2">
            Diagnostic Context
          </div>
          <div className="p-4 rounded-lg border border-white/10 bg-white/[0.02] text-center my-6">
            <Activity className="w-6 h-6 text-white/20 mx-auto mb-2" />
            <p className="text-xs text-white/50 leading-relaxed font-sans">
              Select a failure region or node on the map to inspect model behavior.
            </p>
          </div>
        </div>

        <div className="text-xs text-white/30 font-sans leading-relaxed">
          Interactive cartography isolates clusters with systematic classification errors.
        </div>
      </div>
    );
  }

  const spot = selectedBlindSpot;
  const isHighSeverity = spot?.severity === 'HIGH';

  return (
    <div className="w-full h-full p-4 flex flex-col justify-between font-mono overflow-y-auto">
      <div className="space-y-4">
        {/* Header */}
        <div className="border-b border-white/10 pb-3">
          <div className="flex items-center justify-between text-[10px] font-bold text-white/40 uppercase tracking-wider mb-1">
            <span>Failure Region</span>
            <button
              onClick={() => onSelectCluster(null)}
              className="text-white/40 hover:text-white transition-colors"
              title="Clear selection"
            >
              ✕
            </button>
          </div>

          <h3 className="font-display font-bold text-base text-white tracking-wide">
            {spot ? spot.name : `Sample #${selectedSample?.id}`}
          </h3>

          {spot && (
            <div className="flex items-center gap-2 mt-1.5">
              <span
                className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                  isHighSeverity
                    ? 'bg-[#ff007f]/20 text-[#ff007f] border border-[#ff007f]/40'
                    : 'bg-[#ffb300]/20 text-[#ffb300] border border-[#ffb300]/40'
                }`}
              >
                {spot.severity} Severity
              </span>
              <span className="text-[11px] text-white/50">
                {spot.clusterNumber}
              </span>
            </div>
          )}
        </div>

        {/* Selected Sample Information if applicable */}
        {selectedSample && (
          <div className="p-3 rounded border border-white/10 bg-white/[0.02] space-y-1.5 text-xs">
            <div className="text-[10px] text-white/40 uppercase tracking-wider font-bold">
              Active Node
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">Predicted:</span>
              <span className="text-white font-semibold">
                Class {selectedSample.trueClass} → Class {selectedSample.predictedClass}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">Confidence:</span>
              <span className={selectedSample.confidence < 60 ? 'text-[#ff007f]' : 'text-[#00ff88]'}>
                {selectedSample.confidence}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-white/50">Sensitivity:</span>
              <span className="text-white/80">{selectedSample.sensitivity}</span>
            </div>
          </div>
        )}

        {/* Failure Region Key Metrics */}
        {spot && (
          <div className="space-y-2">
            <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider">
              Diagnostic Metrics
            </div>

            <div className="grid grid-cols-2 gap-2 text-xs">
              <div className="p-2.5 rounded bg-white/[0.02] border border-white/10">
                <span className="text-[10px] text-white/40 block">ACCURACY</span>
                <span className="text-lg font-bold text-[#ff007f]">
                  {formatAccuracy(spot.accuracy)}
                </span>
              </div>

              <div className="p-2.5 rounded bg-white/[0.02] border border-white/10">
                <span className="text-[10px] text-white/40 block">SAMPLES</span>
                <span className="text-lg font-bold text-white">
                  {spot.sampleCount}
                </span>
              </div>

              <div className="col-span-2 p-2.5 rounded bg-white/[0.02] border border-white/10 space-y-1">
                <div className="flex justify-between text-[11px]">
                  <span className="text-white/40">TRIGGER:</span>
                  <span className="text-white/90 font-medium">{spot.sensitivity}</span>
                </div>
                <div className="flex justify-between text-[11px]">
                  <span className="text-white/40">CONFUSION:</span>
                  <span className="text-[#00f0ff] font-medium">{spot.commonConfusion}</span>
                </div>
              </div>
            </div>

            {/* Description */}
            <p className="text-xs text-white/60 font-sans leading-relaxed pt-1">
              {spot.description}
            </p>
          </div>
        )}

        {/* Investigation Action Buttons */}
        <div className="space-y-2 pt-2 border-t border-white/10">
          <div className="text-[10px] font-bold text-white/40 uppercase tracking-wider mb-1">
            Investigate
          </div>

          <button
            onClick={() => onSwitchToTab('TRACE')}
            className="w-full py-2 px-3 rounded bg-[#180533] border border-white/15 hover:border-[#00f0ff] text-white hover:text-[#00f0ff] text-xs font-semibold flex items-center justify-between transition-colors"
          >
            <span className="flex items-center gap-2">
              <Layers className="w-3.5 h-3.5 text-[#00f0ff]" />
              <span>Trace Activation Path</span>
            </span>
            <ChevronRight className="w-3.5 h-3.5 opacity-50" />
          </button>

          <button
            onClick={() => onSwitchToTab('STRESS')}
            className="w-full py-2 px-3 rounded bg-[#180533] border border-white/15 hover:border-[#ffb300] text-white hover:text-[#ffb300] text-xs font-semibold flex items-center justify-between transition-colors"
          >
            <span className="flex items-center gap-2">
              <Sliders className="w-3.5 h-3.5 text-[#ffb300]" />
              <span>Run Stress Test</span>
            </span>
            <ChevronRight className="w-3.5 h-3.5 opacity-50" />
          </button>

          <button
            onClick={() => onSwitchToTab('RESEARCH')}
            className="w-full py-2 px-3 rounded bg-[#180533] border border-white/15 hover:border-[#d500f9] text-white hover:text-[#d500f9] text-xs font-semibold flex items-center justify-between transition-colors"
          >
            <span className="flex items-center gap-2">
              <Brain className="w-3.5 h-3.5 text-[#d500f9]" />
              <span>Hypothesis Analysis</span>
            </span>
            <ChevronRight className="w-3.5 h-3.5 opacity-50" />
          </button>
        </div>
      </div>

      {/* Mitigation Note */}
      {spot && (
        <div className="pt-3 border-t border-white/10 text-[11px] text-white/50 font-sans">
          <span className="text-[#00ff88] font-mono text-[10px] block font-bold mb-0.5">
            SUGGESTED MITIGATION
          </span>
          <p className="leading-snug">
            {spot.mitigationSuggestion}
          </p>
        </div>
      )}
    </div>
  );
};
