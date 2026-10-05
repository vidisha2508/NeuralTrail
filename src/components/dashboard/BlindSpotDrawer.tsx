import React from 'react';
import { useNavigate } from 'react-router-dom';
import { BlindSpot, Sample } from '../../types/neuralTrail';
import { StatusBadge } from '../common/StatusBadge';
import { Button } from '../common/Button';
import { X, ExternalLink, AlertTriangle, ArrowRight, Sparkles, Cpu, Layers } from 'lucide-react';
import { formatAccuracy } from '../../utils/formatters';

interface BlindSpotDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  blindSpot: BlindSpot | null;
  selectedSample: Sample | null;
  clusterSamples: Sample[];
  onSelectSample: (sample: Sample) => void;
}

export const BlindSpotDrawer: React.FC<BlindSpotDrawerProps> = ({
  isOpen,
  onClose,
  blindSpot,
  selectedSample,
  clusterSamples,
  onSelectSample,
}) => {
  const navigate = useNavigate();

  if (!isOpen || (!blindSpot && !selectedSample)) return null;

  return (
    <div className="fixed inset-y-0 right-0 w-96 max-w-full bg-[#0D1117] border-l border-border-subtle shadow-panel z-50 flex flex-col transform transition-transform duration-300 ease-in-out">
      {/* Header */}
      <div className="h-14 px-4 border-b border-border-subtle flex items-center justify-between bg-bg-panel">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-failure shadow-glow-failure animate-pulse" />
          <span className="font-heading font-semibold text-xs tracking-wider text-text-primary uppercase">
            {blindSpot ? `BLIND SPOT ${blindSpot.clusterNumber}` : 'SAMPLE INSPECTION'}
          </span>
          {blindSpot && <StatusBadge status={blindSpot.severity} size="sm" />}
        </div>
        <button
          onClick={onClose}
          className="p-1 rounded text-text-dim hover:text-text-primary hover:bg-bg-subtle transition-colors"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Drawer Body */}
      <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs font-mono">
        {/* Blind Spot High-level Summary */}
        {blindSpot && (
          <div className="space-y-4">
            <div>
              <h3 className="text-base font-heading font-bold text-text-primary mb-1">
                {blindSpot.name}
              </h3>
              <p className="text-text-secondary text-[11px] font-sans leading-relaxed">
                {blindSpot.description}
              </p>
            </div>

            {/* Readout Metrics Grid */}
            <div className="grid grid-cols-2 gap-2 bg-bg-card p-3 rounded border border-border-subtle">
              <div>
                <span className="text-[10px] text-text-dim uppercase">Cluster Samples</span>
                <div className="text-sm font-heading font-semibold text-text-primary mt-0.5">
                  {blindSpot.sampleCount.toLocaleString()}
                </div>
              </div>
              <div>
                <span className="text-[10px] text-text-dim uppercase">Cluster Accuracy</span>
                <div className="text-sm font-heading font-semibold text-failure glow-text-red mt-0.5">
                  {formatAccuracy(blindSpot.accuracy)}
                </div>
              </div>
              <div className="mt-1">
                <span className="text-[10px] text-text-dim uppercase">Mean Confidence</span>
                <div className="text-sm font-heading font-semibold text-uncertain mt-0.5">
                  {blindSpot.confidence}%
                </div>
              </div>
              <div className="mt-1">
                <span className="text-[10px] text-text-dim uppercase">Common Confusion</span>
                <div className="text-sm font-heading font-semibold text-accent-cyan mt-0.5 flex items-center gap-1">
                  <span>{blindSpot.commonConfusion}</span>
                </div>
              </div>
            </div>

            {/* Sensitivity Tag */}
            <div className="flex items-center justify-between p-2 rounded bg-failure-dark/30 border border-failure/30">
              <span className="text-text-secondary text-[11px]">Dominant Trigger:</span>
              <span className="text-failure font-bold">{blindSpot.sensitivity}</span>
            </div>

            {/* Scientific Mitigation Suggestion */}
            <div className="p-3 rounded bg-bg-subtle/70 border border-border-subtle/80 space-y-1">
              <div className="text-[10px] text-accent-cyan flex items-center gap-1 font-semibold uppercase">
                <Sparkles className="w-3 h-3" />
                <span>Intervention Protocol</span>
              </div>
              <p className="text-[11px] font-sans text-text-secondary">
                {blindSpot.mitigationSuggestion}
              </p>
            </div>
          </div>
        )}

        {/* Selected Sample Deep Visualizer: Digit Grid Panel + Probability Bars */}
        {selectedSample && (
          <div className="lab-panel p-3 rounded border border-accent-cyan/40 bg-bg-card/90 space-y-3 shadow-glow-cyan">
            <div className="flex items-center justify-between border-b border-border-subtle pb-2">
              <div className="flex items-center gap-1.5">
                <Cpu className="w-3.5 h-3.5 text-accent-cyan" />
                <span className="font-heading font-bold text-text-primary">
                  {selectedSample.id}
                </span>
              </div>
              <StatusBadge status={selectedSample.stability} size="sm" />
            </div>

            {/* 2-Column Neural Inspection: Digit Grid Left | Probability Bars Right */}
            <div className="grid grid-cols-2 gap-3 items-center">
              {/* Digit Grid Left */}
              <div className="bg-[#07090D] p-2 rounded border border-border-subtle flex flex-col items-center">
                <span className="text-[9px] text-text-dim uppercase tracking-wider mb-1">
                  Reconstructed Activation
                </span>
                <div className="grid grid-cols-8 gap-0.5 w-24 h-24 bg-black p-1 rounded border border-border-subtle/60">
                  {selectedSample.thumbnailGrid ? (
                    selectedSample.thumbnailGrid.flat().map((intensity, idx) => (
                      <div
                        key={idx}
                        className="w-full h-full rounded-[1px] transition-colors"
                        style={{
                          backgroundColor:
                            intensity > 0
                              ? selectedSample.stability === 'failure'
                                ? `rgba(255, 90, 95, ${intensity})`
                                : `rgba(101, 214, 255, ${intensity})`
                              : '#0a0d14',
                        }}
                      />
                    ))
                  ) : (
                    <div className="col-span-8 flex items-center justify-center text-[10px] text-text-dim">
                      Raw Tensor
                    </div>
                  )}
                </div>
                <div className="text-[10px] text-text-secondary mt-1 flex items-center gap-1">
                  <span>True:</span>
                  <span className="text-stable font-bold">{selectedSample.trueClass}</span>
                  <span className="mx-0.5">→</span>
                  <span>Pred:</span>
                  <span className="text-failure font-bold">{selectedSample.predictedClass}</span>
                </div>
              </div>

              {/* Probability Bars Right */}
              <div className="space-y-1.5">
                <span className="text-[9px] text-text-dim uppercase tracking-wider block">
                  Softmax Distribution
                </span>
                {/* Predicted Class Bar */}
                <div>
                  <div className="flex justify-between text-[10px] text-text-secondary mb-0.5">
                    <span>Class {selectedSample.predictedClass}</span>
                    <span className="text-failure font-bold">{selectedSample.confidence}%</span>
                  </div>
                  <div className="h-1.5 w-full bg-bg-subtle rounded-full overflow-hidden">
                    <div
                      className="h-full bg-failure shadow-glow-failure transition-all duration-300"
                      style={{ width: `${selectedSample.confidence}%` }}
                    />
                  </div>
                </div>

                {/* True Class Bar */}
                <div>
                  <div className="flex justify-between text-[10px] text-text-secondary mb-0.5">
                    <span>Class {selectedSample.trueClass}</span>
                    <span className="text-stable">
                      {Math.max(12, Math.round(100 - selectedSample.confidence - 10))}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-bg-subtle rounded-full overflow-hidden">
                    <div
                      className="h-full bg-stable transition-all duration-300"
                      style={{
                        width: `${Math.max(12, Math.round(100 - selectedSample.confidence - 10))}%`,
                      }}
                    />
                  </div>
                </div>

                {/* Perturbation reading */}
                <div className="pt-1 text-[10px] text-text-dim">
                  <span>Sensitivity: </span>
                  <span className="text-accent-cyan">{selectedSample.sensitivity}</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* 3-4 Sample Cards inside Cluster */}
        <div className="space-y-2">
          <div className="flex items-center justify-between text-[11px] text-text-dim">
            <span>REPRESENTATIVE EXAMPLES</span>
            <span>{clusterSamples.length} samples</span>
          </div>

          <div className="space-y-1.5">
            {clusterSamples.slice(0, 4).map((sample) => (
              <div
                key={sample.id}
                onClick={() => onSelectSample(sample)}
                className={`p-2.5 rounded border transition-all cursor-pointer flex items-center justify-between ${
                  selectedSample?.id === sample.id
                    ? 'bg-accent-cyan/15 border-accent-cyan/50 shadow-glow-cyan'
                    : 'bg-bg-card/70 border-border-subtle hover:border-border-active hover:bg-bg-subtle'
                }`}
              >
                <div className="flex items-center gap-2.5">
                  {/* Micro 4x4 indicator */}
                  <div className="w-6 h-6 rounded bg-[#07090D] border border-border-subtle flex items-center justify-center font-bold text-[10px]">
                    <span
                      className={
                        sample.stability === 'failure' ? 'text-failure' : 'text-stable'
                      }
                    >
                      {sample.trueClass}
                    </span>
                  </div>
                  <div>
                    <div className="text-xs text-text-primary font-mono font-medium">
                      {sample.id}
                    </div>
                    <div className="text-[10px] text-text-dim flex items-center gap-1">
                      <span>True: {sample.trueClass}</span>
                      <span>·</span>
                      <span>Pred: {sample.predictedClass}</span>
                    </div>
                  </div>
                </div>

                <div className="text-right">
                  <span
                    className={`text-xs font-bold ${
                      sample.confidence < 60 ? 'text-failure' : 'text-uncertain'
                    }`}
                  >
                    {sample.confidence}%
                  </span>
                  <div className="text-[9px] text-text-dim uppercase tracking-wider">
                    {sample.stability}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Primary Action Button */}
        <div className="pt-2">
          <Button
            variant="vaporwave"
            size="md"
            className="w-full justify-center tracking-wider uppercase"
            icon={<ExternalLink className="w-3.5 h-3.5" />}
            onClick={() => navigate('/blind-spots')}
          >
            OPEN BLIND SPOT EXPLORER
          </Button>
        </div>
      </div>
    </div>
  );
};
