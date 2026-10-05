import React, { useState, useEffect, useRef } from 'react';
import {
  Sliders,
  RotateCw,
  Sun,
  Scissors,
  EyeOff,
  Sparkles,
  AlertTriangle,
  RefreshCw,
  Zap,
  ShieldCheck,
  ImageIcon,
  ArrowRight,
} from 'lucide-react';
import { mlApiClient } from '../../services/mlApiClient';
import { WorkstationTab } from '../../types/neuralTrail';
import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';
import { ProgressBar } from '../common/ProgressBar';

interface WhatIfLabViewProps {
  initialSampleId?: string;
  selectedClusterId?: string | null;
  onSwitchToTab?: (tab: WorkstationTab) => void;
  onRunRemediation?: () => Promise<void> | void;
}

export const WhatIfLabView: React.FC<WhatIfLabViewProps> = ({
  initialSampleId,
  selectedClusterId,
  onSwitchToTab,
  onRunRemediation,
}) => {
  const [isRemediating, setIsRemediating] = useState<boolean>(false);
  // 6 Perturbation slider states
  const [rotation, setRotation] = useState<number>(16);
  const [noise, setNoise] = useState<number>(10);
  const [blur, setBlur] = useState<number>(0);
  const [crop, setCrop] = useState<number>(0);
  const [brightness, setBrightness] = useState<number>(0);
  const [occlusion, setOcclusion] = useState<number>(0);

  // Sample Selection
  const [selectedSampleId, setSelectedSampleId] = useState<string>(initialSampleId || '');
  const [availableSamples, setAvailableSamples] = useState<any[]>([]);

  // Real backend experiment state
  const [experimentResult, setExperimentResult] = useState<any>(null);

  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const debounceTimerRef = useRef<any>(null);

  // Load available candidate samples on mount
  useEffect(() => {
    const fetchSamples = async () => {
      try {
        const samples = await mlApiClient.getWhatIfSamples();
        if (Array.isArray(samples) && samples.length > 0) {
          setAvailableSamples(samples);
          if (initialSampleId && samples.some((s) => s.id === initialSampleId)) {
            setSelectedSampleId(initialSampleId);
          } else {
            setSelectedSampleId(samples[0].id);
          }
        }
      } catch {
        // Standby
      }
    };
    fetchSamples();
  }, [initialSampleId]);

  // Run What-If experiment whenever parameters change (with 150ms debounce)
  useEffect(() => {
    if (!selectedSampleId) return;

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(async () => {
      setIsLoading(true);
      try {
        const res = await mlApiClient.runWhatIfExperiment({
          sample_id: selectedSampleId || undefined,
          rotation,
          noise,
          blur,
          crop,
          brightness,
          occlusion,
          include_images: true,
        });

        if (res && res.original_prediction) {
          setExperimentResult(res);
          setIsBackendConnected(true);
        }
      } catch {
        setIsBackendConnected(false);
      } finally {
        setIsLoading(false);
      }
    }, 150);

    return () => {
      if (debounceTimerRef.current) {
        clearTimeout(debounceTimerRef.current);
      }
    };
  }, [selectedSampleId, rotation, noise, blur, crop, brightness, occlusion]);

  const handleReset = () => {
    setRotation(0);
    setNoise(0);
    setBlur(0);
    setCrop(0);
    setBrightness(0);
    setOcclusion(0);
  };

  const isFlipped = experimentResult?.prediction_flip ?? false;
  const robustnessScore = experimentResult?.robustness_score ?? 100;

  if (availableSamples.length === 0) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center font-mono">
        <div className="w-14 h-14 rounded-xl bg-white/[0.03] border border-white/10 flex items-center justify-center mb-4">
          <Sliders className="w-7 h-7 text-[#ff007f]" />
        </div>
        <h3 className="font-display font-bold text-xl text-white">NO STRESS SAMPLES AVAILABLE</h3>
        <p className="text-xs text-white/50 max-w-md mt-2 font-sans leading-relaxed">
          Stress testing requires candidate samples from the loaded PyTorch model.
          Load and evaluate a model to run real-time manifold perturbation passes.
        </p>
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-[1440px] mx-auto w-full px-6 py-8 md:px-8 space-y-6">
        {/* Page Header */}
        <PageHeader
          stepNumber="03"
          stepCode="STRESS"
          title="What-If Testing Lab"
          description="Simulate real-time manifold stress perturbations. Adjust geometric, image quality, and occlusion parameters to detect prediction flip boundaries."
          badge={
            isBackendConnected ? (
              <span className="font-mono text-xs text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
                ACTIVE MODEL
              </span>
            ) : null
          }
          actions={
            <div className="flex items-center gap-3">
              {isLoading && (
                <span className="text-xs font-mono text-[#00f0ff] flex items-center gap-1.5 animate-pulse">
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  INFERENCING...
                </span>
              )}
              <button
                onClick={handleReset}
                className="font-mono text-xs font-semibold px-3 py-2 rounded-md bg-[#180933] border border-white/15 text-white/80 hover:text-white flex items-center gap-1.5 transition-colors"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                <span>Reset to Nominal</span>
              </button>
            </div>
          }
        />

        {/* 40 / 60 Layout Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Left Column: Grouped Perturbation Controls (5 cols) */}
          <div className="lg:col-span-5 space-y-4">
            <Card
              headerTitle="Perturbation Controls"
              headerSubtitle="Real-time parameter stress configuration"
              headerBadge={
                <span className="text-[11px] font-mono font-semibold text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
                  ONLINE
                </span>
              }
            >
              <div className="space-y-5">
                {/* Target Class Selector */}
                {availableSamples.length > 0 && (
                  <div className="space-y-1.5 pb-4 border-b border-white/10">
                    <label className="flex items-center justify-between text-xs font-sans font-medium text-white/70">
                      <span className="flex items-center gap-1.5 text-white">
                        <ImageIcon className="w-3.5 h-3.5 text-[#00f0ff]" />
                        Target Class Sample
                      </span>
                      <span className="font-mono text-[11px] text-white/40">
                        {availableSamples.length} candidates
                      </span>
                    </label>
                    <select
                      value={selectedSampleId}
                      onChange={(e) => setSelectedSampleId(e.target.value)}
                      className="w-full bg-[#0e051c] border border-white/15 text-white font-mono text-xs rounded-md px-3 py-2 focus:outline-none focus:border-[#00f0ff] transition-colors"
                    >
                      {availableSamples.map((s) => (
                        <option key={s.id} value={s.id}>
                          {s.id}: {s.trueClassName} (Class {s.trueClass})
                        </option>
                      ))}
                    </select>
                  </div>
                )}

                {/* Group 1: Geometric Controls */}
                <div className="space-y-3 pb-4 border-b border-white/10">
                  <div className="text-[11px] font-mono font-semibold text-[#00f0ff] uppercase tracking-wider">
                    Geometric Transformations
                  </div>

                  {/* Rotation */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <RotateCw className="w-3.5 h-3.5 text-[#ff007f]" />
                        Rotation Angle
                      </span>
                      <span className="font-mono text-xs text-white">
                        {rotation > 0 ? `+${rotation}°` : `${rotation}°`}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-45"
                      max="45"
                      step="1"
                      value={rotation}
                      onChange={(e) => setRotation(Number(e.target.value))}
                      className="w-full accent-[#ff007f] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>-45°</span>
                      <span>+45°</span>
                    </div>
                  </div>

                  {/* Scale / Crop */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <Scissors className="w-3.5 h-3.5 text-[#00f0ff]" />
                        Scale / Crop
                      </span>
                      <span className="font-mono text-xs text-white">
                        {crop}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="50"
                      step="5"
                      value={crop}
                      onChange={(e) => setCrop(Number(e.target.value))}
                      className="w-full accent-[#00f0ff] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>0%</span>
                      <span>50%</span>
                    </div>
                  </div>
                </div>

                {/* Group 2: Image Quality */}
                <div className="space-y-3 pb-4 border-b border-white/10">
                  <div className="text-[11px] font-mono font-semibold text-[#ffb300] uppercase tracking-wider">
                    Image Quality & Degradation
                  </div>

                  {/* Gaussian Noise */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <Sparkles className="w-3.5 h-3.5 text-[#ffb300]" />
                        Gaussian Noise
                      </span>
                      <span className="font-mono text-xs text-white">
                        {noise}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="100"
                      step="2"
                      value={noise}
                      onChange={(e) => setNoise(Number(e.target.value))}
                      className="w-full accent-[#ffb300] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>0%</span>
                      <span>100%</span>
                    </div>
                  </div>

                  {/* Blur */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <EyeOff className="w-3.5 h-3.5 text-[#b388ff]" />
                        Kernel Blur (σ)
                      </span>
                      <span className="font-mono text-xs text-white">
                        {blur}px
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="8"
                      step="0.5"
                      value={blur}
                      onChange={(e) => setBlur(Number(e.target.value))}
                      className="w-full accent-[#b388ff] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>0px</span>
                      <span>8px</span>
                    </div>
                  </div>

                  {/* Brightness */}
                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <Sun className="w-3.5 h-3.5 text-[#00ff88]" />
                        Illumination Delta
                      </span>
                      <span className="font-mono text-xs text-white">
                        {brightness > 0 ? `+${brightness}%` : `${brightness}%`}
                      </span>
                    </div>
                    <input
                      type="range"
                      min="-80"
                      max="80"
                      step="5"
                      value={brightness}
                      onChange={(e) => setBrightness(Number(e.target.value))}
                      className="w-full accent-[#00ff88] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>-80%</span>
                      <span>+80%</span>
                    </div>
                  </div>
                </div>

                {/* Group 3: Occlusion */}
                <div className="space-y-3">
                  <div className="text-[11px] font-mono font-semibold text-[#ff007f] uppercase tracking-wider">
                    Spatial Occlusion
                  </div>

                  <div className="space-y-1">
                    <div className="flex items-center justify-between text-xs font-sans">
                      <span className="text-white/80 flex items-center gap-1.5">
                        <AlertTriangle className="w-3.5 h-3.5 text-[#ff007f]" />
                        Occlusion Patch Mask
                      </span>
                      <span className="font-mono text-xs text-white">
                        {occlusion}%
                      </span>
                    </div>
                    <input
                      type="range"
                      min="0"
                      max="60"
                      step="5"
                      value={occlusion}
                      onChange={(e) => setOcclusion(Number(e.target.value))}
                      className="w-full accent-[#ff007f] cursor-pointer h-1.5 bg-white/10 rounded-lg"
                    />
                    <div className="flex justify-between text-[10px] text-white/40 font-mono">
                      <span>0%</span>
                      <span>60%</span>
                    </div>
                  </div>
                </div>
              </div>
            </Card>
          </div>

          {/* Right Column: Prediction Comparison Hero (7 cols) */}
          <div className="lg:col-span-7 space-y-4">
            {!experimentResult ? (
              <div className="p-12 rounded-lg bg-white/[0.02] border border-white/10 flex flex-col items-center justify-center text-center">
                <RefreshCw className="w-8 h-8 text-[#00f0ff] animate-spin mb-3" />
                <div className="font-mono text-sm text-white font-semibold">EXECUTING LIVE PERTURBATION PASS</div>
                <div className="text-xs text-white/50 mt-1 max-w-sm font-sans">
                  Computing real PyTorch forward pass with live geometric and pixel perturbations...
                </div>
              </div>
            ) : (
              <>
                {/* Status Alert Banner */}
                {isFlipped ? (
              <div className="p-4 rounded-lg bg-[#290c23] border border-[#ff007f]/50 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <AlertTriangle className="w-6 h-6 text-[#ff007f] shrink-0" />
                  <div>
                    <div className="font-sans font-semibold text-sm text-[#ff007f]">
                      Prediction Flip Triggered
                    </div>
                    <div className="text-xs text-white/70 font-sans mt-0.5">
                      Model collapsed from <strong className="text-white">{experimentResult.original_prediction}</strong> into{' '}
                      <strong className="text-[#ff007f]">{experimentResult.perturbed_prediction}</strong>
                    </div>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-[#ff007f]/20 text-[#ff007f] border border-[#ff007f]/40">
                  Δ {experimentResult.confidence_change_percentage > 0 ? '+' : ''}
                  {experimentResult.confidence_change_percentage}%
                </span>
              </div>
            ) : (
              <div className="p-4 rounded-lg bg-[#0e2118] border border-[#00ff88]/40 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <ShieldCheck className="w-6 h-6 text-[#00ff88] shrink-0" />
                  <div>
                    <div className="font-sans font-semibold text-sm text-[#00ff88]">
                      Canonical Invariance Preserved
                    </div>
                    <div className="text-xs text-white/70 font-sans mt-0.5">
                      Model classification remains invariant within current stress perturbation boundaries.
                    </div>
                  </div>
                </div>
                <span className="px-2.5 py-1 rounded text-xs font-mono font-semibold bg-[#00ff88]/20 text-[#00ff88] border border-[#00ff88]/40">
                  NOMINAL
                </span>
              </div>
            )}

            {/* Central Side-by-Side Comparison */}
            <Card
              headerTitle="Prediction Comparison"
              headerSubtitle="Baseline inference vs perturbed input response"
            >
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 items-center">
                {/* Before: Canonical */}
                <div className="flex flex-col items-center p-5 rounded-lg bg-[#0e051c] border border-[#00ff88]/30 space-y-3">
                  <span className="font-mono text-xs font-semibold text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
                    BEFORE (CANONICAL)
                  </span>

                  <div className="w-36 h-36 bg-[#080210] rounded-lg border border-white/10 flex items-center justify-center p-2 relative overflow-hidden">
                    {experimentResult.original_image_url ? (
                      <img
                        src={experimentResult.original_image_url}
                        alt="Canonical Input"
                        className="w-full h-full object-contain rounded"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-center p-2">
                        <div className="font-display font-bold text-2xl text-[#00ff88]">
                          {experimentResult.original_prediction}
                        </div>
                        <span className="text-[10px] text-white/40 font-mono mt-1">Class #{experimentResult.original_prediction_idx}</span>
                      </div>
                    )}
                  </div>

                  <div className="text-center space-y-0.5">
                    <div className="text-[11px] text-white/40 font-sans">ORIGINAL INFERENCE</div>
                    <div className="font-sans font-semibold text-sm text-white truncate max-w-[200px]">
                      {experimentResult.original_prediction}
                    </div>
                    <div className="font-mono text-sm text-[#00ff88] font-semibold">
                      {experimentResult.original_confidence_percentage}% CONF
                    </div>
                  </div>
                </div>

                {/* After: Perturbed */}
                <div
                  className={`flex flex-col items-center p-5 rounded-lg border space-y-3 transition-colors ${
                    isFlipped
                      ? 'bg-[#220a1f] border-[#ff007f]/50'
                      : 'bg-[#0e051c] border-white/10'
                  }`}
                >
                  <span
                    className={`font-mono text-xs font-semibold px-2 py-0.5 rounded border ${
                      isFlipped
                        ? 'text-[#ff007f] bg-[#ff007f]/10 border-[#ff007f]/30'
                        : 'text-[#00f0ff] bg-[#00f0ff]/10 border-[#00f0ff]/30'
                    }`}
                  >
                    AFTER (PERTURBED)
                  </span>

                  <div className="w-36 h-36 bg-[#080210] rounded-lg border border-white/10 flex items-center justify-center p-2 relative overflow-hidden">
                    {experimentResult.perturbed_image_url ? (
                      <img
                        src={experimentResult.perturbed_image_url}
                        alt="Perturbed Input"
                        className="w-full h-full object-contain rounded"
                        style={{ imageRendering: 'pixelated' }}
                      />
                    ) : (
                      <div className="flex flex-col items-center justify-center text-center p-2">
                        <div
                          className="font-display font-bold text-2xl transition-transform duration-150"
                          style={{
                            transform: `rotate(${rotation}deg)`,
                            color: isFlipped ? '#ff007f' : '#00f0ff',
                          }}
                        >
                          {experimentResult.perturbed_prediction}
                        </div>
                        <span className="text-[10px] text-white/40 font-mono mt-1">Class #{experimentResult.perturbed_prediction_idx}</span>
                      </div>
                    )}
                  </div>

                  <div className="text-center space-y-0.5">
                    <div className="text-[11px] text-white/40 font-sans">PERTURBED INFERENCE</div>
                    <div
                      className={`font-sans font-semibold text-sm truncate max-w-[200px] ${
                        isFlipped ? 'text-[#ff007f]' : 'text-white'
                      }`}
                    >
                      {experimentResult.perturbed_prediction}
                    </div>
                    <div
                      className={`font-mono text-sm font-semibold ${
                        experimentResult.perturbed_confidence_percentage < 60
                          ? 'text-[#ff007f]'
                          : 'text-[#ffb300]'
                      }`}
                    >
                      {experimentResult.perturbed_confidence_percentage}% CONF
                    </div>
                  </div>
                </div>
              </div>

              {/* Robustness Score */}
              <div className="pt-5 mt-5 border-t border-white/10 space-y-2">
                <div className="flex items-center justify-between text-xs">
                  <span className="flex items-center gap-1.5 text-white/70 font-sans font-medium">
                    <Zap className="w-4 h-4 text-[#00f0ff]" />
                    Robustness Resilience Index
                  </span>
                  <span
                    className={`font-mono text-xs font-semibold px-2 py-0.5 rounded border ${
                      robustnessScore < 50
                        ? 'bg-[#ff007f]/10 text-[#ff007f] border-[#ff007f]/30'
                        : 'bg-[#00ff88]/10 text-[#00ff88] border-[#00ff88]/30'
                    }`}
                  >
                    {robustnessScore} / 100
                  </span>
                </div>
                <ProgressBar
                  value={robustnessScore}
                  accent={robustnessScore < 50 ? 'magenta' : 'green'}
                  size="md"
                />
              </div>

              {/* Differential Predictions Breakdown */}
              {experimentResult.top_perturbed_predictions && (
                <div className="pt-4 mt-4 border-t border-white/10">
                  <div className="text-[11px] font-sans font-medium text-white/50 mb-2">
                    Top-3 Class Logits Under Perturbation
                  </div>
                  <div className="grid grid-cols-3 gap-3">
                    {experimentResult.top_perturbed_predictions.slice(0, 3).map((p: any, idx: number) => (
                      <div
                        key={idx}
                        className="p-2 rounded bg-[#0e051c] border border-white/10 text-center font-mono text-xs"
                      >
                        <div className="text-white/60 truncate font-sans">{p.label}</div>
                        <div className={`font-semibold mt-0.5 ${idx === 0 ? 'text-[#00ff88]' : 'text-white'}`}>
                          {p.confidence_percentage}%
                        </div>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {/* Connected Stage Progression: STRESS → GEMMA / IMPROVE */}
              <div className="pt-4 mt-4 border-t border-white/10 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-mono font-semibold text-white/70 uppercase">
                    Stage Progression (Step 03 → 04 / 06)
                  </span>
                  <span className="text-[11px] font-sans text-white/40">
                    {isFlipped ? 'Failure Boundary Triggered' : 'Stress Response Measured'}
                  </span>
                </div>

                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <button
                    onClick={async () => {
                      if (onRunRemediation) {
                        setIsRemediating(true);
                        try {
                          await onRunRemediation();
                        } finally {
                          setIsRemediating(false);
                        }
                      } else if (onSwitchToTab) {
                        onSwitchToTab('IMPROVE');
                      }
                    }}
                    disabled={isRemediating}
                    className="w-full font-mono text-xs font-semibold px-4 py-2.5 rounded-md bg-[#00ff88]/15 border border-[#00ff88]/40 text-[#00ff88] hover:bg-[#00ff88]/25 flex items-center justify-center gap-2 transition-all shadow-neon-cyan disabled:opacity-50"
                  >
                    {isRemediating ? (
                      <>
                        <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                        <span>EXECUTING MITIGATION...</span>
                      </>
                    ) : (
                      <>
                        <ShieldCheck className="w-3.5 h-3.5 text-[#00ff88]" />
                        <span>Run Mitigation → View Improvement</span>
                        <ArrowRight className="w-3.5 h-3.5" />
                      </>
                    )}
                  </button>

                  {onSwitchToTab && (
                    <button
                      onClick={() => onSwitchToTab('RESEARCH')}
                      className="w-full font-mono text-xs font-semibold px-4 py-2.5 rounded-md bg-[#d500f9]/15 border border-[#d500f9]/40 text-[#d500f9] hover:bg-[#d500f9]/25 flex items-center justify-center gap-2 transition-all"
                    >
                      <Sparkles className="w-3.5 h-3.5 text-[#d500f9]" />
                      <span>Formulate Hypothesis (Gemma 4)</span>
                      <ArrowRight className="w-3.5 h-3.5" />
                    </button>
                  )}
                </div>
              </div>
            </Card>
          </>
        )}
      </div>
        </div>
      </div>
    </div>
  );
};
