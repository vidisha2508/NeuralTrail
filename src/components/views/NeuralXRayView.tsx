import React, { useState, useEffect } from 'react';
import { Activity, AlertTriangle, ShieldCheck, RefreshCw, SlidersHorizontal, Info, Layers } from 'lucide-react';
import { mlApiClient } from '../../services/mlApiClient';
import { ActivationLayer } from '../../types/neuralTrail';
import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';
import { Network3DVisualizer } from './Network3DVisualizer';

export const NeuralXRayView: React.FC = () => {
  const [selectedLayerId, setSelectedLayerId] = useState<string>('layer-middle');
  const [layers, setLayers] = useState<ActivationLayer[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);
  const [isBackendConnected, setIsBackendConnected] = useState<boolean>(false);
  const [showDifferenceMap, setShowDifferenceMap] = useState<boolean>(false);

  // Active compared samples
  const [sampleAId, setSampleAId] = useState<string>('');
  const [sampleBId, setSampleBId] = useState<string>('');
  const [sampleAMeta, setSampleAMeta] = useState<any>(null);
  const [sampleBMeta, setSampleBMeta] = useState<any>(null);

  // Available pools for selection
  const [successfulOptions, setSuccessfulOptions] = useState<any[]>([]);
  const [failedOptions, setFailedOptions] = useState<any[]>([]);

  const fetchPathway = async (aId?: string, bId?: string, layerId?: string) => {
    setIsLoading(true);
    try {
      const data = await mlApiClient.getActivationPathway({
        sample_a_id: aId || sampleAId || undefined,
        sample_b_id: bId || sampleBId || undefined,
        selected_layer_id: layerId || selectedLayerId,
      });

      if (data && Array.isArray(data.layers) && data.layers.length > 0) {
        setLayers(data.layers);
        setIsBackendConnected(true);
        if (data.sample_a) {
          setSampleAMeta(data.sample_a);
          setSampleAId(data.sample_a.id);
        }
        if (data.sample_b) {
          setSampleBMeta(data.sample_b);
          setSampleBId(data.sample_b.id);
        }
        if (Array.isArray(data.available_successful_samples)) {
          setSuccessfulOptions(data.available_successful_samples);
        }
        if (Array.isArray(data.available_failed_samples)) {
          setFailedOptions(data.available_failed_samples);
        }
      } else {
        setLayers([]);
        setIsBackendConnected(false);
      }
    } catch {
      setLayers([]);
      setIsBackendConnected(false);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchPathway();
  }, []);

  const handleSelectSampleA = (id: string) => {
    setSampleAId(id);
    fetchPathway(id, sampleBId, selectedLayerId);
  };

  const handleSelectSampleB = (id: string) => {
    setSampleBId(id);
    fetchPathway(sampleAId, id, selectedLayerId);
  };

  const currentLayer = layers.find((l) => l.id === selectedLayerId) || layers[0] || null;

  if (layers.length === 0 && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center font-mono">
        <div className="w-12 h-12 rounded-xl bg-white/[0.03] border border-white/10 flex items-center justify-center mb-3">
          <Layers className="w-6 h-6 text-[#d500f9]" />
        </div>
        <h3 className="font-display font-bold text-lg text-white">NO ACTIVATIONS RECORDED</h3>
        <p className="text-xs text-white/50 max-w-md mt-1">
          Load an active PyTorch model in the Model Registry to extract live intermediate tensor activations across layers.
        </p>
        <button
          onClick={() => fetchPathway()}
          className="mt-4 px-4 py-2 rounded text-xs uppercase font-bold bg-[#d500f9]/20 text-[#d500f9] border border-[#d500f9]/40 hover:bg-[#d500f9]/30 transition-all"
        >
          Probe Active Model Tensors
        </button>
      </div>
    );
  }

  return (
    <div className="flex flex-col h-full overflow-hidden">
      <div className="px-6 pt-6 pb-4 md:px-8 shrink-0">
        <PageHeader
          stepNumber="02"
          stepCode="TRACE"
          title="Interactive 3D Neural Visualization"
          description="Explore intermediate feature representations in real 3D space. Compare canonical vs degraded signal pathways across model stages."
          badge={
            isBackendConnected ? (
              <span className="font-mono text-xs text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
                LIVE PYTORCH TENSORS
              </span>
            ) : null
          }
          actions={
            <div className="flex items-center gap-3">
              <button
                onClick={() => setShowDifferenceMap((prev) => !prev)}
                className={`font-mono text-xs font-semibold px-3 py-2 rounded-md border flex items-center gap-2 transition-all ${
                  showDifferenceMap
                    ? 'bg-[#3d1b33] border-[#ff007f] text-[#ff007f]'
                    : 'bg-[#180933] border-white/15 text-white/80 hover:text-white'
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>{showDifferenceMap ? 'DIFFERENCE DELTA ACTIVE' : 'SHOW DELTA |A - B|'}</span>
              </button>

              <button
                onClick={() => fetchPathway()}
                disabled={isLoading}
                className="font-mono text-xs font-semibold text-[#00f0ff] bg-[#00f0ff]/10 px-3.5 py-2 rounded-md border border-[#00f0ff]/40 flex items-center gap-2 hover:bg-[#00f0ff]/20 transition-all disabled:opacity-50"
              >
                <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
                <span>PROBE {layers.length} TENSORS</span>
              </button>
            </div>
          }
        />
      </div>

      <div className="flex-1 relative w-full border-t border-white/10 bg-[#020005]">
        {/* 3D Canvas Container */}
        <div className="absolute inset-0">
          <Network3DVisualizer
            layers={layers}
            selectedLayerId={selectedLayerId}
            onSelectLayer={setSelectedLayerId}
            showDifferenceMap={showDifferenceMap}
          />
        </div>

        {/* HUD Overlay - Top Left: Sample Comparison */}
        {sampleAMeta && sampleBMeta && (
          <div className="absolute top-4 left-4 z-10 w-full max-w-[450px]">
            <Card noPadding className="bg-black/70 backdrop-blur-md border-white/10 shadow-2xl">
              <div className="flex flex-col divide-y divide-white/10">
                <div className="flex items-center gap-3 p-3 text-xs font-mono">
                  <ShieldCheck className="w-4 h-4 text-[#00ff88] shrink-0" />
                  <span className="text-white/50 font-medium shrink-0">CANONICAL (A):</span>
                  {successfulOptions.length > 0 ? (
                    <select
                      value={sampleAId}
                      onChange={(e) => handleSelectSampleA(e.target.value)}
                      className="bg-transparent text-[#00ff88] font-semibold flex-1 text-xs truncate focus:outline-none"
                    >
                      {successfulOptions.map((opt) => (
                        <option key={opt.id} value={opt.id} className="bg-black">
                          {opt.id} [{opt.trueClassName}]
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="text-[#00ff88] font-semibold truncate">
                      #{sampleAMeta.id} [{sampleAMeta.trueClassName}]
                    </span>
                  )}
                </div>

                <div className="flex items-center gap-3 p-3 text-xs font-mono">
                  <AlertTriangle className="w-4 h-4 text-[#ff007f] shrink-0" />
                  <span className="text-white/50 font-medium shrink-0">DEGRADED (B):</span>
                  {failedOptions.length > 0 ? (
                    <select
                      value={sampleBId}
                      onChange={(e) => handleSelectSampleB(e.target.value)}
                      className="bg-transparent text-[#ff007f] font-semibold flex-1 text-xs truncate focus:outline-none"
                    >
                      {failedOptions.map((opt) => (
                        <option key={opt.id} value={opt.id} className="bg-black">
                          {opt.id} [{opt.trueClassName} → {opt.predictedClassName}]
                        </option>
                      ))}
                    </select>
                  ) : (
                    <span className="text-[#ff007f] font-semibold truncate">
                      #{sampleBMeta.id} [{sampleBMeta.trueClassName} → {sampleBMeta.predictedClassName}]
                    </span>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}

        {/* HUD Overlay - Right Panel: Layer Inspector */}
        {currentLayer && (
          <div className="absolute top-4 right-4 z-10 w-[350px] bottom-4 flex flex-col gap-4">
            <Card 
              className="flex-1 bg-black/70 backdrop-blur-md border-white/10 shadow-2xl overflow-y-auto"
              headerTitle={`STAGE: ${currentLayer.name}`}
              headerSubtitle={currentLayer.stage}
              headerIcon={<Activity className="w-4 h-4 text-[#00f0ff]" />}
            >
              <div className="space-y-6">
                {/* Divergence Meter */}
                <div>
                  <div className="flex justify-between font-mono text-[10px] text-white/50 mb-2">
                    <span>REPRESENTATION DIVERGENCE</span>
                    <span className={currentLayer.anomalyScore > 0.4 ? 'text-[#ff007f]' : 'text-[#00ff88]'}>
                      {(currentLayer.anomalyScore * 100).toFixed(1)}%
                    </span>
                  </div>
                  <div className="h-1.5 w-full bg-white/10 rounded-full overflow-hidden">
                    <div
                      className={`h-full transition-all duration-300 ${
                        currentLayer.anomalyScore > 0.4 ? 'bg-[#ff007f]' : 'bg-[#00f0ff]'
                      }`}
                      style={{ width: `${Math.min(100, currentLayer.anomalyScore * 100)}%` }}
                    />
                  </div>
                </div>

                {/* Metrics Grid */}
                <div className="grid grid-cols-2 gap-3">
                  <div className="bg-white/5 border border-white/10 rounded p-3">
                    <div className="font-mono text-[9px] text-white/40 mb-1">TENSOR SHAPE</div>
                    <div className="font-mono text-xs text-white">{currentLayer.tensorShape}</div>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded p-3">
                    <div className="font-mono text-[9px] text-white/40 mb-1">DEAD NEURONS</div>
                    <div className="font-mono text-xs text-[#ffb300]">{(currentLayer.deadNeuronRatio * 100).toFixed(1)}%</div>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded p-3">
                    <div className="font-mono text-[9px] text-white/40 mb-1">CANONICAL NORM</div>
                    <div className="font-mono text-xs text-[#00ff88]">{currentLayer.activationNorm.toFixed(2)}</div>
                  </div>
                  <div className="bg-white/5 border border-white/10 rounded p-3">
                    <div className="font-mono text-[9px] text-white/40 mb-1">DEGRADED NORM</div>
                    <div className="font-mono text-xs text-[#ff007f]">
                      {(currentLayer.activationNormPerturbed ?? currentLayer.activationNorm).toFixed(2)}
                    </div>
                  </div>
                </div>

                {/* Analysis Text */}
                <div className="bg-[#0a0510] border border-white/10 rounded p-3 text-xs text-white/70 font-sans leading-relaxed">
                  <span className="text-[#00f0ff] font-mono font-semibold block mb-1">EMPIRICAL DIAGNOSIS:</span>
                  {currentLayer.anomalyScore > 0.4 ? (
                    <>Feature divergence of {(currentLayer.anomalyScore * 100).toFixed(1)}% detected in this stage. Latent activations drift significantly from the canonical manifold under stress.</>
                  ) : (
                    <>Feature density aligned with canonical attractor. Minimal drift ({(currentLayer.anomalyScore * 100).toFixed(1)}%) measured across spatial filters.</>
                  )}
                </div>
              </div>
            </Card>
          </div>
        )}
      </div>
    </div>
  );
};
