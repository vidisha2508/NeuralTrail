import React, { useState, useEffect, useCallback } from 'react';
import { 
  WorkstationTab, 
  FilterState, 
  Sample, 
  BlindSpot, 
  ExperimentTelemetry, 
  Experiment, 
  GemmaInterpretationResponse 
} from '../../types/neuralTrail';
import { analysisService } from '../../services/analysisService';
import { mlApiClient } from '../../services/mlApiClient';
import { WorkstationTopBar } from './WorkstationTopBar';
import { WorkstationNavDeck } from './WorkstationNavDeck';
import { ContextUtilityRail } from './ContextUtilityRail';
import { TelemetryStatusBar } from './TelemetryStatusBar';
import { LandingHeroModal } from './LandingHeroModal';

// Views
import { BlindSpotMapView } from '../views/BlindSpotMapView';
import { NeuralXRayView } from '../views/NeuralXRayView';
import { WhatIfLabView } from '../views/WhatIfLabView';
import { BeforeAfterView } from '../views/BeforeAfterView';
import { FailureReplayView } from '../views/FailureReplayView';
import { ExperimentEngineView } from '../views/ExperimentEngineView';
import { ResearchAssistant } from '../../pages/ResearchAssistant';
import { Play, Cpu, Sparkles, FolderCode, Upload, AlertCircle, Compass } from 'lucide-react';

export const WorkstationMaster: React.FC = () => {
  // Navigation & Active View
  const [activeTab, setActiveTab] = useState<WorkstationTab>('FIND');
  const [landingModalOpen, setLandingModalOpen] = useState<boolean>(false);
  const [crtActive, setCrtActive] = useState<boolean>(false);
  const [isScanning, setIsScanning] = useState<boolean>(false);
  const [isExporting, setIsExporting] = useState<boolean>(false);
  const [isLoadingModel, setIsLoadingModel] = useState<boolean>(false);

  // Model & Metrics Metadata
  const [modelMetadata, setModelMetadata] = useState<any>(null);
  const [metrics, setMetrics] = useState<any>(null);

  // Data States
  const [samples, setSamples] = useState<Sample[]>([]);
  const [blindSpots, setBlindSpots] = useState<BlindSpot[]>([]);
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);

  // Filter State for Cartography Map
  const [filterState, setFilterState] = useState<FilterState>({
    status: 'all',
    minConfidence: 0,
    selectedClass: 'all',
    perturbation: 'all',
    searchQuery: '',
    selectedClusterId: null,
  });

  // Active Experiment & Registry
  const [activeExperiment, setActiveExperiment] = useState<Experiment | null>(null);
  const [experimentsList, setExperimentsList] = useState<Experiment[]>([]);
  const [activeInterpretation, setActiveInterpretation] = useState<GemmaInterpretationResponse | null>(null);

  // Telemetry Engine State (Real Hardware Telemetry)
  const [telemetry, setTelemetry] = useState<ExperimentTelemetry>({
    id: 'STANDBY',
    name: 'Local PyTorch Runtime',
    status: 'STANDBY',
    progress: 0,
    samplesProcessed: 0,
    totalSamples: 0,
    gpuTemp: 'Host RAM',
    gpuLoad: 'CPU',
    currentLoss: 0,
    accuracyDelta: '--',
    logs: [
      `[${new Date().toLocaleTimeString()}] RUNTIME: Neural Trail PyTorch engine standby. Awaiting model selection.`,
    ],
  });

  const syncExperimentToTelemetry = useCallback((exp: Experiment) => {
    setActiveExperiment(exp);
    setTelemetry({
      id: exp.id,
      name: exp.parameters?.protocol || exp.parameters?.experiment_protocol || exp.type || 'Mitigation Protocol',
      status: exp.status,
      progress: exp.progress,
      samplesProcessed: exp.results?.samples_processed || exp.results?.total_samples_evaluated || 0,
      totalSamples: exp.results?.total_samples_evaluated || 0,
      gpuTemp: 'Host RAM',
      gpuLoad: exp.results?.device || 'CPU',
      currentLoss: exp.results?.current_loss || 0,
      accuracyDelta: exp.results?.accuracy_delta || '--',
      logs: exp.logs && exp.logs.length > 0 ? exp.logs : [
        `[${new Date().toLocaleTimeString()}] TRIAL: Experiment ${exp.id} active on ${exp.provider.toUpperCase()} compute.`,
      ],
    });
  }, []);

  // Check health and load existing model data if backend already has one loaded
  const refreshInvestigationData = useCallback(async () => {
    try {
      const health = await mlApiClient.checkHealth();
      if (health && health.model_loaded) {
        const [meta, s, b, m, exps] = await Promise.all([
          mlApiClient.getModelInfo().catch(() => null),
          analysisService.getSamples(),
          analysisService.getBlindSpots(),
          analysisService.getModelMetrics(),
          mlApiClient.listExperiments().catch(() => []),
        ]);

        setModelMetadata(meta);
        setSamples(s);
        setBlindSpots(b);
        setMetrics(m);
        if (b.length > 0) setSelectedClusterId(b[0].id);
        if (s.length > 0) setSelectedSample(s[0]);

        if (exps && exps.length > 0) {
          setExperimentsList(exps);
          syncExperimentToTelemetry(exps[0]);
        }
      } else {
        setModelMetadata(null);
        setSamples([]);
        setBlindSpots([]);
        setMetrics(null);
        setSelectedSample(null);
        setSelectedClusterId(null);
      }
    } catch (err) {
      console.warn('Backend connection standby:', err);
    }
  }, [syncExperimentToTelemetry]);

  useEffect(() => {
    refreshInvestigationData();
  }, [refreshInvestigationData]);

  // Handle Model Selection
  const handleSelectModel = async (modelType: string, customData?: any) => {
    setIsLoadingModel(true);
    try {
      if (modelType === 'resnet18') {
        await mlApiClient.loadModel(undefined, 'resnet18');
      } else if (modelType === 'mobilenet') {
        await mlApiClient.loadModel(undefined, 'mobilenet');
      } else if (modelType === 'custom_test') {
        await mlApiClient.loadCustomTestModel();
      } else if (modelType === 'upload' && customData instanceof FormData) {
        await mlApiClient.uploadCustomModel(customData);
      }

      // Reset experiment state for new model
      setActiveExperiment(null);
      setActiveInterpretation(null);

      // Refresh all investigation views with real data from newly loaded model
      await refreshInvestigationData();
      setActiveTab('FIND');

      setTelemetry((prev) => ({
        ...prev,
        status: 'IDLE',
        progress: 100,
        logs: [
          ...prev.logs,
          `[${new Date().toLocaleTimeString()}] MODEL: Successfully loaded and evaluated '${modelType}'. Investigation initialized.`,
        ],
      }));
    } catch (err: any) {
      console.error('Failed to load model:', err);
      alert(err.message || 'Failed to load model.');
    } finally {
      setIsLoadingModel(false);
    }
  };

  // Handle Model Unload
  const handleUnloadModel = async () => {
    try {
      await mlApiClient.unloadModel();
      setModelMetadata(null);
      setSamples([]);
      setBlindSpots([]);
      setMetrics(null);
      setSelectedSample(null);
      setSelectedClusterId(null);
      setActiveExperiment(null);
      setActiveInterpretation(null);
      setActiveTab('FIND');

      setTelemetry({
        id: 'STANDBY',
        name: 'Local PyTorch Runtime',
        status: 'STANDBY',
        progress: 0,
        samplesProcessed: 0,
        totalSamples: 0,
        gpuTemp: 'Host RAM',
        gpuLoad: 'CPU',
        currentLoss: 0,
        accuracyDelta: '--',
        logs: [
          `[${new Date().toLocaleTimeString()}] RESET: Model unloaded. Neural Trail returned to standby state.`,
        ],
      });
    } catch (err) {
      console.warn('Unload error:', err);
    }
  };

  // Select Sample Handler (syncs across FIND, TRACE, and STRESS)
  const handleSelectSample = (sample: Sample | null) => {
    setSelectedSample(sample);
    if (sample) {
      mlApiClient.selectInvestigationSample(sample.id);
    }
  };

  // Trigger diagnostic scan simulation / re-evaluation
  const handleRunScan = async () => {
    setIsScanning(true);
    try {
      await mlApiClient.startAnalysis({ sample_count: 100, async_run: false });
      await refreshInvestigationData();
      setTelemetry((prev) => ({
        ...prev,
        progress: 100,
        status: 'COMPLETED',
        logs: [
          ...prev.logs,
          `[${new Date().toLocaleTimeString()}] SCAN: Full evaluation pass complete. Fresh PCA manifold generated.`,
        ],
      }));
    } catch (err) {
      console.error('Evaluation scan failed:', err);
    } finally {
      setIsScanning(false);
    }
  };

  const handleExportModel = async () => {
    setIsExporting(true);
    try {
      const blob = await mlApiClient.exportModel();
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${modelMetadata?.model_name || 'model'}_export.pth`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      a.remove();
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setIsExporting(false);
    }
  };

  // Trigger Experiment from AI Console
  const handleTriggerExperiment = (
    experimentName: string, 
    exp?: Experiment, 
    interp?: GemmaInterpretationResponse
  ) => {
    if (exp) {
      syncExperimentToTelemetry(exp);
      setExperimentsList((prev) => {
        const exists = prev.some((e) => e.id === exp.id);
        if (exists) {
          return prev.map((e) => (e.id === exp.id ? exp : e));
        }
        return [exp, ...prev];
      });
    }
    if (interp) {
      setActiveInterpretation(interp);
    }
  };

  // Run new trial from Experiment Engine View
  const handleRunNewTrial = async (params?: { type: string; parameters: Record<string, any>; provider: string }) => {
    try {
      const exp = await mlApiClient.submitExperiment({
        type: params?.type || 'invariant_mitigation',
        provider: params?.provider || 'local',
        parameters: params?.parameters || {
          cluster_id: selectedClusterId || (blindSpots[0]?.id ?? 'bs-01'),
          intensity: 20.0,
          apply_mitigation: true,
        },
      });

      syncExperimentToTelemetry(exp);
      setExperimentsList((prev) => [exp, ...prev]);

      // Poll until finished
      let pollJob = exp;
      while (pollJob.status === 'QUEUED' || pollJob.status === 'RUNNING') {
        await new Promise((r) => setTimeout(r, 400));
        try {
          pollJob = await mlApiClient.getExperiment(exp.id);
          syncExperimentToTelemetry(pollJob);
          setExperimentsList((prev) => prev.map((e) => (e.id === pollJob.id ? pollJob : e)));
        } catch {
          break;
        }
      }

      if (pollJob.status === 'COMPLETED') {
        try {
          const interp = await mlApiClient.interpretExperiment(pollJob.id);
          setActiveInterpretation(interp);
        } catch (e) {
          console.warn('Auto-interpret error:', e);
        }
      }
    } catch (err) {
      console.warn('New trial submission error:', err);
    }
  };

  const handleCancelExperiment = async (expId: string) => {
    try {
      await mlApiClient.cancelExperiment(expId);
      const updated = await mlApiClient.getExperiment(expId);
      syncExperimentToTelemetry(updated);
      setExperimentsList((prev) => prev.map((e) => (e.id === expId ? updated : e)));
    } catch (err) {
      console.warn('Cancel experiment error:', err);
    }
  };

  const handleInterpretExperiment = async (expId: string) => {
    try {
      const interp = await mlApiClient.interpretExperiment(expId);
      setActiveInterpretation(interp);
      setActiveTab('IMPROVE');
    } catch (err) {
      console.warn('Interpret experiment error:', err);
    }
  };

  const isModelLoaded = Boolean(modelMetadata && samples.length > 0);
  const selectedSpot = blindSpots.find((b) => b.id === selectedClusterId) || null;

  return (
    <div className="flex flex-col h-screen w-screen overflow-hidden bg-[#0c0418] text-white relative font-sans">
      {/* Perspective Horizon Grid */}
      <div className="vapor-grid-horizon" />

      {/* CRT Scanline Atmospheric Overlays */}
      {crtActive && (
        <>
          <div className="crt-overlay" />
          <div className="crt-scanbar" />
        </>
      )}

      {/* Top Toolbar */}
      <WorkstationTopBar
        activeModelName={modelMetadata?.model_name || 'No Model Loaded'}
        modelStatus={isModelLoaded ? 'Active' : 'Standby'}
        isLoaded={isModelLoaded}
        crtActive={crtActive}
        onToggleCrt={() => setCrtActive(!crtActive)}
        onOpenModelModal={() => setLandingModalOpen(true)}
        onSelectModel={handleSelectModel}
        onUnloadModel={handleUnloadModel}
        onRunScan={handleRunScan}
        onExportModel={handleExportModel}
        isScanning={isScanning}
        isExporting={isExporting}
      />

      {/* Main Workspace Area */}
      <div className="flex-1 flex min-h-0 overflow-hidden relative z-10">
        {/* Navigation Deck */}
        <WorkstationNavDeck
          activeTab={activeTab}
          onSelectTab={setActiveTab}
          clusterCount={blindSpots.length}
          failureCount={samples.filter((s) => s.stability === 'failure').length}
        />

        {/* Center Diagnostic Area or Empty State */}
        <main className="flex-1 flex flex-col min-w-0 overflow-hidden bg-transparent">
          {!isModelLoaded ? (
            /* HONEST EMPTY STATE WHEN NO MODEL IS LOADED */
            <div className="flex-1 flex items-center justify-center p-6 md:p-12 overflow-y-auto font-mono">
              <div className="max-w-2xl w-full border border-white/15 bg-black/60 backdrop-blur-md rounded-xl p-8 shadow-2xl text-center space-y-6">
                <div className="w-14 h-14 mx-auto rounded-xl bg-[#16092e] border border-[#7c4dff]/40 flex items-center justify-center">
                  <Cpu className="w-7 h-7 text-[#00ff88]" />
                </div>

                <div className="space-y-2">
                  <h2 className="font-display font-black text-2xl md:text-3xl text-white tracking-wide">
                    NO MODEL LOADED (STANDBY)
                  </h2>
                  <p className="text-xs text-[#00ff88] font-semibold uppercase tracking-wider">
                    Select a PyTorch vision model to initiate investigation
                  </p>
                  <p className="text-xs text-white/60 font-sans max-w-md mx-auto leading-relaxed pt-1">
                    Neural Trail operates exclusively on genuine PyTorch models and real evaluation data.
                    Select an architecture below to load its weights and run the diagnostics pipeline.
                  </p>
                </div>

                {/* 3 Model Selection Cards */}
                <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2 text-left">
                  {/* Card 1: ResNet18 */}
                  <div
                    onClick={() => handleSelectModel('resnet18')}
                    className="p-4 rounded-lg bg-white/[0.03] border border-white/10 hover:border-[#00ff88]/60 cursor-pointer transition-all hover:bg-white/[0.06] flex flex-col justify-between group"
                  >
                    <div>
                      <div className="text-[10px] text-[#00ff88] font-semibold">01 • TORCHVISION</div>
                      <div className="font-display font-bold text-sm text-white mt-1 group-hover:text-[#00ff88] transition-colors">
                        ResNet-18
                      </div>
                      <p className="text-[11px] text-white/50 font-sans mt-1">
                        18-layer residual convnet on ImageNet.
                      </p>
                    </div>
                    <button className="mt-3 text-[11px] text-[#00ff88] font-bold flex items-center gap-1 uppercase">
                      <Play className="w-3 h-3 fill-current" />
                      <span>Load ResNet</span>
                    </button>
                  </div>

                  {/* Card 2: MobileNet */}
                  <div
                    onClick={() => handleSelectModel('mobilenet')}
                    className="p-4 rounded-lg bg-white/[0.03] border border-white/10 hover:border-[#00f0ff]/60 cursor-pointer transition-all hover:bg-white/[0.06] flex flex-col justify-between group"
                  >
                    <div>
                      <div className="text-[10px] text-[#00f0ff] font-semibold">02 • TORCHVISION</div>
                      <div className="font-display font-bold text-sm text-white mt-1 group-hover:text-[#00f0ff] transition-colors">
                        MobileNet-V2
                      </div>
                      <p className="text-[11px] text-white/50 font-sans mt-1">
                        Inverted residual mobile vision architecture.
                      </p>
                    </div>
                    <button className="mt-3 text-[11px] text-[#00f0ff] font-bold flex items-center gap-1 uppercase">
                      <Play className="w-3 h-3 fill-current" />
                      <span>Load MobileNet</span>
                    </button>
                  </div>

                  {/* Card 3: Custom PyTorch Test Model */}
                  <div
                    onClick={() => handleSelectModel('custom_test')}
                    className="p-4 rounded-lg bg-white/[0.03] border border-white/10 hover:border-[#d500f9]/60 cursor-pointer transition-all hover:bg-white/[0.06] flex flex-col justify-between group"
                  >
                    <div>
                      <div className="text-[10px] text-[#d500f9] font-semibold">03 • TEST MODEL</div>
                      <div className="font-display font-bold text-sm text-white mt-1 group-hover:text-[#d500f9] transition-colors">
                        NeuralTrailCNN
                      </div>
                      <p className="text-[11px] text-white/50 font-sans mt-1">
                        custom/ test model on ShapeDataset.
                      </p>
                    </div>
                    <button className="mt-3 text-[11px] text-[#d500f9] font-bold flex items-center gap-1 uppercase">
                      <Play className="w-3 h-3 fill-current" />
                      <span>Load Test Model</span>
                    </button>
                  </div>
                </div>

                <div className="pt-2">
                  <button
                    onClick={() => setLandingModalOpen(true)}
                    className="text-xs text-white/50 hover:text-white flex items-center justify-center gap-2 mx-auto transition-colors"
                  >
                    <Upload className="w-3.5 h-3.5" />
                    <span>Upload your own .pt / .pth checkpoint...</span>
                  </button>
                </div>
              </div>
            </div>
          ) : (
            /* ACTIVE INVESTIGATION TABS (FLOW: FIND → TRACE → STRESS → GEMMA → EXPERIMENT → IMPROVE) */
            <>
              {activeTab === 'FIND' && (
                <BlindSpotMapView
                  samples={samples}
                  blindSpots={blindSpots}
                  metrics={metrics}
                  selectedClusterId={selectedClusterId}
                  onSelectCluster={setSelectedClusterId}
                  selectedSample={selectedSample}
                  onSelectSample={handleSelectSample}
                  filterState={filterState}
                  onFilterChange={(f) => setFilterState((prev) => ({ ...prev, ...f }))}
                  onSwitchToTab={setActiveTab}
                />
              )}

              {activeTab === 'TRACE' && <NeuralXRayView />}

              {activeTab === 'STRESS' && <WhatIfLabView initialSampleId={selectedSample?.id} />}

              {activeTab === 'IMPROVE' && (
                <BeforeAfterView 
                  experiment={activeExperiment}
                  interpretation={activeInterpretation}
                  metrics={metrics}
                  blindSpots={blindSpots}
                />
              )}

              {activeTab === 'REPLAY' && <FailureReplayView />}

              {activeTab === 'EXPERIMENTS' && (
                <ExperimentEngineView
                  telemetry={telemetry}
                  activeExperiment={activeExperiment}
                  experimentsList={experimentsList}
                  onSelectExperiment={(exp) => syncExperimentToTelemetry(exp)}
                  onRunNewTrial={handleRunNewTrial}
                  onCancelExperiment={handleCancelExperiment}
                  onInterpretExperiment={handleInterpretExperiment}
                  onNavigateToBeforeAfter={() => setActiveTab('IMPROVE')}
                />
              )}

              {activeTab === 'RESEARCH' && <ResearchAssistant />}
            </>
          )}
        </main>

        {/* Right Collapsible Utility Rail */}
        {isModelLoaded && activeTab !== 'RESEARCH' && (
          <ContextUtilityRail
            selectedBlindSpot={selectedSpot}
            selectedSample={selectedSample}
            selectedClusterId={selectedClusterId}
            onSelectCluster={setSelectedClusterId}
            onSwitchToTab={setActiveTab}
            onTriggerExperiment={handleTriggerExperiment}
            activeExperiment={activeExperiment}
            interpretation={activeInterpretation}
          />
        )}
      </div>

      {/* Bottom Telemetry Status Bar */}
      <TelemetryStatusBar telemetry={telemetry} />

      {/* Model Loader Modal */}
      <LandingHeroModal
        isOpen={landingModalOpen}
        onClose={() => setLandingModalOpen(false)}
        activeModelName={modelMetadata?.model_name || 'No Model Loaded'}
        isLoaded={isModelLoaded}
        onModelSelected={handleSelectModel}
      />
    </div>
  );
};
