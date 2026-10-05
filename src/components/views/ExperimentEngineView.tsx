import React, { useState } from 'react';
import { 
  Terminal, 
  Cpu, 
  Zap, 
  Cloud, 
  HardDrive, 
  CheckCircle2, 
  AlertTriangle, 
  XCircle, 
  Brain, 
  ArrowRight, 
  RefreshCw,
  GitCompare,
  Copy,
  Check,
  Activity
} from 'lucide-react';
import { Experiment, ExperimentTelemetry, ExperimentStatus } from '../../types/neuralTrail';
import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';
import { MetricCard } from '../common/MetricCard';
import { ProgressBar } from '../common/ProgressBar';

interface ExperimentEngineViewProps {
  telemetry: ExperimentTelemetry;
  activeExperiment?: Experiment | null;
  experimentsList?: Experiment[];
  onSelectExperiment?: (exp: Experiment) => void;
  onRunNewTrial?: (params?: { type: string; parameters: Record<string, any>; provider: string }) => void;
  onCancelExperiment?: (id: string) => void;
  onInterpretExperiment?: (id: string) => void;
  onNavigateToBeforeAfter?: () => void;
}

export const ExperimentEngineView: React.FC<ExperimentEngineViewProps> = ({
  telemetry,
  activeExperiment,
  experimentsList = [],
  onSelectExperiment,
  onRunNewTrial,
  onCancelExperiment,
  onInterpretExperiment,
  onNavigateToBeforeAfter,
}) => {
  // Modal state for spawning new trial
  const [spawnModalOpen, setSpawnModalOpen] = useState<boolean>(false);
  const [selectedProvider, setSelectedProvider] = useState<'local' | 'digitalocean'>('local');
  const [selectedType, setSelectedType] = useState<string>('invariant_mitigation');
  const [targetCluster, setTargetCluster] = useState<string>('bs-01');
  const [intensity, setIntensity] = useState<number>(20);
  const [copiedLogs, setCopiedLogs] = useState<boolean>(false);

  // Active logs & experiment source
  const currentExpId = activeExperiment?.id || telemetry.id || 'IDLE';
  const currentStatus: ExperimentStatus = (activeExperiment?.status as ExperimentStatus) || (telemetry.status as ExperimentStatus) || 'QUEUED';
  const currentProgress = activeExperiment ? activeExperiment.progress : telemetry.progress;
  const currentProvider = activeExperiment?.provider || 'local';
  const activeLogs = activeExperiment?.logs && activeExperiment.logs.length > 0 ? activeExperiment.logs : telemetry.logs;

  const samplesCount = activeExperiment?.results?.total_samples_evaluated ?? 
    activeExperiment?.results?.samples_processed ?? 
    telemetry.samplesProcessed ?? 
    0;

  const gpuLoad = activeExperiment?.results?.gpu_load ?? telemetry.gpuLoad ?? '--';
  const gpuTemp = activeExperiment?.results?.gpu_temp ?? telemetry.gpuTemp ?? '--';
  const currentLoss = activeExperiment?.results?.current_loss ?? telemetry.currentLoss ?? null;
  const accuracyDelta = activeExperiment?.results?.accuracy_delta ?? telemetry.accuracyDelta ?? '--';
  const flipRateDelta = activeExperiment?.results?.flip_rate_delta ?? '--';
  const repairedCount = activeExperiment?.results?.repaired_samples_count ?? 0;

  const handleLaunchTrial = () => {
    if (onRunNewTrial) {
      onRunNewTrial({
        type: selectedType,
        provider: selectedProvider,
        parameters: {
          cluster_id: targetCluster,
          intensity: intensity,
          apply_mitigation: true,
          experiment_protocol: `${selectedType.toUpperCase().replace('_', ' ')} (${selectedProvider.toUpperCase()})`,
        },
      });
    }
    setSpawnModalOpen(false);
  };

  const handleCopyLogs = () => {
    if (activeLogs) {
      navigator.clipboard.writeText(activeLogs.join('\n'));
      setCopiedLogs(true);
      setTimeout(() => setCopiedLogs(false), 2000);
    }
  };

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-[1440px] mx-auto w-full px-6 py-8 md:px-8 space-y-6">
        {/* Page Header */}
        <PageHeader
          stepNumber="06"
          stepCode="ENGINE"
          title="Experiment Telemetry Console"
          description="High-throughput execution abstraction serving local worker threads and cloud GPU instances."
          badge={
            <div className={`px-2.5 py-0.5 rounded text-xs font-mono font-medium flex items-center gap-1.5 border ${
              currentProvider === 'digitalocean'
                ? 'bg-[#00f0ff]/10 text-[#00f0ff] border-[#00f0ff]/30'
                : 'bg-[#7c4dff]/10 text-[#b388ff] border-[#7c4dff]/30'
            }`}>
              {currentProvider === 'digitalocean' ? (
                <>
                  <Cloud className="w-3.5 h-3.5" />
                  <span>DIGITALOCEAN GPU</span>
                </>
              ) : (
                <>
                  <HardDrive className="w-3.5 h-3.5" />
                  <span>LOCAL WORKER</span>
                </>
              )}
            </div>
          }
          actions={
            <button
              onClick={() => setSpawnModalOpen(true)}
              className="px-4 py-2 rounded-md bg-[#00ff88] text-black font-sans text-xs font-semibold flex items-center gap-2 hover:bg-[#00ff88]/90 transition-colors shadow-[0_0_15px_rgba(0,255,136,0.3)]"
            >
              <Zap className="w-4 h-4 fill-current" />
              <span>Spawn New Trial</span>
            </button>
          }
        />

        {/* 1. Top Metric Cards: ID, Status, Samples, GPU */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="EXPERIMENT ID"
            value={currentExpId}
            subValue={`Type: ${activeExperiment?.type || 'INVARIANT_MITIGATION'}`}
            accent="purple"
            icon={<Terminal className="w-3.5 h-3.5" />}
          />

          <MetricCard
            label="EXECUTION STATUS"
            value={currentStatus}
            subValue={`Provider: ${currentProvider.toUpperCase()}`}
            accent={currentStatus === 'COMPLETED' ? 'green' : currentStatus === 'FAILED' ? 'magenta' : 'amber'}
            icon={currentStatus === 'COMPLETED' ? <CheckCircle2 className="w-3.5 h-3.5 text-[#00ff88]" /> : <Activity className="w-3.5 h-3.5" />}
          />

          <MetricCard
            label="SAMPLES EVALUATED"
            value={samplesCount > 0 ? samplesCount.toLocaleString() : '--'}
            subValue={currentProvider === 'digitalocean' ? 'Cloud Droplet Pool' : 'Local Workers'}
            accent="cyan"
            icon={<Cpu className="w-3.5 h-3.5" />}
          />

          <MetricCard
            label="GPU LOAD / THERMALS"
            value={gpuLoad !== '--' ? `${gpuLoad} Active` : '--'}
            subValue={`Thermals: ${gpuTemp}`}
            accent="green"
            icon={<Zap className="w-3.5 h-3.5" />}
          />
        </div>

        {/* 2. Middle: Execution Progress */}
        <Card
          headerTitle="Execution Progress"
          headerSubtitle={`Runtime trial status on ${currentProvider.toUpperCase()} compute`}
          headerIcon={<Cpu className="w-4 h-4 text-[#00ff88]" />}
          headerAction={
            (currentStatus === 'RUNNING' || currentStatus === 'QUEUED') && onCancelExperiment && (
              <button
                onClick={() => onCancelExperiment(currentExpId)}
                className="px-2.5 py-1 rounded text-xs font-mono font-medium bg-[#ff007f]/10 text-[#ff007f] border border-[#ff007f]/30 hover:bg-[#ff007f]/20 transition-colors"
              >
                Cancel Job
              </button>
            )
          }
        >
          <div className="space-y-4">
            <ProgressBar
              value={currentProgress}
              accent="green"
              size="lg"
              showLabel
              label="Overall Evaluation Completion"
            />

            {/* Empirical Summary Strip */}
            <div className="grid grid-cols-2 md:grid-cols-4 gap-3 pt-3 border-t border-white/10 text-xs font-mono">
              <div className="p-2.5 rounded bg-white/[0.02] border border-white/5">
                <span className="text-white/40 block text-[11px] font-sans">Current Loss</span>
                <span className="text-[#ff007f] font-semibold text-sm mt-0.5 block">
                  {currentLoss !== null ? currentLoss.toFixed(4) : '--'}
                </span>
              </div>
              <div className="p-2.5 rounded bg-white/[0.02] border border-white/5">
                <span className="text-white/40 block text-[11px] font-sans">Accuracy Delta</span>
                <span className="text-[#00ff88] font-semibold text-sm mt-0.5 block">
                  {accuracyDelta}
                </span>
              </div>
              <div className="p-2.5 rounded bg-white/[0.02] border border-white/5">
                <span className="text-white/40 block text-[11px] font-sans">Flip Rate Delta</span>
                <span className="text-[#00f0ff] font-semibold text-sm mt-0.5 block">
                  {flipRateDelta}
                </span>
              </div>
              <div className="p-2.5 rounded bg-white/[0.02] border border-white/5">
                <span className="text-white/40 block text-[11px] font-sans">Restored Samples</span>
                <span className="text-[#b388ff] font-semibold text-sm mt-0.5 block">
                  {repairedCount > 0 ? `${repairedCount} Samples` : '--'}
                </span>
              </div>
            </div>
          </div>
        </Card>

        {/* Completed Trial Action Card */}
        {currentStatus === 'COMPLETED' && (
          <div className="p-4 rounded-lg bg-[#14231b] border border-[#00ff88]/40 flex flex-wrap items-center justify-between gap-4">
            <div className="flex items-center gap-3 min-w-0">
              <CheckCircle2 className="w-5 h-5 text-[#00ff88] shrink-0" />
              <div>
                <div className="font-sans font-semibold text-sm text-white">
                  Trial Execution Complete
                </div>
                <p className="text-xs text-white/70 font-sans mt-0.5 truncate">
                  {activeExperiment?.results?.summary || `Trial completed successfully with ${accuracyDelta} accuracy recovery.`}
                </p>
              </div>
            </div>

            <div className="flex items-center gap-2.5 shrink-0">
              {onInterpretExperiment && (
                <button
                  onClick={() => onInterpretExperiment(currentExpId)}
                  className="px-3.5 py-1.5 rounded-md bg-[#251245] border border-[#d500f9]/40 text-[#d500f9] hover:bg-[#32175e] text-xs font-sans font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <Brain className="w-3.5 h-3.5" />
                  <span>Interpret Results</span>
                </button>
              )}
              {onNavigateToBeforeAfter && (
                <button
                  onClick={onNavigateToBeforeAfter}
                  className="px-3.5 py-1.5 rounded-md bg-[#00ff88] text-black hover:bg-[#00ff88]/90 text-xs font-sans font-semibold flex items-center gap-1.5 transition-colors"
                >
                  <GitCompare className="w-3.5 h-3.5" />
                  <span>Compare Models</span>
                  <ArrowRight className="w-3.5 h-3.5" />
                </button>
              )}
            </div>
          </div>
        )}

        {/* 3. Bottom Split: Telemetry Terminal + Experiment Registry */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
          {/* Telemetry Stream Terminal (7 cols) */}
          <div className="lg:col-span-7">
            <Card
              headerTitle="Runtime Telemetry Log Stream"
              headerSubtitle={`Real-time stdout/stderr from ${currentExpId}`}
              headerIcon={<Terminal className="w-4 h-4 text-[#00f0ff]" />}
              headerAction={
                <button
                  onClick={handleCopyLogs}
                  className="p-1.5 rounded bg-white/5 hover:bg-white/10 text-white/50 hover:text-white transition-colors text-xs flex items-center gap-1"
                  title="Copy log buffer"
                >
                  {copiedLogs ? <Check className="w-3.5 h-3.5 text-[#00ff88]" /> : <Copy className="w-3.5 h-3.5" />}
                </button>
              }
              noPadding
            >
              <div className="h-72 bg-[#090214] p-4 font-mono text-xs overflow-y-auto space-y-1 text-white/80 leading-relaxed border-t border-white/5">
                {activeLogs && activeLogs.length > 0 ? (
                  activeLogs.map((log, idx) => (
                    <div key={idx} className="flex gap-2">
                      <span className="text-white/30 select-none shrink-0">{String(idx + 1).padStart(2, '0')}</span>
                      <span className="break-all font-mono">{log}</span>
                    </div>
                  ))
                ) : (
                  <div className="text-white/40 italic">Awaiting telemetry logs from worker...</div>
                )}
              </div>
            </Card>
          </div>

          {/* Experiment Registry List (5 cols) */}
          <div className="lg:col-span-5">
            <Card
              headerTitle="Experiment Registry"
              headerSubtitle={`${experimentsList.length} total trials recorded`}
              noPadding
            >
              <div className="max-h-72 overflow-y-auto divide-y divide-white/5 border-t border-white/5">
                {experimentsList.length > 0 ? (
                  experimentsList.map((exp) => {
                    const isSelected = exp.id === currentExpId;
                    return (
                      <button
                        key={exp.id}
                        onClick={() => onSelectExperiment && onSelectExperiment(exp)}
                        className={`w-full p-3.5 text-left flex items-center justify-between transition-colors ${
                          isSelected ? 'bg-white/[0.06]' : 'hover:bg-white/[0.02]'
                        }`}
                      >
                        <div className="min-w-0 pr-2">
                          <div className="flex items-center gap-2">
                            <span className="font-mono text-xs font-semibold text-white">{exp.id}</span>
                            <span className={`text-[10px] font-mono px-1.5 py-0.2 rounded border ${
                              exp.status === 'COMPLETED'
                                ? 'bg-[#00ff88]/10 text-[#00ff88] border-[#00ff88]/30'
                                : exp.status === 'FAILED'
                                ? 'bg-[#ff007f]/10 text-[#ff007f] border-[#ff007f]/30'
                                : 'bg-[#ffb300]/10 text-[#ffb300] border-[#ffb300]/30'
                            }`}>
                              {exp.status}
                            </span>
                          </div>
                          <div className="text-xs text-white/50 font-sans truncate mt-0.5">
                            {exp.parameters?.experiment_protocol || exp.type}
                          </div>
                        </div>

                        <div className="text-right shrink-0">
                          <span className="font-mono text-xs text-white/60 block">
                            {exp.progress}%
                          </span>
                          <span className="text-[10px] font-mono text-white/40 uppercase">
                            {exp.provider}
                          </span>
                        </div>
                      </button>
                    );
                  })
                ) : (
                  <div className="p-4 text-center text-xs text-white/40 font-sans">
                    No experiments registered yet.
                  </div>
                )}
              </div>
            </Card>
          </div>
        </div>

        {/* Modal: Spawn New Trial */}
        {spawnModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/70 backdrop-blur-sm">
            <div className="w-full max-w-md bg-[#130726] border border-white/20 rounded-lg p-6 space-y-5 shadow-2xl">
              <div className="flex items-center justify-between border-b border-white/10 pb-3">
                <h3 className="font-sans font-semibold text-base text-white">Spawn Experiment Trial</h3>
                <button
                  onClick={() => setSpawnModalOpen(false)}
                  className="text-white/40 hover:text-white transition-colors"
                >
                  ✕
                </button>
              </div>

              <div className="space-y-4">
                {/* Compute Provider Selection */}
                <div className="space-y-1.5">
                  <label className="text-xs font-sans text-white/60">Execution Compute Provider</label>
                  <div className="grid grid-cols-2 gap-2">
                    <button
                      type="button"
                      onClick={() => setSelectedProvider('local')}
                      className={`p-2.5 rounded-md border text-xs font-sans font-medium flex items-center justify-center gap-2 transition-colors ${
                        selectedProvider === 'local'
                          ? 'bg-[#180933] border-[#00f0ff] text-white'
                          : 'bg-white/[0.02] border-white/10 text-white/60 hover:text-white'
                      }`}
                    >
                      <HardDrive className="w-4 h-4" />
                      <span>Local PyTorch</span>
                    </button>

                    <button
                      type="button"
                      onClick={() => setSelectedProvider('digitalocean')}
                      className={`p-2.5 rounded-md border text-xs font-sans font-medium flex items-center justify-center gap-2 transition-colors ${
                        selectedProvider === 'digitalocean'
                          ? 'bg-[#180933] border-[#00f0ff] text-white'
                          : 'bg-white/[0.02] border-white/10 text-white/60 hover:text-white'
                      }`}
                    >
                      <Cloud className="w-4 h-4" />
                      <span>DigitalOcean GPU</span>
                    </button>
                  </div>
                </div>

                {/* Protocol Type */}
                <div className="space-y-1.5">
                  <label className="text-xs font-sans text-white/60">Remediation Protocol</label>
                  <select
                    value={selectedType}
                    onChange={(e) => setSelectedType(e.target.value)}
                    className="w-full bg-[#0a0314] border border-white/15 rounded-md px-3 py-2 text-xs font-sans text-white focus:outline-none focus:border-[#00f0ff]"
                  >
                    <option value="invariant_mitigation">Invariant Representation Regularization</option>
                    <option value="dihedral_data_aug">SO(2) Dihedral Symmetry Augmentation</option>
                    <option value="feature_denoise">Adversarial Spatial Denoising</option>
                  </select>
                </div>

                {/* Target Cluster */}
                <div className="space-y-1.5">
                  <label className="text-xs font-sans text-white/60">Target Failure Cluster</label>
                  <input
                    type="text"
                    value={targetCluster}
                    onChange={(e) => setTargetCluster(e.target.value)}
                    className="w-full bg-[#0a0314] border border-white/15 rounded-md px-3 py-2 text-xs font-mono text-white focus:outline-none focus:border-[#00f0ff]"
                    placeholder="e.g. bs-01"
                  />
                </div>

                {/* Intensity Slider */}
                <div className="space-y-1">
                  <div className="flex justify-between text-xs font-sans">
                    <span className="text-white/60">Regularization Intensity</span>
                    <span className="font-mono text-white">{intensity}%</span>
                  </div>
                  <input
                    type="range"
                    min="5"
                    max="50"
                    step="5"
                    value={intensity}
                    onChange={(e) => setIntensity(Number(e.target.value))}
                    className="w-full accent-[#00f0ff] cursor-pointer"
                  />
                </div>
              </div>

              <div className="flex items-center justify-end gap-2 pt-3 border-t border-white/10">
                <button
                  type="button"
                  onClick={() => setSpawnModalOpen(false)}
                  className="px-3.5 py-1.5 rounded-md border border-white/15 text-xs text-white/60 hover:text-white transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={handleLaunchTrial}
                  className="px-4 py-1.5 rounded-md bg-[#00ff88] text-black font-sans text-xs font-semibold hover:bg-[#00ff88]/90 transition-colors"
                >
                  Launch Trial
                </button>
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
