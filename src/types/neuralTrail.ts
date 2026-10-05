export type StabilityType = 'stable' | 'uncertain' | 'failure';
export type SeverityType = 'HIGH' | 'MEDIUM' | 'LOW';

export interface Sample {
  id: string;
  x: number;
  y: number;
  trueClass: number;
  predictedClass: number;
  confidence: number;
  correct: boolean;
  stability: StabilityType;
  clusterId?: string;
  sensitivity: string;
  thumbnailGrid?: number[][];
  features?: {
    rotation: number;
    brightness: number;
    blur: number;
  };
}

export interface BlindSpot {
  id: string;
  clusterNumber: string; // e.g. '#01'
  name: string;
  severity: SeverityType;
  sampleCount: number;
  accuracy: number;
  confidence: number;
  commonConfusion: string;
  sensitivity: string;
  clusterCenter: { x: number; y: number };
  radius: number;
  description: string;
  mitigationSuggestion: string;
}

export interface ModelMetrics {
  name: string;
  architecture: string;
  dataset: string;
  health: number;
  healthStatus: string;
  accuracy: number;
  accuracyDelta: string;
  avgConfidence: number;
  blindSpotCount: number;
  highSeverityCount: number;
  totalSamples: number;
  overallVsBlindSpotComparison: {
    overallAccuracy: number;
    blindSpotAccuracy: number;
    annotation: string;
  };
}

export interface FilterState {
  status: 'all' | 'stable' | 'uncertain' | 'failure';
  minConfidence: number;
  selectedClass: string;
  perturbation: string;
  searchQuery: string;
  selectedClusterId: string | null;
}

export interface StructuredEvidence {
  model: string;
  overall_accuracy: number;
  blind_spot_accuracy: number;
  common_error: string;
  rotation_flip_rate: number;
  cluster_id?: string;
  cluster_name?: string;
  sample_count?: number;
  additional_metrics?: Record<string, any>;
}

export interface GemmaResearchOutput {
  observation: string;
  hypothesis: string;
  recommended_experiment: string;
  expected_signal: string;
  priority: 'high' | 'medium' | 'low';
}

export interface GemmaHypothesisResponse {
  evidence: StructuredEvidence;
  research_output: GemmaResearchOutput;
  model_version: string;
  tokens_used: number;
  source: string;
}

export interface ResearcherExperimentResult {
  cluster_id: string;
  experiment_protocol: string;
  baseline_accuracy: number;
  observed_accuracy: number;
  accuracy_delta: string;
  baseline_flip_rate: number;
  post_flip_rate: number;
  flip_rate_delta: string;
  repaired_samples_count: number;
  total_samples_evaluated: number;
  observed_outcome: string;
  hypothesis_confirmed: boolean;
  telemetry_logs: string[];
}

export interface GemmaInvestigation {
  clusterId: string;
  clusterName: string;
  observation: string;
  hypothesis: string;
  experiment: string;
  expectedSignal: string;
  result: string;
  suggestedPatch: string;
}

export interface ActivationLayer {
  id: string;
  name: string;
  stage: 'INPUT' | 'EARLY' | 'MIDDLE' | 'DEEP' | 'OUTPUT';
  tensorShape: string;
  activationNorm: number;
  activationNormPerturbed?: number;
  deadNeuronRatio: number;
  normalFeatureMap: number[][]; // 6x6 or 8x8 feature activation
  perturbedFeatureMap: number[][];
  anomalyScore: number;
}

export interface ExperimentTelemetry {
  id: string;
  name: string;
  status: 'IDLE' | 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED' | 'STANDBY';
  progress: number; // 0 - 100
  samplesProcessed: number;
  totalSamples: number;
  gpuTemp: string;
  gpuLoad: string;
  currentLoss: number;
  accuracyDelta: string;
  logs: string[];
}

export type ExperimentStatus = 'QUEUED' | 'RUNNING' | 'COMPLETED' | 'FAILED';

export interface ExperimentTimestamps {
  created_at: number;
  started_at?: number | null;
  completed_at?: number | null;
}

export interface ExperimentResult {
  cluster_id?: string;
  protocol?: string;
  baseline_accuracy?: number;
  observed_accuracy?: number;
  accuracy_delta?: string;
  baseline_flip_rate?: number;
  post_flip_rate?: number;
  flip_rate_delta?: string;
  repaired_samples_count?: number;
  total_samples_evaluated?: number;
  current_loss?: number;
  gpu_load?: string;
  gpu_temp?: string;
  hypothesis_confirmed?: boolean;
  summary?: string;
  provider?: string;
  region?: string;
  droplet_id?: string;
  samples_processed?: number;
  [key: string]: any;
}

export interface Experiment {
  id: string;
  type: string;
  parameters: Record<string, any>;
  status: ExperimentStatus;
  progress: number; // 0 - 100
  results?: ExperimentResult | null;
  timestamps: ExperimentTimestamps;
  logs: string[];
  provider: string; // 'local' | 'digitalocean'
  error?: string | null;
}

export interface ComputeProviderInfo {
  active_provider: string;
  default_provider: string;
  supported_providers: string[];
  is_cloud_configured: boolean;
  region: string;
  capabilities: string[];
}

export interface GemmaInterpretationResponse {
  experiment_id: string;
  cluster_id: string;
  confirmation_status: 'CONFIRMED' | 'REFUTED' | string;
  interpretation: string;
  recommended_action: string;
  before_after_patch_ready: boolean;
  suggested_patch?: string | null;
  metrics: Record<string, any>;
}

export type AnalysisStage = 'IDLE' | 'ANALYZING' | 'MAPPING' | 'SCANNING' | 'COMPLETE';
export type WorkstationTab = 'FIND' | 'TRACE' | 'STRESS' | 'IMPROVE' | 'REPLAY' | 'EXPERIMENTS' | 'RESEARCH';


