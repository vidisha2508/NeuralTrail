from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field


# ==========================================
# Single Image Inference & Activation Schemas
# ==========================================

class PredictionItem(BaseModel):
    index: int
    label: str
    probability: float
    confidence_percentage: float


class PredictionResponse(BaseModel):
    model_name: str
    top_label: str
    top_index: int
    confidence: float
    predictions: List[PredictionItem]
    inference_time_ms: float
    input_shape: List[int]


class EmbeddingResponse(BaseModel):
    model_name: str
    dimension: int
    embedding: List[float]
    norm: float
    inference_time_ms: float


class LayerActivation(BaseModel):
    layer_name: str
    shape: List[int]
    mean: float
    std: float
    max: float
    min: float
    sparsity: float
    heatmap_2d: Optional[List[List[float]]] = None


class ActivationsResponse(BaseModel):
    model_name: str
    layers: List[LayerActivation]
    inference_time_ms: float


class ModelMetadataResponse(BaseModel):
    model_name: str
    architecture: str
    device: str
    total_parameters: int
    trainable_parameters: int
    num_classes: int
    input_resolution: List[int]
    checkpoint_path: Optional[str]
    is_loaded: bool
    layer_names: List[str]


class ModelLoadRequest(BaseModel):
    checkpoint_path: Optional[str] = None
    model_type: Optional[str] = "resnet18"
    device: Optional[str] = None


class ModelLoadResponse(BaseModel):
    success: bool
    message: str
    metadata: ModelMetadataResponse


# ==========================================
# Neural Trail Analysis Engine Schemas
# ==========================================

class AnalysisStartRequest(BaseModel):
    dataset_source: Optional[str] = Field(default="benchmark", description="benchmark or custom")
    sample_count: Optional[int] = Field(default=100, ge=10, le=500, description="Total sample count")
    reducer_type: Optional[str] = Field(default="pca", description="pca, umap, or tsne")
    k_clusters: Optional[int] = Field(default=None, description="Optional fixed cluster count")
    custom_dir: Optional[str] = Field(default=None, description="Directory path for custom dataset")
    async_run: Optional[bool] = Field(default=False, description="Run asynchronously in background")


class AnalysisStatusResponse(BaseModel):
    is_running: bool
    stage: str
    progress: int
    has_results: bool
    total_samples: int
    blind_spots_count: int
    last_run_time: Optional[float] = None
    error: Optional[str] = None


class OverallVsBlindSpotComparison(BaseModel):
    overallAccuracy: float
    blindSpotAccuracy: float
    annotation: str


class ModelMetrics(BaseModel):
    name: str
    architecture: str
    dataset: str
    health: float
    healthStatus: str
    accuracy: float
    accuracyDelta: str
    avgConfidence: float
    blindSpotCount: int
    highSeverityCount: int
    totalSamples: int
    overallVsBlindSpotComparison: OverallVsBlindSpotComparison


class AnalysisSummaryResponse(BaseModel):
    has_data: bool
    dataset_name: Optional[str] = None
    total_samples: Optional[int] = 0
    reducer: Optional[str] = None
    metrics: Optional[ModelMetrics] = None
    blind_spots_count: int = 0
    latency_ms: Optional[float] = None
    timestamp: Optional[float] = None


class SampleTrackItem(BaseModel):
    id: str
    x: float
    y: float
    trueClass: int
    trueClassName: str
    predictedClass: int
    predictedClassName: str
    confidence: float
    correct: bool
    stability: str
    clusterId: Optional[str] = None
    sensitivity: Optional[str] = None
    perturbation: Optional[str] = None
    features: Optional[Dict[str, float]] = None
    thumbnailGrid: Optional[List[List[float]]] = None


class SamplesResponse(BaseModel):
    total: int
    offset: int
    limit: int
    samples: List[SampleTrackItem]


class ClusterCenter(BaseModel):
    x: float
    y: float


class BlindSpotItem(BaseModel):
    id: str
    cluster_number: str
    name: str
    severity: str
    sample_count: int
    accuracy: float
    failure_rate: float
    average_confidence: float
    common_prediction_error: str
    cluster_center: ClusterCenter
    radius: float
    description: str
    mitigation_suggestion: str
    representative_sample_ids: Optional[List[str]] = None


class BlindSpotDetailResponse(BaseModel):
    id: str
    cluster_number: str
    name: str
    severity: str
    sample_count: int
    accuracy: float
    failure_rate: float
    average_confidence: float
    common_prediction_error: str
    dominant_class: Optional[str] = None
    sensitivity: Optional[str] = None
    cluster_center: ClusterCenter
    radius: float
    description: str
    mitigation_suggestion: str
    representative_samples: List[SampleTrackItem]
    samples: List[SampleTrackItem]
    error_distribution: List[Dict[str, Any]]


# ==========================================
# Neural X-Ray: Activation Pathway Schemas
# ==========================================

class ActivationLayerItem(BaseModel):
    id: str
    name: str
    stage: str
    tensorShape: str
    activationNorm: float
    deadNeuronRatio: float
    activationNormPerturbed: Optional[float] = None
    deadNeuronRatioPerturbed: Optional[float] = None
    anomalyScore: float
    normalFeatureMap: List[List[float]]
    perturbedFeatureMap: List[List[float]]
    differenceMap: Optional[List[List[float]]] = None


class XRaySampleMeta(BaseModel):
    id: str
    trueClass: int
    trueClassName: str
    predictedClass: int
    predictedClassName: str
    confidence: float
    correct: bool
    stability: str
    perturbation: Optional[str] = None


class ActivationPathwayCompareRequest(BaseModel):
    sample_a_id: Optional[str] = None
    sample_b_id: Optional[str] = None
    selected_layer_id: Optional[str] = "layer-middle"


class ActivationPathwayResponse(BaseModel):
    feature_name: str = "Activation Pathway Visualization"
    selected_layer_id: str
    layers: List[ActivationLayerItem]
    sample_a: XRaySampleMeta
    sample_b: XRaySampleMeta
    available_successful_samples: Optional[List[XRaySampleMeta]] = None
    available_failed_samples: Optional[List[XRaySampleMeta]] = None
    causality_disclaimer: str


# ==========================================
# What-If / Perturbation Lab Schemas
# ==========================================

class WhatIfExperimentRequest(BaseModel):
    sample_id: Optional[str] = None
    rotation: Optional[float] = Field(default=0.0, description="Rotation angle in degrees (-45 to 45)")
    noise: Optional[float] = Field(default=0.0, description="Gaussian noise percent (0 to 100)")
    blur: Optional[float] = Field(default=0.0, description="Blur kernel radius in px (0 to 12)")
    crop: Optional[float] = Field(default=0.0, description="Scale/crop percent (0 to 50)")
    brightness: Optional[float] = Field(default=0.0, description="Brightness shift percent (-80 to 80)")
    occlusion: Optional[float] = Field(default=0.0, description="Occlusion patch percent (0 to 60)")
    include_images: Optional[bool] = Field(default=True, description="Return base64 data URLs for UI preview")


class WhatIfExperimentResponse(BaseModel):
    sample_id: str
    original_prediction: str
    original_prediction_idx: int
    original_confidence: float
    original_confidence_percentage: float
    perturbed_prediction: str
    perturbed_prediction_idx: int
    perturbed_confidence: float
    perturbed_confidence_percentage: float
    confidence_change: float
    confidence_change_percentage: float
    prediction_flip: bool
    robustness_score: int
    parameters_applied: Dict[str, float]
    original_image_url: Optional[str] = None
    perturbed_image_url: Optional[str] = None
    top_original_predictions: List[PredictionItem]
    top_perturbed_predictions: List[PredictionItem]
    latency_ms: float


class WhatIfSampleCandidate(BaseModel):
    id: str
    trueClass: int
    trueClassName: str
    confidence: float
    correct: bool
    thumbnailGrid: Optional[List[List[float]]] = None


# ==========================================
# Gemma 4 AI Researcher Schemas
# ==========================================

class StructuredEvidence(BaseModel):
    model: str = Field(default="ResNet18", description="Target model under test")
    overall_accuracy: float = Field(..., description="Overall model accuracy across dataset")
    blind_spot_accuracy: float = Field(..., description="Local accuracy within the failure manifold / blind spot")
    common_error: str = Field(..., description="Dominant error pattern (e.g. cat_to_dog)")
    rotation_flip_rate: float = Field(..., description="Flip rate under rotation stress perturbation")
    cluster_id: Optional[str] = Field(default=None, description="Identifier of the blind spot cluster")
    cluster_name: Optional[str] = Field(default=None, description="Human readable name of the cluster")
    sample_count: Optional[int] = Field(default=None, description="Total sample count in cluster")
    additional_metrics: Optional[Dict[str, Any]] = Field(default=None, description="Additional empirical metrics from Analysis Engine")


class GemmaResearchOutput(BaseModel):
    observation: str = Field(..., description="Concise empirical summary citing exact numbers from evidence")
    hypothesis: str = Field(..., description="Mechanistic explanation of representation failure")
    recommended_experiment: str = Field(..., description="Specific test or intervention protocol")
    expected_signal: str = Field(..., description="Measurable target delta citing baseline metrics")
    priority: str = Field(default="high", description="Priority level: high, medium, or low")


class GemmaHypothesizeRequest(BaseModel):
    cluster_id: Optional[str] = Field(default=None, description="Blind spot cluster ID to analyze")
    evidence: Optional[StructuredEvidence] = Field(default=None, description="Optional manual evidence override")


class GemmaHypothesizeResponse(BaseModel):
    evidence: StructuredEvidence
    research_output: GemmaResearchOutput
    model_version: str = "Gemma 4 (AI Researcher)"
    tokens_used: int = Field(default=0, description="Estimated total tokens used (minimized)")
    source: str = Field(default="gemma_4_research_engine", description="Inference source")


class ResearcherExperimentRunRequest(BaseModel):
    cluster_id: Optional[str] = Field(default=None, description="Target cluster ID")
    experiment_protocol: Optional[str] = Field(default=None, description="Protocol description")
    perturbation_type: Optional[str] = Field(default="rotation", description="Target perturbation")
    intensity: Optional[float] = Field(default=20.0, description="Perturbation intensity")
    apply_mitigation: Optional[bool] = Field(default=True, description="Apply test-time invariant mitigation")


class ResearcherExperimentRunResponse(BaseModel):
    cluster_id: str
    experiment_protocol: str
    baseline_accuracy: float
    observed_accuracy: float
    accuracy_delta: str
    baseline_flip_rate: float
    post_flip_rate: float
    flip_rate_delta: str
    repaired_samples_count: int
    total_samples_evaluated: int
    observed_outcome: str
    hypothesis_confirmed: bool
    telemetry_logs: List[str]


# ==========================================
# Experiment Execution Layer Schemas
# (ComputeProvider: Local & DigitalOcean)
# ==========================================

class ExperimentCreateRequest(BaseModel):
    type: Optional[str] = Field(default="invariant_mitigation", description="Job type: invariant_mitigation, perturbation_sweep, batch_inference, gemma_reasoning")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Job parameters (cluster_id, intensity, etc.)")
    provider: Optional[str] = Field(default=None, description="Compute provider: 'local' or 'digitalocean'. Defaults to COMPUTE_PROVIDER env.")


class ExperimentCancelResponse(BaseModel):
    success: bool
    message: str
    experiment_id: str


class GemmaInterpretationRequest(BaseModel):
    experiment_id: str


class GemmaInterpretationResponse(BaseModel):
    experiment_id: str
    cluster_id: str
    confirmation_status: str
    interpretation: str
    recommended_action: str
    before_after_patch_ready: bool
    suggested_patch: Optional[str] = None
    metrics: Dict[str, Any]




