/**
 * Real ML Backend API Client for Neural Trail
 * Connects to the FastAPI PyTorch Inference Engine (Default: http://127.0.0.1:8000)
 */

import { Experiment, ComputeProviderInfo, GemmaInterpretationResponse } from '../types/neuralTrail';

const API_BASE_URL = (import.meta as any).env?.VITE_API_URL || 'http://127.0.0.1:8000';

export interface PredictionItem {
  index: number;
  label: string;
  probability: number;
  confidence_percentage: number;
}

export interface PredictionResponse {
  model_name: string;
  top_label: string;
  top_index: number;
  confidence: number;
  predictions: PredictionItem[];
  inference_time_ms: number;
  input_shape: number[];
}

export interface EmbeddingResponse {
  model_name: string;
  dimension: number;
  embedding: number[];
  norm: number;
  inference_time_ms: number;
}

export interface LayerActivation {
  layer_name: string;
  shape: number[];
  mean: number;
  std: number;
  max: number;
  min: number;
  sparsity: number;
  heatmap_2d?: number[][];
}

export interface ActivationsResponse {
  model_name: string;
  layers: LayerActivation[];
  inference_time_ms: number;
}

export interface ModelMetadata {
  model_name: string;
  architecture: string;
  device: string;
  total_parameters: number;
  trainable_parameters: number;
  num_classes: number;
  input_resolution: number[];
  checkpoint_path?: string;
  is_loaded: boolean;
  layer_names: string[];
}

export class MLApiClient {
  private baseUrl: string;

  constructor(baseUrl: string = API_BASE_URL) {
    this.baseUrl = baseUrl;
  }

  async checkHealth(): Promise<{ status: string; service: string; model_loaded: boolean; device: string }> {
    const res = await fetch(`${this.baseUrl}/api/health`);
    if (!res.ok) throw new Error(`Health check failed: ${res.statusText}`);
    return res.json();
  }

  async exportModel(): Promise<Blob> {
    const res = await fetch(`${this.baseUrl}/api/model/export`, { method: 'POST' });
    if (!res.ok) throw new Error(`Model export failed: ${res.statusText}`);
    return res.blob();
  }


  async getModelInfo(): Promise<ModelMetadata> {
    const res = await fetch(`${this.baseUrl}/api/model/info`);
    if (!res.ok) throw new Error(`Failed to fetch model info: ${res.statusText}`);
    return res.json();
  }

  async predict(imageFile: File | Blob, topK = 5): Promise<PredictionResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const res = await fetch(`${this.baseUrl}/api/model/predict?top_k=${topK}`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`Inference failed: ${res.statusText}`);
    return res.json();
  }

  async getEmbedding(imageFile: File | Blob): Promise<EmbeddingResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const res = await fetch(`${this.baseUrl}/api/model/embedding`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`Embedding extraction failed: ${res.statusText}`);
    return res.json();
  }

  async getActivations(imageFile: File | Blob): Promise<ActivationsResponse> {
    const formData = new FormData();
    formData.append('file', imageFile);

    const res = await fetch(`${this.baseUrl}/api/model/activations`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) throw new Error(`Activation extraction failed: ${res.statusText}`);
    return res.json();
  }

  async loadModel(checkpointPath?: string, modelType = 'resnet18'): Promise<{ success: boolean; message: string; metadata: ModelMetadata }> {
    const res = await fetch(`${this.baseUrl}/api/model/load`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ checkpoint_path: checkpointPath, model_type: modelType }),
    });
    if (!res.ok) throw new Error(`Model load failed: ${res.statusText}`);
    return res.json();
  }

  async loadCustomTestModel(): Promise<{ success: boolean; message: string; metadata: ModelMetadata }> {
    const res = await fetch(`${this.baseUrl}/api/model/load-custom-test`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Custom test model load failed: ${res.statusText}`);
    return res.json();
  }

  async unloadModel(): Promise<{ success: boolean; message: string }> {
    const res = await fetch(`${this.baseUrl}/api/model/unload`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Model unload failed: ${res.statusText}`);
    return res.json();
  }

  async uploadCustomModel(formData: FormData): Promise<{ success: boolean; message: string; metadata: ModelMetadata }> {
    const res = await fetch(`${this.baseUrl}/api/model/upload`, {
      method: 'POST',
      body: formData,
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `Upload failed: ${res.statusText}`);
    }
    return res.json();
  }

  async getReplayEvents(): Promise<{ total: number; events: any[] }> {
    const res = await fetch(`${this.baseUrl}/api/analysis/replay`);
    if (!res.ok) return { total: 0, events: [] };
    return res.json();
  }

  async selectInvestigationSample(sampleId: string): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/analysis/select-sample?sample_id=${encodeURIComponent(sampleId)}`, {
      method: 'POST',
    });
    if (!res.ok) return null;
    return res.json();
  }

  // ==========================================
  // Neural Trail Analysis Engine Endpoints
  // ==========================================

  async startAnalysis(params?: {
    dataset_source?: string;
    sample_count?: number;
    reducer_type?: string;
    k_clusters?: number;
    custom_dir?: string;
    async_run?: boolean;
  }): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/analysis/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params || {}),
    });
    if (!res.ok) throw new Error(`Start analysis failed: ${res.statusText}`);
    return res.json();
  }

  async getAnalysisStatus(): Promise<{
    is_running: boolean;
    stage: string;
    progress: number;
    has_results: boolean;
    total_samples: number;
    blind_spots_count: number;
    last_run_time?: number;
    error?: string;
  }> {
    const res = await fetch(`${this.baseUrl}/api/analysis/status`);
    if (!res.ok) throw new Error(`Fetch status failed: ${res.statusText}`);
    return res.json();
  }

  async getAnalysisSummary(): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/analysis/summary`);
    if (!res.ok) throw new Error(`Fetch summary failed: ${res.statusText}`);
    return res.json();
  }

  async getAnalysisSamples(params?: {
    cluster_id?: string;
    status?: string;
    min_confidence?: number;
    class_id?: number;
    limit?: number;
    offset?: number;
  }): Promise<{ total: number; offset: number; limit: number; samples: any[] }> {
    const query = new URLSearchParams();
    if (params?.cluster_id) query.set('cluster_id', params.cluster_id);
    if (params?.status && params.status !== 'all') query.set('status', params.status);
    if (params?.min_confidence !== undefined) query.set('min_confidence', String(params.min_confidence));
    if (params?.class_id !== undefined) query.set('class_id', String(params.class_id));
    if (params?.limit !== undefined) query.set('limit', String(params.limit));
    if (params?.offset !== undefined) query.set('offset', String(params.offset));

    const qs = query.toString();
    const url = `${this.baseUrl}/api/analysis/samples${qs ? `?${qs}` : ''}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Fetch samples failed: ${res.statusText}`);
    return res.json();
  }

  async getBlindSpots(): Promise<any[]> {
    const res = await fetch(`${this.baseUrl}/api/analysis/blind-spots`);
    if (!res.ok) throw new Error(`Fetch blind spots failed: ${res.statusText}`);
    return res.json();
  }

  async getBlindSpotDetail(clusterId: string): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/analysis/blind-spots/${encodeURIComponent(clusterId)}`);
    if (!res.ok) throw new Error(`Fetch blind spot detail failed: ${res.statusText}`);
    return res.json();
  }

  async getActivationPathway(params?: {
    sample_a_id?: string;
    sample_b_id?: string;
    selected_layer_id?: string;
  }): Promise<any> {
    const query = new URLSearchParams();
    if (params?.sample_a_id) query.set('sample_a_id', params.sample_a_id);
    if (params?.sample_b_id) query.set('sample_b_id', params.sample_b_id);
    if (params?.selected_layer_id) query.set('selected_layer_id', params.selected_layer_id);

    const qs = query.toString();
    const url = `${this.baseUrl}/api/analysis/xray/pathway${qs ? `?${qs}` : ''}`;
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Fetch activation pathway failed: ${res.statusText}`);
    return res.json();
  }

  // ==========================================
  // What-If / Perturbation Lab Endpoints
  // ==========================================

  async getWhatIfSamples(): Promise<any[]> {
    const res = await fetch(`${this.baseUrl}/api/analysis/whatif/samples`);
    if (!res.ok) throw new Error(`Fetch What-If samples failed: ${res.statusText}`);
    return res.json();
  }

  async runWhatIfExperiment(params: {
    sample_id?: string;
    rotation?: number;
    noise?: number;
    blur?: number;
    crop?: number;
    brightness?: number;
    occlusion?: number;
    include_images?: boolean;
  }): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/analysis/whatif/experiment`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`What-If experiment failed: ${res.statusText}`);
    return res.json();
  }

  // ==========================================
  // Gemma 4 AI Researcher Endpoints
  // DETECT → HYPOTHESIZE → EXPERIMENT → OBSERVE
  // ==========================================

  async getResearcherEvidence(clusterId?: string): Promise<any> {
    const qs = clusterId ? `?cluster_id=${encodeURIComponent(clusterId)}` : '';
    const res = await fetch(`${this.baseUrl}/api/researcher/evidence${qs}`);
    if (!res.ok) throw new Error(`Fetch researcher evidence failed: ${res.statusText}`);
    return res.json();
  }

  async getGemmaHypothesis(params?: { cluster_id?: string; evidence?: any }): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/researcher/hypothesize`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params || {}),
    });
    if (!res.ok) throw new Error(`Gemma hypothesizing failed: ${res.statusText}`);
    return res.json();
  }

  async runResearcherExperiment(params: {
    cluster_id?: string;
    experiment_protocol?: string;
    perturbation_type?: string;
    intensity?: number;
    apply_mitigation?: boolean;
  }): Promise<any> {
    const res = await fetch(`${this.baseUrl}/api/researcher/experiment/run`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Researcher experiment execution failed: ${res.statusText}`);
    return res.json();
  }

  // ==========================================
  // Unified Compute & Experiment Execution Layer
  // (Local & DigitalOcean Providers)
  // ==========================================

  async getComputeInfo(): Promise<ComputeProviderInfo> {
    const res = await fetch(`${this.baseUrl}/api/compute/info`);
    if (!res.ok) throw new Error(`Fetch compute info failed: ${res.statusText}`);
    return res.json();
  }

  async submitExperiment(params: {
    type?: string;
    parameters?: Record<string, any>;
    provider?: string;
  }): Promise<Experiment> {
    const res = await fetch(`${this.baseUrl}/api/experiments`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(params),
    });
    if (!res.ok) throw new Error(`Experiment submission failed: ${res.statusText}`);
    return res.json();
  }

  async getExperiment(experimentId: string): Promise<Experiment> {
    const res = await fetch(`${this.baseUrl}/api/experiments/${encodeURIComponent(experimentId)}`);
    if (!res.ok) throw new Error(`Fetch experiment failed: ${res.statusText}`);
    return res.json();
  }

  async listExperiments(): Promise<Experiment[]> {
    const res = await fetch(`${this.baseUrl}/api/experiments`);
    if (!res.ok) throw new Error(`List experiments failed: ${res.statusText}`);
    return res.json();
  }

  async cancelExperiment(experimentId: string): Promise<{ success: boolean; message: string; experiment_id: string }> {
    const res = await fetch(`${this.baseUrl}/api/experiments/${encodeURIComponent(experimentId)}/cancel`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Cancel experiment failed: ${res.statusText}`);
    return res.json();
  }

  async interpretExperiment(experimentId: string): Promise<GemmaInterpretationResponse> {
    const res = await fetch(`${this.baseUrl}/api/experiments/${encodeURIComponent(experimentId)}/interpret`, {
      method: 'POST',
    });
    if (!res.ok) throw new Error(`Gemma interpretation failed: ${res.statusText}`);
    return res.json();
  }
}

export const mlApiClient = new MLApiClient();
