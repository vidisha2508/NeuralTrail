import { BlindSpot, FilterState, ModelMetrics, Sample } from '../types/neuralTrail';
import { mlApiClient } from './mlApiClient';

/**
 * Service Layer for Neural Trail Analysis
 * Connected strictly to the live PyTorch Backend.
 * NO MOCK DATA. Never falls back to mock or simulated metrics.
 */
class AnalysisService {
  private mapBackendBlindSpot(b: any): BlindSpot {
    return {
      id: b.id,
      clusterNumber: b.cluster_number || b.clusterNumber || '#01',
      name: b.name,
      severity: (b.severity as any) || 'LOW',
      sampleCount: b.sample_count ?? b.sampleCount ?? 0,
      accuracy: b.accuracy ?? 0,
      confidence: b.average_confidence ?? b.confidence ?? 0.0,
      commonConfusion: b.common_prediction_error || b.commonConfusion || 'None',
      sensitivity: b.sensitivity || 'General',
      clusterCenter: b.cluster_center || b.clusterCenter || { x: 0, y: 0 },
      radius: b.radius || 15.0,
      description: b.description || '',
      mitigationSuggestion: b.mitigation_suggestion || b.mitigationSuggestion || '',
    };
  }

  private mapBackendSample(s: any): Sample {
    return {
      id: s.id,
      x: s.x,
      y: s.y,
      trueClass: s.trueClass ?? s.true_class_idx ?? 0,
      predictedClass: s.predictedClass ?? s.predicted_class_idx ?? 0,
      confidence: s.confidence ?? 0.0,
      correct: Boolean(s.correct),
      stability: s.stability || (s.correct ? 'stable' : 'failure'),
      clusterId: s.clusterId,
      sensitivity: s.sensitivity || 'General',
      thumbnailGrid: s.thumbnailGrid,
      features: s.features || { rotation: 0, brightness: 1, blur: 0 },
    };
  }

  async getModelMetrics(): Promise<ModelMetrics | null> {
    try {
      const summary = await mlApiClient.getAnalysisSummary();
      if (summary && summary.has_data && summary.metrics) {
        return summary.metrics;
      }
    } catch {
      // Honest state: no backend data
    }
    return null;
  }

  async getBlindSpots(): Promise<BlindSpot[]> {
    try {
      const rawSpots = await mlApiClient.getBlindSpots();
      if (Array.isArray(rawSpots) && rawSpots.length > 0) {
        return rawSpots.map((b) => this.mapBackendBlindSpot(b));
      }
    } catch {
      // Honest state
    }
    return [];
  }

  async getBlindSpotById(id: string): Promise<BlindSpot | null> {
    try {
      const rawDetail = await mlApiClient.getBlindSpotDetail(id);
      if (rawDetail) {
        return this.mapBackendBlindSpot(rawDetail);
      }
    } catch {
      // Honest state
    }
    return null;
  }

  async getSamples(filter?: Partial<FilterState>): Promise<Sample[]> {
    try {
      const resp = await mlApiClient.getAnalysisSamples({
        cluster_id: filter?.selectedClusterId || undefined,
        status: filter?.status !== 'all' ? filter?.status : undefined,
        min_confidence: filter?.minConfidence,
        class_id: filter?.selectedClass && filter.selectedClass !== 'all' ? Number(filter.selectedClass) : undefined,
        limit: 500,
      });

      if (resp && Array.isArray(resp.samples)) {
        let list = resp.samples.map((s) => this.mapBackendSample(s));

        if (filter?.searchQuery && filter.searchQuery.trim().length > 0) {
          const q = filter.searchQuery.trim().toLowerCase();
          list = list.filter((s) =>
            s.id.toLowerCase().includes(q) ||
            String(s.trueClass).includes(q) ||
            String(s.predictedClass).includes(q) ||
            s.sensitivity.toLowerCase().includes(q)
          );
        }
        return list;
      }
    } catch {
      // Honest state: return empty array if no model is loaded
    }
    return [];
  }

  async getSamplesByCluster(clusterId: string, limit = 8): Promise<Sample[]> {
    const clusterSamples = await this.getSamples({ selectedClusterId: clusterId });
    return clusterSamples.slice(0, limit);
  }

  async triggerAnalysis(sampleCount = 100, reducer = 'pca'): Promise<any> {
    return mlApiClient.startAnalysis({
      dataset_source: 'benchmark',
      sample_count: sampleCount,
      reducer_type: reducer,
      async_run: false,
    });
  }
}

export const analysisService = new AnalysisService();
