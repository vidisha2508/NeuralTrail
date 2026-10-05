import { ModelMetrics } from '../types/neuralTrail';

export const mockModelMetrics: ModelMetrics = {
  name: 'Active Vision Classifier',
  architecture: 'Multi-Stage Deep Residual Network',
  dataset: 'High-Dimensional Evaluation Manifold',
  health: 98.2,
  healthStatus: 'Stable',
  accuracy: 98.2,
  accuracyDelta: '+0.4%',
  avgConfidence: 91.4,
  blindSpotCount: 3,
  highSeverityCount: 2,
  totalSamples: 1842,
  overallVsBlindSpotComparison: {
    overallAccuracy: 98.2,
    blindSpotAccuracy: 61.8,
    annotation: 'Aggregate accuracy hides localized failure.'
  }
};
