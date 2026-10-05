import React from 'react';
import { GitCompare } from 'lucide-react';
import { LabPlaceholder } from '../components/common/LabPlaceholder';

export const ModelComparison: React.FC = () => {
  return (
    <LabPlaceholder
      moduleName="Model Comparison"
      category="Analysis"
      description="Side-by-side behavioral topology diffing between candidate checkpoints (e.g. Baseline Checkpoint vs Pruned Candidate vs Quantized Core)."
      icon={GitCompare}
      estimatedFeatures={[
        'Dual-manifold alignment and Procrustes analysis',
        'Failure cluster intersection and unique error attribution',
        'Inference latency vs blind spot count Pareto frontiers',
        'Checkpoint regression diff reporting',
      ]}
    />
  );
};
