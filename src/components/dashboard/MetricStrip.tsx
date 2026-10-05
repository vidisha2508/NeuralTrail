import React from 'react';
import { ModelMetrics } from '../../types/neuralTrail';
import { Metric } from '../common/Metric';
import { StatusBadge } from '../common/StatusBadge';
import { Activity, Target, ShieldCheck, AlertOctagon } from 'lucide-react';

interface MetricStripProps {
  metrics: ModelMetrics;
}

export const MetricStrip: React.FC<MetricStripProps> = ({ metrics }) => {
  return (
    <div className="lab-panel rounded-lg py-2.5 px-4 mb-4 border border-border-subtle bg-bg-panel/90 backdrop-blur-sm">
      <div className="grid grid-cols-2 md:grid-cols-4 gap-2 divide-y md:divide-y-0 md:divide-x divide-border-subtle/70">
        {/* Metric 1: Model Health */}
        <Metric
          label="Model Health"
          value={`${metrics.health}%`}
          badge={<StatusBadge status="stable" size="sm" pulse />}
          accent="green"
          icon={<ShieldCheck className="w-3.5 h-3.5 text-stable" />}
        />

        {/* Metric 2: Accuracy */}
        <Metric
          label="Accuracy"
          value={`${metrics.accuracy}%`}
          subValue={metrics.accuracyDelta}
          trend="up"
          accent="cyan"
          icon={<Target className="w-3.5 h-3.5 text-accent-cyan" />}
        />

        {/* Metric 3: Avg Confidence */}
        <Metric
          label="Avg Confidence"
          value={`${metrics.avgConfidence}%`}
          accent="cyan"
          icon={<Activity className="w-3.5 h-3.5 text-accent-cyan" />}
        />

        {/* Metric 4: Blind Spots */}
        <Metric
          label="Blind Spots"
          value={String(metrics.blindSpotCount).padStart(2, '0')}
          subValue={`(${metrics.highSeverityCount} high severity)`}
          accent="red"
          icon={<AlertOctagon className="w-3.5 h-3.5 text-failure" />}
        />
      </div>
    </div>
  );
};
