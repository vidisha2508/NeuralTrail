import React from 'react';
import { ModelMetrics } from '../../types/neuralTrail';
import { AlertCircle, TrendingDown, Info } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts';

interface ModelHealthProps {
  metrics: ModelMetrics;
}

export const ModelHealth: React.FC<ModelHealthProps> = ({ metrics }) => {
  const comp = metrics.overallVsBlindSpotComparison;

  const chartData = [
    {
      category: 'Overall Model',
      accuracy: comp.overallAccuracy,
      color: '#4FD18B',
    },
    {
      category: 'Blind Spot #01',
      accuracy: comp.blindSpotAccuracy,
      color: '#FF5A5F',
    },
    {
      category: 'Blind Spot #03',
      accuracy: 68.4,
      color: '#FF5A5F',
    },
    {
      category: 'Blind Spot #02',
      accuracy: 74.2,
      color: '#F5C451',
    },
  ];

  return (
    <div className="lab-panel rounded-lg border border-border-subtle p-4 flex flex-col justify-between">
      {/* Panel Header */}
      <div>
        <div className="flex items-center justify-between pb-3 border-b border-border-subtle mb-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-accent-cyan shadow-glow-cyan" />
            <h3 className="font-heading font-semibold text-xs tracking-wider text-text-primary uppercase">
              MODEL HEALTH & AGGREGATE DIVERGENCE
            </h3>
          </div>
          <span className="text-[10px] font-mono text-text-dim px-2 py-0.5 rounded bg-bg-card border border-border-subtle">
            Active Model Benchmark
          </span>
        </div>

        {/* Visual Bar Comparison */}
        <div className="space-y-4 my-2">
          {/* Overall Accuracy Bar */}
          <div>
            <div className="flex items-center justify-between text-xs font-mono mb-1">
              <span className="text-text-secondary flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm bg-stable" />
                Overall Benchmark Accuracy
              </span>
              <span className="text-stable font-bold text-sm">{comp.overallAccuracy}%</span>
            </div>
            <div className="h-3 w-full bg-bg-subtle rounded-sm overflow-hidden p-0.5 border border-border-subtle">
              <div
                className="h-full bg-stable rounded-sm shadow-glow-stable transition-all duration-500"
                style={{ width: `${comp.overallAccuracy}%` }}
              />
            </div>
          </div>

          {/* Blind Spot Local Accuracy Bar */}
          <div>
            <div className="flex items-center justify-between text-xs font-mono mb-1">
              <span className="text-text-secondary flex items-center gap-1.5">
                <span className="w-2 h-2 rounded-sm bg-failure" />
                Blind Spot Local Accuracy (#01 Rotated)
              </span>
              <span className="text-failure font-bold text-sm glow-text-red">
                {comp.blindSpotAccuracy}%
              </span>
            </div>
            <div className="h-3 w-full bg-bg-subtle rounded-sm overflow-hidden p-0.5 border border-border-subtle">
              <div
                className="h-full bg-failure rounded-sm shadow-glow-failure transition-all duration-500"
                style={{ width: `${comp.blindSpotAccuracy}%` }}
              />
            </div>
          </div>

          {/* Mini Delta Indicator */}
          <div className="flex items-center justify-between px-3 py-2 rounded bg-failure-dark/20 border border-failure/30 text-xs font-mono">
            <span className="text-text-secondary flex items-center gap-1">
              <TrendingDown className="w-3.5 h-3.5 text-failure" />
              Local Performance Collapse:
            </span>
            <span className="text-failure font-bold">
              -{(comp.overallAccuracy - comp.blindSpotAccuracy).toFixed(1)}% Accuracy Drop
            </span>
          </div>
        </div>
      </div>

      {/* Required Annotation */}
      <div className="mt-4 pt-3 border-t border-border-subtle/80 flex items-start gap-2.5 bg-bg-card/50 p-2.5 rounded">
        <Info className="w-4 h-4 text-accent-cyan flex-shrink-0 mt-0.5" />
        <div className="text-xs font-mono leading-relaxed">
          <span className="text-accent-cyan font-semibold block mb-0.5">Key Insight</span>
          <span className="text-text-primary italic">
            "{comp.annotation}"
          </span>
          <p className="text-[11px] text-text-dim mt-1 font-sans">
            While high-level macro validation yields 98.2%, specific manifold subspaces exhibit acute
            catastrophic breakdown when faced with slight perturbation.
          </p>
        </div>
      </div>
    </div>
  );
};
