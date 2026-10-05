import React, { useState } from 'react';
import { 
  GitCompare, 
  CheckCircle2, 
  Code2, 
  ShieldCheck, 
  ArrowRight,
  Download,
  AlertCircle,
  FlaskConical
} from 'lucide-react';
import { Experiment, GemmaInterpretationResponse } from '../../types/neuralTrail';
import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';

interface BeforeAfterViewProps {
  experiment?: Experiment | null;
  interpretation?: GemmaInterpretationResponse | null;
  metrics?: any | null;
  blindSpots?: any[];
  onNavigateToExperiments?: () => void;
}

export const BeforeAfterView: React.FC<BeforeAfterViewProps> = ({
  experiment,
  interpretation,
  metrics,
  blindSpots = [],
  onNavigateToExperiments,
}) => {
  const results = experiment?.results;
  const isCompleted = experiment?.status === 'COMPLETED';

  if (!experiment || !isCompleted || !results || results.observed_accuracy === undefined) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center font-mono">
        <div className="w-14 h-14 rounded-xl bg-white/[0.03] border border-white/10 flex items-center justify-center mb-4">
          <FlaskConical className="w-7 h-7 text-[#00f0ff]" />
        </div>
        <h3 className="font-display font-bold text-xl text-white">NO INTERVENTION EXECUTED</h3>
        <p className="text-xs text-white/50 max-w-md mt-2 font-sans leading-relaxed">
          Neural Trail strictly measures real intervention deltas. No simulated improvement claims are shown.
          Execute a remediation trial or test-time regularization experiment to evaluate post-mitigation performance.
        </p>
        <div className="mt-5 p-3 rounded bg-white/[0.02] border border-white/10 text-xs text-white/70 max-w-sm">
          <span className="text-[#ffb300] font-bold block mb-1">WORKFLOW SEQUENCE:</span>
          FIND (01) → TRACE (02) → STRESS (03) → GEMMA (04) → EXPERIMENT (05) → IMPROVE (06)
        </div>
      </div>
    );
  }

  const baselineAcc = (results.baseline_accuracy ?? 0) * 100;
  const improvedAcc = (results.observed_accuracy ?? 0) * 100;
  const baselineFlipRate = (results.baseline_flip_rate ?? 0) * 100;
  const improvedFlipRate = (results.post_flip_rate ?? 0) * 100;
  const repairedCount = results.repaired_samples_count ?? 0;
  const totalSamples = results.total_samples_evaluated ?? 0;
  const targetCluster = results.cluster_id || 'Cluster';

  const baselineMetrics = {
    accuracy: Number(baselineAcc.toFixed(1)),
    robustnessScore: Number(Math.max(0, 100 - baselineFlipRate).toFixed(1)),
    failureRate: Number(baselineFlipRate.toFixed(1)),
  };

  const improvedMetrics = {
    accuracy: Number(improvedAcc.toFixed(1)),
    robustnessScore: Number(Math.max(0, 100 - improvedFlipRate).toFixed(1)),
    failureRate: Number(improvedFlipRate.toFixed(1)),
  };

  const suggestedPatchCode = interpretation?.suggested_patch || 
    "transforms.Compose([\n  transforms.RandomRotation(degrees=(-20, 20)),\n  transforms.ColorJitter(brightness=0.1),\n  transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])\n])";

  const comparisonRows = [
    {
      metric: `Target Cluster [${targetCluster}] Accuracy`,
      baseline: `${baselineMetrics.accuracy}%`,
      improved: `${improvedMetrics.accuracy}%`,
      delta: `${(improvedMetrics.accuracy - baselineMetrics.accuracy) >= 0 ? '+' : ''}${(improvedMetrics.accuracy - baselineMetrics.accuracy).toFixed(1)}%`,
      isPositive: (improvedMetrics.accuracy - baselineMetrics.accuracy) >= 0,
    },
    {
      metric: 'Prediction Flip Susceptibility',
      baseline: `${baselineMetrics.failureRate}%`,
      improved: `${improvedMetrics.failureRate}%`,
      delta: `${(improvedMetrics.failureRate - baselineMetrics.failureRate) <= 0 ? '' : '+'}${(improvedMetrics.failureRate - baselineMetrics.failureRate).toFixed(1)}%`,
      isPositive: (improvedMetrics.failureRate - baselineMetrics.failureRate) <= 0,
    },
    {
      metric: 'Robustness Resilience Index',
      baseline: `${baselineMetrics.robustnessScore} / 100`,
      improved: `${improvedMetrics.robustnessScore} / 100`,
      delta: `${(improvedMetrics.robustnessScore - baselineMetrics.robustnessScore) >= 0 ? '+' : ''}${(improvedMetrics.robustnessScore - baselineMetrics.robustnessScore).toFixed(1)}`,
      isPositive: (improvedMetrics.robustnessScore - baselineMetrics.robustnessScore) >= 0,
    },
  ];

  return (
    <div className="flex-1 flex flex-col overflow-y-auto px-6 py-6 md:px-8 space-y-6">
      <PageHeader
        stepNumber="06"
        stepCode="IMPROVE"
        title="Intervention Verification & Before/After Comparison"
        description="Verify empirical recovery across target failure clusters following executed model interventions."
        badge={
          <span className="font-mono text-xs text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
            EXPERIMENT EMPIRICALLY CONFIRMED
          </span>
        }
      />

      {/* Summary Highlight Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card className="bg-white/[0.02] border-white/10">
          <div className="font-mono text-[10px] text-white/40 uppercase">TARGET CLUSTER</div>
          <div className="font-display font-bold text-lg text-white mt-1">{targetCluster}</div>
          <div className="text-[11px] text-white/50 mt-1 font-sans">{results.protocol}</div>
        </Card>

        <Card className="bg-white/[0.02] border-white/10">
          <div className="font-mono text-[10px] text-white/40 uppercase">BASELINE ACCURACY</div>
          <div className="font-mono font-bold text-2xl text-white/70 mt-1">{baselineMetrics.accuracy}%</div>
          <div className="text-[11px] text-white/40 mt-1">Pre-intervention failure state</div>
        </Card>

        <Card className="bg-white/[0.02] border-[#00ff88]/30 bg-[#00ff88]/[0.02]">
          <div className="font-mono text-[10px] text-[#00ff88] uppercase">POST-MITIGATION ACCURACY</div>
          <div className="font-mono font-bold text-2xl text-[#00ff88] mt-1">{improvedMetrics.accuracy}%</div>
          <div className="text-[11px] text-[#00ff88] mt-1 font-semibold">
            {results.accuracy_delta} Empirical Recovery
          </div>
        </Card>

        <Card className="bg-white/[0.02] border-white/10">
          <div className="font-mono text-[10px] text-white/40 uppercase">REPAIRED SAMPLES</div>
          <div className="font-mono font-bold text-2xl text-[#00f0ff] mt-1">
            {repairedCount} <span className="text-xs text-white/40 font-normal">/ {totalSamples}</span>
          </div>
          <div className="text-[11px] text-white/50 mt-1">Recovered to target attractor</div>
        </Card>
      </div>

      {/* Side-by-Side Comparison Table */}
      <Card
        headerTitle="EMPIRICAL BEFORE / AFTER METRIC DELTAS"
        headerSubtitle={`Trial ID: ${experiment.id}`}
        headerIcon={<GitCompare className="w-4 h-4 text-[#00f0ff]" />}
        className="bg-black/50 border-white/10"
      >
        <div className="overflow-x-auto">
          <table className="w-full text-left font-mono text-xs">
            <thead>
              <tr className="border-b border-white/10 text-white/40">
                <th className="pb-3 font-medium">METRIC</th>
                <th className="pb-3 font-medium">BEFORE INTERVENTION</th>
                <th className="pb-3 font-medium">AFTER INTERVENTION</th>
                <th className="pb-3 font-medium">MEASURED DELTA</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {comparisonRows.map((row, idx) => (
                <tr key={idx} className="hover:bg-white/[0.02] transition-colors">
                  <td className="py-3.5 font-sans font-medium text-white">{row.metric}</td>
                  <td className="py-3.5 text-white/60">{row.baseline}</td>
                  <td className="py-3.5 font-bold text-white">{row.improved}</td>
                  <td className={`py-3.5 font-bold ${row.isPositive ? 'text-[#00ff88]' : 'text-[#ff007f]'}`}>
                    {row.delta}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </Card>

      {/* Suggested Mitigation Remediation Code */}
      <Card
        headerTitle="RECOMMENDED SURGICAL MITIGATION TRANSFORM"
        headerSubtitle="Python / PyTorch transforms pipeline"
        headerIcon={<Code2 className="w-4 h-4 text-[#d500f9]" />}
        className="bg-black/50 border-white/10"
      >
        <pre className="p-4 rounded bg-[#0a0214] border border-white/10 font-mono text-xs text-[#00ff88] overflow-x-auto">
          {suggestedPatchCode}
        </pre>
      </Card>
    </div>
  );
};
