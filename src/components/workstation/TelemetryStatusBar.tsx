import React from 'react';
import { Cpu } from 'lucide-react';
import { ExperimentTelemetry } from '../../types/neuralTrail';

interface TelemetryStatusBarProps {
  telemetry: ExperimentTelemetry;
}

export const TelemetryStatusBar: React.FC<TelemetryStatusBarProps> = ({ telemetry }) => {
  const isRunning = telemetry.status === 'RUNNING';

  return (
    <footer className="h-8 border-t border-white/10 bg-[#0a0114]/95 px-4 flex items-center justify-between font-mono text-[11px] text-white/70 select-none z-30">
      {/* Left: Status & Experiment ID */}
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span
            className={`w-2 h-2 rounded-full ${
              isRunning ? 'bg-[#ffb300] animate-pulse' : 'bg-[#00ff88]'
            }`}
          />
          <span className="font-semibold text-white uppercase text-[10px]">
            {telemetry.status}
          </span>
          <span className="text-white/40">[{telemetry.id}]</span>
        </div>

        {/* Progress % */}
        <div className="flex items-center gap-1.5">
          <span className="text-white/40">Progress:</span>
          <span className="font-bold text-white">{telemetry.progress}%</span>
        </div>

        {/* Samples Processed */}
        <div className="hidden sm:flex items-center gap-1.5">
          <span className="text-white/40">Samples:</span>
          <span className="text-white">
            {telemetry.samplesProcessed.toLocaleString()} / {telemetry.totalSamples.toLocaleString()}
          </span>
        </div>
      </div>

      {/* Right: GPU / System Status */}
      <div className="flex items-center gap-4 text-[10px]">
        <div className="flex items-center gap-1.5 text-white/60">
          <Cpu className="w-3 h-3 text-white/40" />
          <span>GPU: {telemetry.gpuLoad} ({telemetry.gpuTemp})</span>
        </div>

        <div className="text-[#00ff88] font-semibold">
          Accuracy: {telemetry.accuracyDelta}
        </div>
      </div>
    </footer>
  );
};
