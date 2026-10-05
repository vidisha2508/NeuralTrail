import React from 'react';
import { BlindSpot } from '../../types/neuralTrail';
import { StatusBadge } from '../common/StatusBadge';
import { ChevronRight, AlertOctagon, RotateCw, Moon, EyeOff } from 'lucide-react';

interface FailureRegionsProps {
  blindSpots: BlindSpot[];
  selectedClusterId: string | null;
  onSelectCluster: (clusterId: string) => void;
}

export const FailureRegions: React.FC<FailureRegionsProps> = ({
  blindSpots,
  selectedClusterId,
  onSelectCluster,
}) => {
  const getIcon = (id: string) => {
    switch (id) {
      case 'bs-01':
        return <RotateCw className="w-4 h-4 text-failure" />;
      case 'bs-02':
        return <Moon className="w-4 h-4 text-uncertain" />;
      case 'bs-03':
        return <EyeOff className="w-4 h-4 text-failure" />;
      default:
        return <AlertOctagon className="w-4 h-4 text-failure" />;
    }
  };

  return (
    <div className="lab-panel rounded-lg border border-border-subtle p-4">
      {/* Panel Header */}
      <div className="flex items-center justify-between pb-3 border-b border-border-subtle mb-3">
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-failure shadow-glow-failure" />
          <h3 className="font-heading font-semibold text-xs tracking-wider text-text-primary uppercase">
            DETECTED FAILURE REGIONS
          </h3>
        </div>
        <span className="text-[11px] font-mono text-text-dim">
          Click row to inspect cluster
        </span>
      </div>

      {/* Rows */}
      <div className="space-y-2">
        {blindSpots.map((spot) => {
          const isSelected = selectedClusterId === spot.id;

          return (
            <div
              key={spot.id}
              onClick={() => onSelectCluster(spot.id)}
              className={`p-3 rounded border transition-all duration-200 cursor-pointer flex flex-wrap md:flex-nowrap items-center justify-between gap-3 ${
                isSelected
                  ? 'bg-failure-dark/30 border-failure/60 shadow-glow-failure scale-[1.008]'
                  : 'bg-bg-card/70 border-border-subtle hover:border-border-active hover:bg-bg-subtle'
              }`}
            >
              {/* Left Identifier & Title */}
              <div className="flex items-center gap-3 min-w-[220px]">
                <div className="p-2 rounded bg-bg-panel border border-border-subtle">
                  {getIcon(spot.id)}
                </div>
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-bold text-accent-cyan">
                      {spot.clusterNumber}
                    </span>
                    <span className="font-heading font-semibold text-sm text-text-primary">
                      {spot.name}
                    </span>
                  </div>
                  <div className="text-[11px] font-mono text-text-dim">
                    Trigger: <span className="text-text-secondary">{spot.sensitivity}</span>
                  </div>
                </div>
              </div>

              {/* Severity Badge */}
              <div className="flex-shrink-0">
                <StatusBadge status={spot.severity} size="sm" />
              </div>

              {/* Accuracy Readout */}
              <div className="text-left md:text-right min-w-[100px]">
                <div className="text-[10px] font-mono text-text-dim uppercase">Accuracy</div>
                <div
                  className={`text-sm font-heading font-bold ${
                    spot.accuracy < 65 ? 'text-failure glow-text-red' : 'text-uncertain'
                  }`}
                >
                  {spot.accuracy}%
                </div>
              </div>

              {/* Sample Count */}
              <div className="text-left md:text-right min-w-[120px]">
                <div className="text-[10px] font-mono text-text-dim uppercase">Samples</div>
                <div className="text-sm font-mono text-text-secondary font-medium">
                  {spot.sampleCount.toLocaleString()} samples
                </div>
              </div>

              {/* Action arrow */}
              <div className="hidden md:flex text-text-dim group-hover:text-text-primary">
                <ChevronRight
                  className={`w-4 h-4 transition-transform ${
                    isSelected ? 'translate-x-1 text-accent-cyan' : ''
                  }`}
                />
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
