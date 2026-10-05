import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { BlindSpot, FilterState, Sample } from '../../types/neuralTrail';
import { 
  Search, 
  ZoomIn, 
  ZoomOut, 
  RotateCcw, 
  SlidersHorizontal, 
  Compass,
  ArrowRight
} from 'lucide-react';

import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';
import { MetricCard } from '../common/MetricCard';

interface BlindSpotMapViewProps {
  samples: Sample[];
  blindSpots: BlindSpot[];
  metrics?: any | null;
  selectedClusterId: string | null;
  onSelectCluster: (clusterId: string | null) => void;
  onSelectSample: (sample: Sample | null) => void;
  selectedSample: Sample | null;
  filterState: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
  onSwitchToTab: (tab: any) => void;
}

export const BlindSpotMapView: React.FC<BlindSpotMapViewProps> = ({
  samples,
  blindSpots,
  metrics,
  selectedClusterId,
  onSelectCluster,
  onSelectSample,
  selectedSample,
  filterState,
  onFilterChange,
  onSwitchToTab,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

  // Dynamic calculations from actual application data (no hardcoded diagnostic values)
  const totalCount = samples.length;
  const correctCount = samples.filter((s) => s.correct).length;
  const computedAccuracy = totalCount > 0 ? (correctCount / totalCount) * 100 : 0;
  const computedConfidence = totalCount > 0 ? samples.reduce((acc, s) => acc + s.confidence, 0) / totalCount : 0;
  
  const displayAccuracy = metrics?.accuracy !== undefined ? metrics.accuracy : Number(computedAccuracy.toFixed(1));
  const accuracyDelta = metrics?.accuracyDelta || '+0.0%';
  const displayConfidence = metrics?.avgConfidence !== undefined ? metrics.avgConfidence : Number(computedConfidence.toFixed(1));
  const displayHealth = metrics?.health !== undefined ? metrics.health : Number(computedAccuracy.toFixed(1));
  const displayHealthStatus = metrics?.healthStatus || (displayHealth >= 90 ? 'Stable' : displayHealth >= 70 ? 'Degraded' : 'Critical');
  
  const blindSpotCount = metrics?.blindSpotCount !== undefined ? metrics.blindSpotCount : blindSpots.length;
  const highSeverityCount = metrics?.highSeverityCount !== undefined ? metrics.highSeverityCount : blindSpots.filter((b) => b.severity === 'HIGH').length;

  const [tooltip, setTooltip] = useState<{
    visible: boolean;
    x: number;
    y: number;
    sample: Sample | null;
  }>({
    visible: false,
    x: 0,
    y: 0,
    sample: null,
  });

  const [showFilters, setShowFilters] = useState(false);

  const V_WIDTH = 1000;
  const V_HEIGHT = 650;

  const filteredSamples = useMemo(() => {
    return samples.filter((s) => {
      if (filterState.status !== 'all' && s.stability !== filterState.status) return false;
      if (s.confidence < filterState.minConfidence) return false;
      if (filterState.selectedClass !== 'all') {
        const clsNum = Number(filterState.selectedClass);
        if (s.trueClass !== clsNum && s.predictedClass !== clsNum) return false;
      }
      if (filterState.perturbation !== 'all') {
        if (!s.sensitivity.toLowerCase().includes(filterState.perturbation.toLowerCase())) return false;
      }
      if (filterState.searchQuery.trim()) {
        const q = filterState.searchQuery.toLowerCase();
        if (
          !s.id.toLowerCase().includes(q) &&
          !String(s.trueClass).includes(q) &&
          !String(s.predictedClass).includes(q) &&
          !s.sensitivity.toLowerCase().includes(q)
        ) {
          return false;
        }
      }
      return true;
    });
  }, [samples, filterState]);

  useEffect(() => {
    if (!svgRef.current) return;
    const svg = d3.select(svgRef.current);
    const contentGroup = svg.select<SVGGElement>('.zoomable-content');

    const zoom = d3
      .zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.6, 5])
      .on('zoom', (event) => {
        contentGroup.attr('transform', event.transform);
      });

    zoomBehaviorRef.current = zoom;
    svg.call(zoom);

    return () => {
      svg.on('.zoom', null);
    };
  }, []);

  const handleResetZoom = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(400).call(zoomBehaviorRef.current.transform, d3.zoomIdentity);
  };

  const handleZoomIn = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(200).call(zoomBehaviorRef.current.scaleBy, 1.3);
  };

  const handleZoomOut = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(200).call(zoomBehaviorRef.current.scaleBy, 0.77);
  };

  useEffect(() => {
    if (!selectedClusterId || !svgRef.current || !zoomBehaviorRef.current) return;
    const cluster = blindSpots.find((b) => b.id === selectedClusterId);
    if (!cluster) return;

    const svg = d3.select(svgRef.current);
    const targetScale = 1.6;
    const targetX = V_WIDTH / 2 - cluster.clusterCenter.x * targetScale;
    const targetY = V_HEIGHT / 2 - cluster.clusterCenter.y * targetScale;

    svg
      .transition()
      .duration(600)
      .ease(d3.easeCubicOut)
      .call(
        zoomBehaviorRef.current.transform,
        d3.zoomIdentity.translate(targetX, targetY).scale(targetScale)
      );
  }, [selectedClusterId, blindSpots]);

  return (
    <div className="flex-1 overflow-y-auto">
      <div className="max-w-[1440px] mx-auto w-full px-6 py-8 md:px-8 space-y-6">
        {/* Page Header */}
        <PageHeader
          stepNumber="01"
          stepCode="FIND"
          title="Blind Spot Cartography"
          description="High-dimensional embedding manifold projection isolating failure clusters, decision boundary sensitivity, and rotational susceptibility."
          actions={
            <div className="flex items-center gap-2">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`px-3 py-1.5 rounded-md border text-xs font-sans font-medium flex items-center gap-1.5 transition-colors ${
                  showFilters
                    ? 'bg-white/15 border-white/40 text-white'
                    : 'bg-[#180933] border-white/15 text-white/70 hover:text-white'
                }`}
              >
                <SlidersHorizontal className="w-3.5 h-3.5" />
                <span>Filters</span>
              </button>

              <button
                onClick={handleZoomIn}
                className="p-1.5 rounded-md bg-[#180933] border border-white/15 text-white/70 hover:text-white transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-4 h-4" />
              </button>

              <button
                onClick={handleZoomOut}
                className="p-1.5 rounded-md bg-[#180933] border border-white/15 text-white/70 hover:text-white transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-4 h-4" />
              </button>

              <button
                onClick={handleResetZoom}
                className="p-1.5 rounded-md bg-[#180933] border border-white/15 text-white/70 hover:text-white transition-colors"
                title="Reset View"
              >
                <RotateCcw className="w-4 h-4" />
              </button>
            </div>
          }
        />

        {/* 1. Standardized Metrics Row */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <MetricCard
            label="MODEL HEALTH"
            value={`${displayHealth.toFixed(1)}%`}
            subValue={displayHealthStatus}
            accent={displayHealth >= 90 ? 'green' : displayHealth >= 70 ? 'amber' : 'magenta'}
          />

          <MetricCard
            label="VALIDATION ACCURACY"
            value={`${displayAccuracy.toFixed(1)}%`}
            subValue={accuracyDelta}
            accent="cyan"
          />

          <MetricCard
            label="MEAN CONFIDENCE"
            value={`${displayConfidence.toFixed(1)}%`}
            subValue={`${totalCount.toLocaleString()} samples evaluated`}
            accent="purple"
          />

          <MetricCard
            label="DETECTED BLIND SPOTS"
            value={String(blindSpotCount).padStart(2, '0')}
            subValue={`${highSeverityCount} High Severity`}
            accent="magenta"
          />
        </div>

        {/* 2. Hero Cartography Map */}
        <Card
          headerTitle="Manifold Embedding Projection"
          headerSubtitle={`${filteredSamples.length} of ${totalCount.toLocaleString()} samples active on projection plane`}
          headerIcon={<Compass className="w-4 h-4 text-[#00ff88]" />}
          headerAction={
            <div className="flex items-center gap-1.5 text-xs">
              <button
                onClick={() => setShowFilters(!showFilters)}
                className={`px-2.5 py-1 rounded border text-[11px] font-medium flex items-center gap-1.5 transition-colors ${
                  showFilters
                    ? 'bg-white/15 border-white/40 text-white'
                    : 'bg-white/[0.03] border-white/10 text-white/60 hover:text-white'
                }`}
              >
                <SlidersHorizontal className="w-3 h-3" />
                <span>Filters</span>
              </button>

              <button
                onClick={handleZoomIn}
                className="p-1.5 rounded bg-white/[0.03] border border-white/10 text-white/60 hover:text-white transition-colors"
                title="Zoom In"
              >
                <ZoomIn className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={handleZoomOut}
                className="p-1.5 rounded bg-white/[0.03] border border-white/10 text-white/60 hover:text-white transition-colors"
                title="Zoom Out"
              >
                <ZoomOut className="w-3.5 h-3.5" />
              </button>

              <button
                onClick={handleResetZoom}
                className="p-1.5 rounded bg-white/[0.03] border border-white/10 text-white/60 hover:text-white transition-colors"
                title="Reset View"
              >
                <RotateCcw className="w-3.5 h-3.5" />
              </button>
            </div>
          }
          noPadding
        >

        {/* Optional Filter Controls Bar */}
        {showFilters && (
          <div className="p-3 bg-[#110224] border-b border-white/10 flex flex-wrap items-center gap-3 text-xs">
            <div className="flex items-center gap-1.5 bg-white/[0.03] border border-white/10 rounded px-2.5 py-1">
              <Search className="w-3 h-3 text-white/40" />
              <input
                type="text"
                value={filterState.searchQuery}
                onChange={(e) => onFilterChange({ searchQuery: e.target.value })}
                placeholder="Search class or ID..."
                className="bg-transparent border-none outline-none text-xs text-white placeholder-white/30 w-36"
              />
            </div>

            <div className="flex items-center gap-2">
              <span className="text-white/40 text-[11px]">Stability:</span>
              <select
                value={filterState.status}
                onChange={(e) => onFilterChange({ status: e.target.value as any })}
                className="bg-[#180533] border border-white/15 rounded px-2 py-1 text-xs text-white"
              >
                <option value="all">All</option>
                <option value="stable">Stable</option>
                <option value="uncertain">Uncertain</option>
                <option value="failure">Failure</option>
              </select>
            </div>

            <div className="flex items-center gap-2">
              <span className="text-white/40 text-[11px]">Min Conf:</span>
              <input
                type="range"
                min="0"
                max="100"
                value={filterState.minConfidence}
                onChange={(e) => onFilterChange({ minConfidence: Number(e.target.value) })}
                className="w-20 accent-[#00ff88]"
              />
              <span className="text-white/60 text-[11px] w-6">{filterState.minConfidence}%</span>
            </div>
          </div>
        )}

        {/* Main D3 Cartography Canvas */}
        <div
          ref={containerRef}
          className="relative flex-1 bg-[#0b0118] overflow-hidden select-none"
          style={{ minHeight: '380px' }}
          onClick={() => {
            onSelectSample(null);
          }}
        >
          <svg
            ref={svgRef}
            viewBox={`0 0 ${V_WIDTH} ${V_HEIGHT}`}
            className="w-full h-full cursor-grab active:cursor-grabbing"
          >
            <defs>
              <pattern id="grid-dots" width="40" height="40" patternUnits="userSpaceOnUse">
                <circle cx="2" cy="2" r="1" fill="#7c4dff" fillOpacity="0.12" />
              </pattern>
            </defs>

            {/* Background Grid Pattern */}
            <rect width={V_WIDTH} height={V_HEIGHT} fill="url(#grid-dots)" />

            <g className="zoomable-content">
              {/* Cluster Boundaries (Progressive Disclosure - Section 6) */}
              {blindSpots.map((cluster) => {
                const isSelected = selectedClusterId === cluster.id;
                const isHigh = cluster.severity === 'HIGH';

                return (
                  <g
                    key={cluster.id}
                    className="cursor-pointer transition-opacity duration-200"
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectCluster(cluster.id);
                    }}
                  >
                    {/* Subtle Area Circle */}
                    <circle
                      cx={cluster.clusterCenter.x}
                      cy={cluster.clusterCenter.y}
                      r={cluster.radius}
                      fill={isHigh ? '#ff007f' : '#ffb300'}
                      fillOpacity={isSelected ? 0.12 : 0.04}
                      stroke={isSelected ? '#00ff88' : isHigh ? '#ff007f' : '#ffb300'}
                      strokeWidth={isSelected ? 1.5 : 1}
                      strokeDasharray={isSelected ? undefined : '3 3'}
                    />

                    {/* Minimal Region Label Badge */}
                    <g transform={`translate(${cluster.clusterCenter.x}, ${cluster.clusterCenter.y - cluster.radius - 8})`}>
                      <rect
                        x="-70"
                        y="-10"
                        width="140"
                        height="20"
                        rx="10"
                        fill="#120224"
                        stroke={isSelected ? '#00ff88' : 'rgba(255,255,255,0.15)'}
                        strokeWidth="1"
                      />
                      <text
                        x="0"
                        y="3"
                        textAnchor="middle"
                        fill={isSelected ? '#00ff88' : 'rgba(255,255,255,0.8)'}
                        fontSize="9"
                        fontFamily="monospace"
                        fontWeight="600"
                      >
                        {cluster.clusterNumber} {cluster.name}
                      </text>
                    </g>
                  </g>
                );
              })}

              {/* Scatter Points */}
              <g className="scatter-points">
                {filteredSamples.map((sample) => {
                  const isSelected = selectedSample?.id === sample.id;
                  const inCluster = selectedClusterId ? sample.clusterId === selectedClusterId : true;

                  let fill = '#00ff88';
                  let radius = 2.4;

                  if (sample.stability === 'failure') {
                    fill = '#ff007f';
                    radius = 3.2;
                  } else if (sample.stability === 'uncertain') {
                    fill = '#ffb300';
                    radius = 2.6;
                  }

                  if (isSelected) {
                    fill = '#00f0ff';
                    radius = 6;
                  }

                  let opacity = 0.85;
                  if (selectedClusterId && !inCluster) {
                    opacity = 0.18;
                  }

                  return (
                    <circle
                      key={sample.id}
                      cx={sample.x}
                      cy={sample.y}
                      r={radius}
                      fill={fill}
                      opacity={opacity}
                      className="cursor-pointer transition-all hover:scale-150"
                      onMouseEnter={(e) => {
                        const rect = containerRef.current?.getBoundingClientRect();
                        if (rect) {
                          setTooltip({
                            visible: true,
                            x: e.clientX - rect.left,
                            y: e.clientY - rect.top,
                            sample,
                          });
                        }
                      }}
                      onMouseMove={(e) => {
                        const rect = containerRef.current?.getBoundingClientRect();
                        if (rect) {
                          setTooltip((prev) => ({
                            ...prev,
                            x: e.clientX - rect.left,
                            y: e.clientY - rect.top,
                          }));
                        }
                      }}
                      onMouseLeave={() => setTooltip((prev) => ({ ...prev, visible: false }))}
                      onClick={(e) => {
                        e.stopPropagation();
                        onSelectSample(sample);
                        if (sample.clusterId) onSelectCluster(sample.clusterId);
                      }}
                    />
                  );
                })}
              </g>
            </g>
          </svg>

          {/* Clean Hover Tooltip (Progressive Disclosure) */}
          {tooltip.visible && tooltip.sample && (
            <div
              className="absolute z-30 pointer-events-none p-2.5 rounded bg-[#130324] border border-white/20 text-xs font-mono transform -translate-x-1/2 -translate-y-full mb-3 shadow-lg"
              style={{ left: `${tooltip.x}px`, top: `${tooltip.y}px` }}
            >
              <div className="flex items-center justify-between gap-3 border-b border-white/10 pb-1 mb-1 font-bold">
                <span className="text-[#00f0ff]">#{tooltip.sample.id}</span>
                <span className={`text-[9px] uppercase ${
                  tooltip.sample.stability === 'failure'
                    ? 'text-[#ff007f]'
                    : tooltip.sample.stability === 'uncertain'
                    ? 'text-[#ffb300]'
                    : 'text-[#00ff88]'
                }`}>
                  {tooltip.sample.stability}
                </span>
              </div>
              <div className="text-[11px] text-white/80 space-y-0.5">
                <div>Class {tooltip.sample.trueClass} → Class {tooltip.sample.predictedClass}</div>
                <div>Confidence: <strong className="text-white">{tooltip.sample.confidence}%</strong></div>
              </div>
            </div>
          )}
        </div>

        {/* Legend Bar: Quiet & Minimal */}
        <div className="border-t border-white/10 px-4 py-2 bg-[#100222] flex items-center justify-between text-xs text-white/60">
          <div className="flex items-center gap-4">
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#00ff88]" />
              <span>Stable</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#ffb300]" />
              <span>Uncertain</span>
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2 h-2 rounded-full bg-[#ff007f]" />
              <span>Failure</span>
            </span>
          </div>

          <div className="text-[11px] text-white/40">
            Click cluster or node to inspect details in Context Panel
          </div>
        </div>
      </Card>

      {/* 3. Failure Clusters — Compact Diagnostic Table */}
      <Card
        headerTitle="Detected Failure Regions"
        headerSubtitle="Select row to focus visualization"
        noPadding
      >
        <div className="divide-y divide-white/5 font-mono text-xs">
          {blindSpots.map((spot) => {
            const isSelected = selectedClusterId === spot.id;
            const isHigh = spot.severity === 'HIGH';

            return (
              <div
                key={spot.id}
                onClick={() => onSelectCluster(spot.id)}
                className={`py-2.5 px-4 flex items-center justify-between cursor-pointer transition-colors ${
                  isSelected
                    ? 'bg-[#ff007f]/10 border-l-2 border-l-[#ff007f] text-white'
                    : 'hover:bg-white/[0.03] text-white/80'
                }`}
              >
                {/* Severity & Confusion Error */}
                <div className="flex items-center gap-3 min-w-[260px]">
                  <span className={`text-[9px] px-1.5 py-0.5 rounded font-bold uppercase tracking-wider ${
                    isHigh
                      ? 'bg-[#ff007f]/20 text-[#ff007f] border border-[#ff007f]/30'
                      : 'bg-[#ffb300]/20 text-[#ffb300] border border-[#ffb300]/30'
                  }`}>
                    {spot.severity}
                  </span>

                  <div>
                    <span className="font-semibold text-white">
                      {spot.commonConfusion}
                    </span>
                    <span className="text-white/40 text-[11px] ml-2">
                      ({spot.name})
                    </span>
                  </div>
                </div>

                {/* Accuracy */}
                <div className="text-right w-24">
                  <span className="text-[#ff007f] font-bold">
                    {spot.accuracy}%
                  </span>
                  <span className="text-white/40 text-[10px] ml-1">acc</span>
                </div>

                {/* Samples Count */}
                <div className="text-right w-28 text-white/60 text-[11px]">
                  {spot.sampleCount} samples
                </div>

                {/* Trigger */}
                <div className="text-right w-36 text-white/50 text-[11px] truncate">
                  {spot.sensitivity}
                </div>

                {/* Inspect Action */}
                <div className="w-16 text-right">
                  <span className="text-[11px] text-[#00ff88] flex items-center justify-end gap-1 opacity-80 hover:opacity-100">
                    <span>Inspect</span>
                    <ArrowRight className="w-3 h-3" />
                  </span>
                </div>
              </div>
            );
          })}
        </div>
      </Card>
    </div>
  </div>
);
};
