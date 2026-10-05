import React, { useEffect, useRef, useState, useMemo } from 'react';
import * as d3 from 'd3';
import { BlindSpot, FilterState, Sample } from '../../types/neuralTrail';
import { Search, ZoomIn, ZoomOut, RotateCcw, Filter, Eye, SlidersHorizontal } from 'lucide-react';

interface BlindSpotMapProps {
  samples: Sample[];
  blindSpots: BlindSpot[];
  selectedClusterId: string | null;
  selectedSampleId: string | null;
  onSelectCluster: (clusterId: string | null) => void;
  onSelectSample: (sample: Sample | null) => void;
  filterState: FilterState;
  onFilterChange: (filters: Partial<FilterState>) => void;
}

export const BlindSpotMap: React.FC<BlindSpotMapProps> = ({
  samples,
  blindSpots,
  selectedClusterId,
  selectedSampleId,
  onSelectCluster,
  onSelectSample,
  filterState,
  onFilterChange,
}) => {
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);
  const zoomBehaviorRef = useRef<d3.ZoomBehavior<SVGSVGElement, unknown> | null>(null);

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

  // SVG Virtual Canvas dimensions
  const V_WIDTH = 1000;
  const V_HEIGHT = 860;

  // Filter samples based on props
  const filteredSamples = useMemo(() => {
    return samples.filter((s) => {
      if (filterState.status !== 'all' && s.stability !== filterState.status) {
        return false;
      }
      if (s.confidence < filterState.minConfidence) {
        return false;
      }
      if (filterState.selectedClass !== 'all') {
        const clsNum = Number(filterState.selectedClass);
        if (s.trueClass !== clsNum && s.predictedClass !== clsNum) {
          return false;
        }
      }
      if (filterState.perturbation !== 'all') {
        if (!s.sensitivity.toLowerCase().includes(filterState.perturbation.toLowerCase())) {
          return false;
        }
      }
      if (filterState.searchQuery.trim()) {
        const q = filterState.searchQuery.toLowerCase();
        const match =
          s.id.toLowerCase().includes(q) ||
          String(s.trueClass).includes(q) ||
          String(s.predictedClass).includes(q) ||
          s.sensitivity.toLowerCase().includes(q);
        if (!match) return false;
      }
      return true;
    });
  }, [samples, filterState]);

  // Setup D3 Zoom and SVG
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
    svg.transition().duration(250).call(zoomBehaviorRef.current.scaleBy, 1.3);
  };

  const handleZoomOut = () => {
    if (!svgRef.current || !zoomBehaviorRef.current) return;
    const svg = d3.select(svgRef.current);
    svg.transition().duration(250).call(zoomBehaviorRef.current.scaleBy, 0.77);
  };

  // Center on a cluster when selected
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
      .call(
        zoomBehaviorRef.current.transform,
        d3.zoomIdentity.translate(targetX, targetY).scale(targetScale)
      );
  }, [selectedClusterId, blindSpots]);

  return (
    <div className="lab-panel rounded-lg border border-border-subtle overflow-hidden flex flex-col mb-4">
      {/* Header & Controls Toolbar */}
      <div className="px-4 py-3 border-b border-border-subtle bg-bg-panel flex flex-wrap items-center justify-between gap-3">
        {/* Title & Stats */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-accent-cyan shadow-glow-cyan" />
            <h2 className="font-heading font-semibold text-sm tracking-wider text-text-primary uppercase">
              MODEL BEHAVIOR MAP
            </h2>
          </div>
          <span className="text-border-active">|</span>
          <div className="text-xs font-mono text-text-secondary flex items-center gap-3">
            <span>
              Showing <strong className="text-accent-cyan">{filteredSamples.length}</strong> / 1,842 points
            </span>
          </div>
        </div>

        {/* Right Tools & Filter Controls */}
        <div className="flex items-center gap-2">
          {/* Quick Search */}
          <div className="relative">
            <Search className="w-3.5 h-3.5 absolute left-2.5 top-1/2 -translate-y-1/2 text-text-dim" />
            <input
              type="text"
              placeholder="Search sample ID / class..."
              value={filterState.searchQuery}
              onChange={(e) => onFilterChange({ searchQuery: e.target.value })}
              className="bg-bg-card border border-border-subtle rounded pl-8 pr-2.5 py-1 text-xs font-mono text-text-primary placeholder:text-text-dim focus:outline-none focus:border-accent-cyan/50 w-44"
            />
          </div>

          {/* Toggle Filter Panel */}
          <button
            onClick={() => setShowFilters(!showFilters)}
            className={`px-2.5 py-1 text-xs font-mono rounded border flex items-center gap-1.5 transition-colors ${
              showFilters || filterState.status !== 'all' || filterState.minConfidence > 0 || filterState.selectedClass !== 'all'
                ? 'bg-accent-cyan/15 text-accent-cyan border-accent-cyan/40 shadow-glow-cyan'
                : 'bg-bg-card text-text-secondary border-border-subtle hover:text-text-primary'
            }`}
          >
            <SlidersHorizontal className="w-3.5 h-3.5" />
            <span>Filters</span>
          </button>

          {/* Zoom controls */}
          <div className="flex items-center border border-border-subtle rounded bg-bg-card divide-x divide-border-subtle">
            <button
              onClick={handleZoomIn}
              className="p-1.5 text-text-secondary hover:text-text-primary transition-colors focus:outline-none"
              title="Zoom In"
            >
              <ZoomIn className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleZoomOut}
              className="p-1.5 text-text-secondary hover:text-text-primary transition-colors focus:outline-none"
              title="Zoom Out"
            >
              <ZoomOut className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={handleResetZoom}
              className="p-1.5 text-text-secondary hover:text-text-primary transition-colors focus:outline-none"
              title="Reset View"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>

      {/* Filter drawer row (collapsible) */}
      {showFilters && (
        <div className="px-4 py-2.5 bg-bg-subtle/80 border-b border-border-subtle flex flex-wrap items-center gap-4 text-xs font-mono transition-all">
          {/* Status Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-text-dim">Status:</span>
            <div className="flex items-center gap-1 bg-bg-panel p-0.5 rounded border border-border-subtle">
              {(['all', 'stable', 'uncertain', 'failure'] as const).map((s) => (
                <button
                  key={s}
                  onClick={() => onFilterChange({ status: s })}
                  className={`px-2 py-0.5 rounded capitalize ${
                    filterState.status === s
                      ? 'bg-accent-cyan/20 text-accent-cyan font-semibold'
                      : 'text-text-dim hover:text-text-secondary'
                  }`}
                >
                  {s}
                </button>
              ))}
            </div>
          </div>

          {/* Min Confidence Slider */}
          <div className="flex items-center gap-2">
            <span className="text-text-dim">Min Conf:</span>
            <input
              type="range"
              min="0"
              max="95"
              step="5"
              value={filterState.minConfidence}
              onChange={(e) => onFilterChange({ minConfidence: Number(e.target.value) })}
              className="w-24 accent-accent-cyan cursor-pointer"
            />
            <span className="text-accent-cyan w-8">{filterState.minConfidence}%</span>
          </div>

          {/* Class Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-text-dim">Digit Class:</span>
            <select
              value={filterState.selectedClass}
              onChange={(e) => onFilterChange({ selectedClass: e.target.value })}
              className="bg-bg-panel border border-border-subtle rounded px-2 py-0.5 text-xs text-text-primary focus:outline-none"
            >
              <option value="all">All Classes (0-9)</option>
              {Array.from({ length: 10 }).map((_, i) => (
                <option key={i} value={String(i)}>
                  Class {i}
                </option>
              ))}
            </select>
          </div>

          {/* Perturbation Filter */}
          <div className="flex items-center gap-1.5">
            <span className="text-text-dim">Perturbation:</span>
            <select
              value={filterState.perturbation}
              onChange={(e) => onFilterChange({ perturbation: e.target.value })}
              className="bg-bg-panel border border-border-subtle rounded px-2 py-0.5 text-xs text-text-primary focus:outline-none"
            >
              <option value="all">All Modes</option>
              <option value="rotation">Rotation</option>
              <option value="contrast">Contrast / Low-light</option>
              <option value="scale">Scale / Downsample</option>
            </select>
          </div>

          {/* Reset Filters */}
          <button
            onClick={() =>
              onFilterChange({
                status: 'all',
                minConfidence: 0,
                selectedClass: 'all',
                perturbation: 'all',
                searchQuery: '',
                selectedClusterId: null,
              })
            }
            className="ml-auto text-text-dim hover:text-accent-cyan underline"
          >
            Clear Filters
          </button>
        </div>
      )}

      {/* Main Interactive SVG Canvas */}
      <div
        ref={containerRef}
        className="relative w-full h-[520px] bg-[#07090D] cursor-grab active:cursor-grabbing overflow-hidden"
      >
        {/* Subtle grid background pattern */}
        <div
          className="absolute inset-0 pointer-events-none opacity-20"
          style={{
            backgroundImage: `radial-gradient(rgba(101, 214, 255, 0.2) 1px, transparent 1px)`,
            backgroundSize: '24px 24px',
          }}
        />

        <svg
          ref={svgRef}
          viewBox={`0 0 ${V_WIDTH} ${V_HEIGHT}`}
          className="w-full h-full block"
          style={{ willChange: 'transform' }}
        >
          <defs>
            {/* Glow Filters */}
            <filter id="glow-green" x="-40%" y="-40%" width="180%" height="180%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="2.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glow-red" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3.5" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>
            <filter id="glow-cyan" x="-50%" y="-50%" width="200%" height="200%">
              <feGaussianBlur in="SourceGraphic" stdDeviation="3" result="blur" />
              <feMerge>
                <feMergeNode in="blur" />
                <feMergeNode in="SourceGraphic" />
              </feMerge>
            </filter>

            {/* Gradients */}
            <radialGradient id="cluster-glow-bs-01">
              <stop offset="0%" stopColor="#FF5A5F" stopOpacity="0.18" />
              <stop offset="60%" stopColor="#BF5FFF" stopOpacity="0.08" />
              <stop offset="100%" stopColor="#FF5A5F" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="cluster-glow-bs-02">
              <stop offset="0%" stopColor="#F5C451" stopOpacity="0.15" />
              <stop offset="100%" stopColor="#F5C451" stopOpacity="0" />
            </radialGradient>
            <radialGradient id="cluster-glow-bs-03">
              <stop offset="0%" stopColor="#FF5A5F" stopOpacity="0.18" />
              <stop offset="100%" stopColor="#FF5A5F" stopOpacity="0" />
            </radialGradient>
          </defs>

          <g className="zoomable-content">
            {/* 1. Neural Connection Mesh Lines between cluster centroids */}
            <g className="neural-connections" opacity={selectedClusterId ? 0.2 : 0.45}>
              {blindSpots.map((b1, i) =>
                blindSpots.slice(i + 1).map((b2) => (
                  <line
                    key={`conn-${b1.id}-${b2.id}`}
                    x1={b1.clusterCenter.x}
                    y1={b1.clusterCenter.y}
                    x2={b2.clusterCenter.x}
                    y2={b2.clusterCenter.y}
                    stroke="#FF5A5F"
                    strokeWidth="1.2"
                    strokeDasharray="4 4"
                    opacity="0.5"
                  />
                ))
              )}
            </g>

            {/* 2. Failure Region Clusters (Backdrop & Boundary) */}
            {blindSpots.map((cluster) => {
              const isSelected = selectedClusterId === cluster.id;
              const isOtherSelected = selectedClusterId !== null && !isSelected;

              return (
                <g
                  key={cluster.id}
                  className="cursor-pointer transition-opacity duration-300"
                  opacity={isOtherSelected ? 0.25 : 1}
                  onClick={(e) => {
                    e.stopPropagation();
                    onSelectCluster(isSelected ? null : cluster.id);
                  }}
                >
                  {/* Subtle radial glow fill for the blind spot region */}
                  <circle
                    cx={cluster.clusterCenter.x}
                    cy={cluster.clusterCenter.y}
                    r={cluster.radius + (isSelected ? 18 : 6)}
                    fill={`url(#cluster-glow-${cluster.id})`}
                    className="transition-all duration-300"
                  />

                  {/* Pulsing indicator ring */}
                  <circle
                    cx={cluster.clusterCenter.x}
                    cy={cluster.clusterCenter.y}
                    r={cluster.radius + (isSelected ? 16 : 8)}
                    fill="none"
                    stroke={isSelected ? '#65D6FF' : '#FF5A5F'}
                    strokeWidth={isSelected ? '2' : '1.2'}
                    strokeDasharray={isSelected ? '6 4' : '3 3'}
                    strokeOpacity={isSelected ? 0.9 : 0.45}
                    filter={isSelected ? 'url(#glow-cyan)' : 'url(#glow-red)'}
                  />

                  {/* Cluster Label Tag */}
                  <g
                    transform={`translate(${cluster.clusterCenter.x}, ${
                      cluster.clusterCenter.y - cluster.radius - 12
                    })`}
                  >
                    <rect
                      x="-65"
                      y="-12"
                      width="130"
                      height="22"
                      rx="4"
                      fill="#0D1117"
                      stroke={isSelected ? '#65D6FF' : '#1a2035'}
                      strokeWidth={isSelected ? '1.5' : '1'}
                      filter="drop-shadow(0 2px 8px rgba(0,0,0,0.7))"
                    />
                    <text
                      x="0"
                      y="3"
                      textAnchor="middle"
                      fill={isSelected ? '#65D6FF' : '#E8EDF2'}
                      fontSize="9.5"
                      fontFamily="IBM Plex Mono"
                      fontWeight="600"
                    >
                      {cluster.clusterNumber} {cluster.name}
                    </text>
                  </g>
                </g>
              );
            })}

            {/* 3. Scatter Points (~1,500 points) */}
            <g className="scatter-points">
              {filteredSamples.map((sample) => {
                const isSelected = selectedSampleId === sample.id;
                const isInSelectedCluster = selectedClusterId
                  ? sample.clusterId === selectedClusterId
                  : true;

                // Color & glow by status
                let fill = '#4FD18B';
                let stroke = '#4FD18B';
                let filterId: string | undefined = 'url(#glow-green)';
                let radius = 2.4;

                if (sample.stability === 'failure') {
                  fill = '#FF5A5F';
                  stroke = '#FF5A5F';
                  filterId = 'url(#glow-red)';
                  radius = 3.2;
                } else if (sample.stability === 'uncertain') {
                  fill = '#F5C451';
                  stroke = '#F5C451';
                  filterId = undefined;
                  radius = 2.6;
                }

                if (isSelected) {
                  fill = '#65D6FF';
                  stroke = '#ffffff';
                  filterId = 'url(#glow-cyan)';
                  radius = 6;
                }

                // Dim non-selected points when cluster is selected
                let opacity = 0.85;
                if (selectedClusterId && !isInSelectedCluster) {
                  opacity = 0.12;
                } else if (selectedSampleId && !isSelected) {
                  opacity = 0.4;
                }

                return (
                  <circle
                    key={sample.id}
                    cx={sample.x}
                    cy={sample.y}
                    r={radius}
                    fill={fill}
                    stroke={isSelected ? stroke : undefined}
                    strokeWidth={isSelected ? 2 : 0}
                    filter={filterId}
                    opacity={opacity}
                    className="cursor-pointer transition-all duration-200 hover:scale-150"
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
                    onMouseLeave={() => {
                      setTooltip((prev) => ({ ...prev, visible: false }));
                    }}
                    onClick={(e) => {
                      e.stopPropagation();
                      onSelectSample(sample);
                      if (sample.clusterId) {
                        onSelectCluster(sample.clusterId);
                      }
                    }}
                  />
                );
              })}
            </g>
          </g>
        </svg>

        {/* Floating Tooltip */}
        {tooltip.visible && tooltip.sample && (
          <div
            className="absolute z-30 pointer-events-none lab-panel rounded px-3 py-2 border border-border-active shadow-2xl text-xs font-mono transform -translate-x-1/2 -translate-y-full mb-3"
            style={{
              left: `${tooltip.x}px`,
              top: `${tooltip.y}px`,
            }}
          >
            <div className="flex items-center justify-between gap-3 border-b border-border-subtle pb-1 mb-1.5">
              <span className="text-accent-cyan font-semibold">#{tooltip.sample.id}</span>
              <span
                className={`text-[10px] px-1.5 py-0.5 rounded font-bold uppercase ${
                  tooltip.sample.stability === 'failure'
                    ? 'bg-failure/20 text-failure border border-failure/40'
                    : tooltip.sample.stability === 'uncertain'
                    ? 'bg-uncertain/20 text-uncertain border border-uncertain/40'
                    : 'bg-stable/20 text-stable border border-stable/40'
                }`}
              >
                {tooltip.sample.stability}
              </span>
            </div>
            <div className="space-y-1 text-text-secondary text-[11px]">
              <div className="flex justify-between gap-4">
                <span>True / Pred:</span>
                <span className="text-text-primary font-bold">
                  {tooltip.sample.trueClass} → {tooltip.sample.predictedClass}
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Confidence:</span>
                <span
                  className={
                    tooltip.sample.confidence < 60
                      ? 'text-failure font-bold'
                      : tooltip.sample.confidence < 80
                      ? 'text-uncertain'
                      : 'text-stable'
                  }
                >
                  {tooltip.sample.confidence}%
                </span>
              </div>
              <div className="flex justify-between gap-4">
                <span>Perturbation:</span>
                <span className="text-text-primary truncate max-w-[120px]">
                  {tooltip.sample.sensitivity}
                </span>
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Legend & Summary Footer */}
      <div className="px-4 py-2.5 bg-bg-panel border-t border-border-subtle flex flex-wrap items-center justify-between gap-3 text-xs font-mono">
        {/* Status Indicators */}
        <div className="flex items-center gap-4">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-stable shadow-glow-stable" />
            <span className="text-text-secondary">Stable</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-uncertain" />
            <span className="text-text-secondary">Uncertain</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-full bg-failure shadow-glow-failure" />
            <span className="text-text-secondary">Failure</span>
          </div>
        </div>

        {/* Summary Readout specified: Legend: ● Stable ● Uncertain ● Failure · 1,842 samples · 03 regions · 2 HIGH */}
        <div className="text-text-dim flex items-center gap-2">
          <span>·</span>
          <span>1,842 samples</span>
          <span>·</span>
          <span>03 regions</span>
          <span>·</span>
          <span className="text-failure font-semibold">2 HIGH</span>
        </div>
      </div>
    </div>
  );
};
