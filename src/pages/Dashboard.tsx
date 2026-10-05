import React, { useEffect, useState } from 'react';
import { useOutletContext } from 'react-router-dom';
import { AnalysisStage, BlindSpot, FilterState, ModelMetrics, Sample } from '../types/neuralTrail';
import { analysisService } from '../services/analysisService';
import { MetricStrip } from '../components/dashboard/MetricStrip';
import { BlindSpotMap } from '../components/dashboard/BlindSpotMap';
import { FailureRegions } from '../components/dashboard/FailureRegions';
import { ModelHealth } from '../components/dashboard/ModelHealth';
import { BlindSpotDrawer } from '../components/dashboard/BlindSpotDrawer';

export const Dashboard: React.FC = () => {
  const { analysisStage } = useOutletContext<{ analysisStage: AnalysisStage }>() || {
    analysisStage: 'IDLE',
  };

  const [metrics, setMetrics] = useState<ModelMetrics | null>(null);
  const [blindSpots, setBlindSpots] = useState<BlindSpot[]>([]);
  const [samples, setSamples] = useState<Sample[]>([]);
  const [loading, setLoading] = useState(true);

  // Selection states
  const [selectedClusterId, setSelectedClusterId] = useState<string | null>(null);
  const [selectedSample, setSelectedSample] = useState<Sample | null>(null);
  const [drawerOpen, setDrawerOpen] = useState(false);
  const [clusterSamples, setClusterSamples] = useState<Sample[]>([]);

  // Filter state
  const [filterState, setFilterState] = useState<FilterState>({
    status: 'all',
    minConfidence: 0,
    selectedClass: 'all',
    perturbation: 'all',
    searchQuery: '',
    selectedClusterId: null,
  });

  // Load initial laboratory data
  useEffect(() => {
    async function loadData() {
      setLoading(true);
      const [m, b, s] = await Promise.all([
        analysisService.getModelMetrics(),
        analysisService.getBlindSpots(),
        analysisService.getSamples(),
      ]);
      setMetrics(m);
      setBlindSpots(b);
      setSamples(s);
      setLoading(false);
    }
    loadData();
  }, []);

  // Update cluster samples when a cluster is selected
  useEffect(() => {
    if (selectedClusterId) {
      analysisService.getSamplesByCluster(selectedClusterId, 6).then((res) => {
        setClusterSamples(res);
        setDrawerOpen(true);
      });
    } else if (!selectedSample) {
      setClusterSamples([]);
    }
  }, [selectedClusterId, selectedSample]);

  const handleSelectCluster = (clusterId: string | null) => {
    if (clusterId === selectedClusterId) {
      setSelectedClusterId(null);
      setDrawerOpen(false);
    } else {
      setSelectedClusterId(clusterId);
      setSelectedSample(null);
      if (clusterId) {
        setDrawerOpen(true);
      }
    }
  };

  const handleSelectSample = (sample: Sample | null) => {
    setSelectedSample(sample);
    if (sample) {
      setDrawerOpen(true);
    }
  };

  const handleFilterChange = (updates: Partial<FilterState>) => {
    setFilterState((prev) => ({ ...prev, ...updates }));
  };

  const activeBlindSpot = blindSpots.find((b) => b.id === selectedClusterId) || null;

  if (loading || !metrics) {
    return (
      <div className="h-full w-full flex flex-col items-center justify-center p-8 text-center">
        <div className="w-12 h-12 rounded-full border-2 border-accent-cyan/30 border-t-accent-cyan animate-spin mb-4 shadow-glow-cyan" />
        <span className="font-mono text-sm tracking-wider text-accent-cyan uppercase">
          INITIALIZING LAB NEURAL PROBES...
        </span>
      </div>
    );
  }

  return (
    <div className="p-4 md:p-6 space-y-4 max-w-[1720px] mx-auto transition-opacity duration-300">
      {/* Question Header Banner */}
      <div className="flex flex-wrap items-center justify-between pb-1 gap-2 border-b border-border-subtle/50">
        <div>
          <h1 className="text-lg md:text-xl font-heading font-bold text-text-primary tracking-tight flex items-center gap-2">
            <span>MODEL DIAGNOSTIC OVERVIEW</span>
            <span className="text-xs font-mono font-normal text-text-dim">
              [ Question: Where is this model failing? ]
            </span>
          </h1>
        </div>
        <div className="text-xs font-mono text-text-dim flex items-center gap-2">
          <span>Active Session ID:</span>
          <span className="text-accent-cyan bg-bg-card px-2 py-0.5 rounded border border-border-subtle">
            LAB-ACTIVE-CORE-V2
          </span>
        </div>
      </div>

      {/* 1. Metric Strip */}
      <MetricStrip metrics={metrics} />

      {/* 2. Hero: Blind Spot Map */}
      <BlindSpotMap
        samples={samples}
        blindSpots={blindSpots}
        selectedClusterId={selectedClusterId}
        selectedSampleId={selectedSample?.id || null}
        onSelectCluster={handleSelectCluster}
        onSelectSample={handleSelectSample}
        filterState={filterState}
        onFilterChange={handleFilterChange}
      />

      {/* 3. Bottom Panels Grid: Failure Regions (Left) & Model Health (Right) */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        {/* Failure Region List */}
        <FailureRegions
          blindSpots={blindSpots}
          selectedClusterId={selectedClusterId}
          onSelectCluster={(id) => handleSelectCluster(id)}
        />

        {/* Model Health Panel */}
        <ModelHealth metrics={metrics} />
      </div>

      {/* Slide-in Investigation Drawer */}
      <BlindSpotDrawer
        isOpen={drawerOpen}
        onClose={() => {
          setDrawerOpen(false);
          setSelectedClusterId(null);
          setSelectedSample(null);
        }}
        blindSpot={activeBlindSpot}
        selectedSample={selectedSample}
        clusterSamples={clusterSamples}
        onSelectSample={(s) => setSelectedSample(s)}
      />
    </div>
  );
};
