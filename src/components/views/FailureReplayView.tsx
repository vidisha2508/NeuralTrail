import React, { useState, useEffect } from 'react';
import { 
  Play, 
  Pause, 
  RotateCcw, 
  ChevronRight, 
  ChevronLeft,
  ShieldCheck, 
  AlertTriangle, 
  Activity, 
  Compass, 
  Sliders, 
  Brain, 
  Terminal, 
  CheckCircle2,
  History,
  RefreshCw
} from 'lucide-react';
import { PageHeader } from '../common/PageHeader';
import { Card } from '../common/Card';
import { mlApiClient } from '../../services/mlApiClient';

export const FailureReplayView: React.FC = () => {
  const [activeStep, setActiveStep] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [events, setEvents] = useState<any[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const fetchEvents = async () => {
    setIsLoading(true);
    try {
      const res = await mlApiClient.getReplayEvents();
      if (res && Array.isArray(res.events)) {
        setEvents(res.events);
        if (res.events.length > 0 && activeStep >= res.events.length) {
          setActiveStep(res.events.length - 1);
        }
      }
    } catch {
      // Standby
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    fetchEvents();
  }, []);

  // Step autoplay timer
  useEffect(() => {
    let interval: any;
    if (isPlaying && events.length > 1) {
      interval = setInterval(() => {
        setActiveStep((prev) => (prev + 1) % events.length);
      }, 2500);
    }
    return () => clearInterval(interval);
  }, [isPlaying, events.length]);

  const getEventIcon = (type: string) => {
    switch (type) {
      case 'MODEL_LOADED': return Activity;
      case 'MANIFOLD_MAPPED': return Compass;
      case 'SAMPLE_SELECTED': return ShieldCheck;
      case 'STRESS_TEST': return Sliders;
      case 'HYPOTHESIS_FORMULATED': return Brain;
      case 'EXPERIMENT_EXECUTED': return Terminal;
      default: return CheckCircle2;
    }
  };

  const getAccentColor = (accent: string) => {
    switch (accent) {
      case 'green': return '#00ff88';
      case 'magenta': return '#ff007f';
      case 'amber': return '#ffb300';
      case 'purple': return '#d500f9';
      default: return '#00f0ff';
    }
  };

  if (events.length === 0 && !isLoading) {
    return (
      <div className="flex-1 flex flex-col items-center justify-center p-8 text-center font-mono">
        <div className="w-14 h-14 rounded-xl bg-white/[0.03] border border-white/10 flex items-center justify-center mb-4">
          <History className="w-7 h-7 text-[#ff007f]" />
        </div>
        <h3 className="font-display font-bold text-xl text-white">INVESTIGATION TIMELINE STANDBY</h3>
        <p className="text-xs text-white/50 max-w-md mt-2 font-sans leading-relaxed">
          The replay timeline records genuine events from your active investigation: model loading, manifold clustering, sample inspections, stress perturbations, and executed mitigation trials.
        </p>
        <button
          onClick={fetchEvents}
          className="mt-5 px-4 py-2 rounded text-xs uppercase font-bold bg-white/10 hover:bg-white/15 text-white flex items-center gap-2 transition-colors"
        >
          <RefreshCw className="w-3.5 h-3.5" />
          <span>Refresh Investigation Events</span>
        </button>
      </div>
    );
  }

  const currentEvent = events[activeStep] || events[0];
  const Icon = currentEvent ? getEventIcon(currentEvent.type) : Activity;
  const accentColor = currentEvent ? getAccentColor(currentEvent.accent) : '#00f0ff';

  return (
    <div className="flex-1 flex flex-col overflow-y-auto px-6 py-6 md:px-8 space-y-6">
      <PageHeader
        stepNumber="05"
        stepCode="REPLAY"
        title="Chronological Investigation Timeline"
        description="Step-by-step empirical audit trail of model loading, error discovery, stress tests, and mitigation results."
        badge={
          <span className="font-mono text-xs text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
            {events.length} RECORDED EVENTS
          </span>
        }
        actions={
          <div className="flex items-center gap-2">
            <button
              onClick={() => setIsPlaying(!isPlaying)}
              className="px-3 py-1.5 rounded bg-white/10 hover:bg-white/20 text-white font-mono text-xs font-semibold flex items-center gap-1.5 transition-colors"
            >
              {isPlaying ? <Pause className="w-3 h-3 fill-current" /> : <Play className="w-3 h-3 fill-current" />}
              <span>{isPlaying ? 'PAUSE' : 'PLAY TIMELINE'}</span>
            </button>
            <button
              onClick={() => setActiveStep(0)}
              className="px-2.5 py-1.5 rounded bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
              title="Reset to Step 1"
            >
              <RotateCcw className="w-3.5 h-3.5" />
            </button>
            <button
              onClick={fetchEvents}
              className="px-2.5 py-1.5 rounded bg-white/5 hover:bg-white/10 text-white/60 hover:text-white transition-colors"
              title="Refresh Events"
            >
              <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
            </button>
          </div>
        }
      />

      {/* Interactive Step Strip */}
      <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-7 gap-2">
        {events.map((evt, idx) => {
          const EvtIcon = getEventIcon(evt.type);
          const isSelected = activeStep === idx;
          const col = getAccentColor(evt.accent);

          return (
            <button
              key={evt.id || idx}
              onClick={() => setActiveStep(idx)}
              className={`p-3 rounded-lg border text-left transition-all font-mono relative overflow-hidden ${
                isSelected
                  ? 'bg-white/[0.08] shadow-lg'
                  : 'bg-white/[0.02] border-white/10 hover:bg-white/[0.05]'
              }`}
              style={{ borderColor: isSelected ? col : undefined }}
            >
              {isSelected && (
                <div
                  className="absolute top-0 left-0 right-0 h-0.5"
                  style={{ backgroundColor: col }}
                />
              )}
              <div className="flex items-center justify-between text-[10px] text-white/40 mb-1">
                <span>{evt.code || `0${idx+1}`}</span>
                <span className="text-[9px]">{evt.time_str || ''}</span>
              </div>
              <div className="font-display font-bold text-xs text-white truncate">
                {evt.short || evt.type}
              </div>
              <div className="mt-1 text-[10px] truncate" style={{ color: col }}>
                {evt.title}
              </div>
            </button>
          );
        })}
      </div>

      {/* Main Focus Card for the Active Event */}
      {currentEvent && (
        <Card
          className="bg-black/60 border-white/15 shadow-2xl p-6"
          headerTitle={`EVENT ${currentEvent.code}: ${currentEvent.title}`}
          headerSubtitle={`RECORDED AT: ${currentEvent.time_str || 'SESSION RUNTIME'} • TYPE: ${currentEvent.type}`}
          headerIcon={<Icon className="w-5 h-5" style={{ color: accentColor }} />}
        >
          <div className="space-y-6 pt-2">
            <div>
              <span className="text-white/40 text-[10px] uppercase block mb-1 font-mono">OBSERVED EVENT DESCRIPTION:</span>
              <p className="text-sm text-white font-sans leading-relaxed">
                {currentEvent.desc}
              </p>
            </div>

            {/* Telemetry Strip */}
            <div className="p-4 rounded-lg bg-black/60 border border-white/10 font-mono text-xs space-y-1">
              <span className="text-white/40 text-[10px] uppercase block">RECORDED TELEMETRY & MEASUREMENTS:</span>
              <div className="text-white font-semibold" style={{ color: accentColor }}>
                {currentEvent.telemetry}
              </div>
            </div>

            {/* Stepper Navigation Buttons */}
            <div className="flex items-center justify-between pt-2 border-t border-white/10 font-mono text-xs">
              <button
                onClick={() => setActiveStep((prev) => Math.max(0, prev - 1))}
                disabled={activeStep === 0}
                className="px-4 py-2 rounded bg-white/5 hover:bg-white/10 text-white flex items-center gap-2 disabled:opacity-30 transition-colors"
              >
                <ChevronLeft className="w-4 h-4" />
                <span>PREVIOUS EVENT</span>
              </button>

              <span className="text-white/40">
                {activeStep + 1} OF {events.length}
              </span>

              <button
                onClick={() => setActiveStep((prev) => Math.min(events.length - 1, prev + 1))}
                disabled={activeStep === events.length - 1}
                className="px-4 py-2 rounded bg-white/5 hover:bg-white/10 text-white flex items-center gap-2 disabled:opacity-30 transition-colors"
              >
                <span>NEXT EVENT</span>
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </div>
        </Card>
      )}
    </div>
  );
};
