import React, { useState, useRef, useEffect } from 'react';
import { Play, Tv, Cpu, RefreshCw, ChevronDown, CheckCircle2, AlertCircle, LogOut } from 'lucide-react';

interface WorkstationTopBarProps {
  activeModelName?: string;
  modelStatus?: string;
  isLoaded: boolean;
  crtActive: boolean;
  onToggleCrt: () => void;
  onOpenModelModal: () => void;
  onSelectModel: (modelType: string) => Promise<void>;
  onUnloadModel: () => Promise<void>;
  onRunScan: () => void;
  onExportModel: () => void;
  isScanning: boolean;
  isExporting?: boolean;
}

export const WorkstationTopBar: React.FC<WorkstationTopBarProps> = ({
  activeModelName = 'No Model Loaded',
  modelStatus = 'Standby',
  isLoaded,
  crtActive,
  onToggleCrt,
  onOpenModelModal,
  onSelectModel,
  onUnloadModel,
  onRunScan,
  onExportModel,
  isScanning,
  isExporting = false,
}) => {
  const [dropdownOpen, setDropdownOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setDropdownOpen(false);
      }
    };
    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <header className="h-16 shrink-0 border-b border-white/10 bg-[#0c0418]/95 backdrop-blur-md px-6 flex items-center justify-between z-30 select-none">
      {/* Left: Application Brand */}
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-md bg-[#180933] border border-[#7c4dff]/40 flex items-center justify-center shrink-0">
          <Cpu className="w-4 h-4 text-[#00ff88]" />
        </div>
        <div className="flex items-center gap-2.5">
          <span className="font-display font-bold text-base tracking-wider text-white">
            NEURAL TRAIL
          </span>
          <span className="text-white/20">|</span>
          <span className="font-sans text-xs text-white/60 font-medium tracking-wide">
            Model Diagnostics & Latent Cartography
          </span>
        </div>
      </div>

      {/* Center: Model Selector & System Status */}
      <div className="hidden md:flex items-center gap-3 relative" ref={dropdownRef}>
        {/* Model Selector Dropdown Button */}
        <button
          onClick={() => setDropdownOpen(!dropdownOpen)}
          className={`h-9 px-3.5 rounded-md border flex items-center gap-2.5 transition-all font-sans text-xs ${
            isLoaded
              ? 'bg-[#16092e] border-[#00ff88]/40 text-white'
              : 'bg-[#210f0f] border-[#ffb300]/40 text-[#ffb300]'
          }`}
          title="Switch Active Model"
        >
          <span className="text-white/40 font-medium">Model:</span>
          <span className="font-mono font-semibold tracking-wide">
            {isLoaded ? activeModelName : 'No Model Loaded (Standby)'}
          </span>
          <ChevronDown className="w-3.5 h-3.5 text-white/40 ml-1" />
        </button>

        {/* Dropdown Menu */}
        {dropdownOpen && (
          <div className="absolute top-11 left-0 w-64 bg-[#14052b] border border-white/20 rounded-lg shadow-2xl overflow-hidden py-1 z-50 font-mono text-xs">
            <div className="px-3 py-1.5 text-[10px] text-white/40 uppercase tracking-wider border-b border-white/10">
              Select Model Architecture
            </div>

            <button
              onClick={() => {
                setDropdownOpen(false);
                onSelectModel('resnet18');
              }}
              className="w-full text-left px-3 py-2 hover:bg-white/10 flex items-center justify-between text-white transition-colors"
            >
              <span>ResNet-18 (Torchvision)</span>
              <span className="text-[10px] text-[#00ff88]">18-layer</span>
            </button>

            <button
              onClick={() => {
                setDropdownOpen(false);
                onSelectModel('mobilenet');
              }}
              className="w-full text-left px-3 py-2 hover:bg-white/10 flex items-center justify-between text-white transition-colors"
            >
              <span>MobileNet-V2 (Torchvision)</span>
              <span className="text-[10px] text-[#00f0ff]">Inverted</span>
            </button>

            <button
              onClick={() => {
                setDropdownOpen(false);
                onSelectModel('custom_test');
              }}
              className="w-full text-left px-3 py-2 hover:bg-white/10 flex items-center justify-between text-white transition-colors"
            >
              <span>NeuralTrailCNN (custom/)</span>
              <span className="text-[10px] text-[#d500f9]">Test Model</span>
            </button>

            <div className="border-t border-white/10 my-1" />

            <button
              onClick={() => {
                setDropdownOpen(false);
                onOpenModelModal();
              }}
              className="w-full text-left px-3 py-2 hover:bg-white/10 flex items-center gap-2 text-[#00f0ff] transition-colors"
            >
              <span>Upload Custom Checkpoint...</span>
            </button>

            {isLoaded && (
              <button
                onClick={() => {
                  setDropdownOpen(false);
                  onUnloadModel();
                }}
                className="w-full text-left px-3 py-2 hover:bg-[#ff007f]/20 flex items-center gap-2 text-[#ff007f] transition-colors border-t border-white/10"
              >
                <LogOut className="w-3.5 h-3.5" />
                <span>Unload Model (Reset to Standby)</span>
              </button>
            )}
          </div>
        )}

        {/* System Status */}
        <div
          className={`h-9 px-3 rounded-md border flex items-center gap-2 text-xs font-mono font-medium ${
            isLoaded
              ? 'bg-[#0e2118] border-[#00ff88]/30 text-[#00ff88]'
              : 'bg-[#2a1b0a] border-[#ffb300]/30 text-[#ffb300]'
          }`}
        >
          <span
            className={`w-2 h-2 rounded-full ${
              isLoaded ? 'bg-[#00ff88]' : 'bg-[#ffb300]'
            }`}
          />
          <span>{isLoaded ? modelStatus : 'Standby (Empty State)'}</span>
        </div>
      </div>

      {/* Right Controls */}
      <div className="flex items-center gap-2.5">
        {/* CRT FX Toggle */}
        <button
          onClick={onToggleCrt}
          className={`px-2 py-1 rounded font-mono text-[11px] border transition-colors flex items-center gap-1 font-medium ${
            crtActive
              ? 'bg-[#d500f9]/20 border-[#d500f9]/60 text-[#d500f9]'
              : 'bg-[#180533] border-white/10 text-white/50 hover:text-white'
          }`}
          title="Toggle CRT Screen Scanlines"
        >
          <Tv className="w-3 h-3" />
          <span className="hidden sm:inline">CRT FX</span>
        </button>

        {isLoaded && (
          <button
            onClick={onExportModel}
            disabled={isExporting}
            className="px-3 py-1.5 rounded bg-[#180533] border border-white/15 hover:border-white/30 text-white font-mono text-xs font-semibold transition-colors disabled:opacity-50"
            title="Export Model (.pth)"
          >
            {isExporting ? 'Exporting...' : 'Export .pth'}
          </button>
        )}

        {/* Switch / Load Model Button */}
        <button
          onClick={onOpenModelModal}
          className="px-3.5 py-1.5 rounded bg-[#180533] border border-white/20 hover:border-white/40 text-white font-mono text-xs font-semibold transition-colors"
        >
          {isLoaded ? 'Switch Model' : 'Load Model'}
        </button>

        {/* Run Evaluation Scan Button */}
        {isLoaded && (
          <button
            onClick={onRunScan}
            disabled={isScanning}
            className="vapor-btn-green px-4 py-1.5 text-xs font-mono tracking-wider flex items-center gap-1.5 font-bold shadow-neon-green"
          >
            {isScanning ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Evaluating...</span>
              </>
            ) : (
              <>
                <Play className="w-3 h-3 fill-current" />
                <span>Re-Evaluate</span>
              </>
            )}
          </button>
        )}
      </div>
    </header>
  );
};
