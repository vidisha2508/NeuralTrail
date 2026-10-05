import React from 'react';
import { WorkstationTab } from '../../types/neuralTrail';
import { 
  Compass, 
  Activity, 
  Sliders, 
  GitCompare, 
  History, 
  Terminal, 
  Brain
} from 'lucide-react';

interface WorkstationNavDeckProps {
  activeTab: WorkstationTab;
  onSelectTab: (tab: WorkstationTab) => void;
  clusterCount?: number;
  failureCount?: number;
}

export const WorkstationNavDeck: React.FC<WorkstationNavDeckProps> = ({
  activeTab,
  onSelectTab,
}) => {
  const navItems = [
    {
      id: 'FIND' as WorkstationTab,
      code: '01',
      label: 'FIND',
      sub: 'Blind Spot Mapping',
      icon: Compass,
      activeColor: 'text-[#00ff88]',
      activeIndicator: 'bg-[#00ff88]',
      activeBg: 'bg-[#00ff88]/[0.08]',
    },
    {
      id: 'TRACE' as WorkstationTab,
      code: '02',
      label: 'TRACE',
      sub: 'Activation Path',
      icon: Activity,
      activeColor: 'text-[#d500f9]',
      activeIndicator: 'bg-[#d500f9]',
      activeBg: 'bg-[#d500f9]/[0.08]',
    },
    {
      id: 'STRESS' as WorkstationTab,
      code: '03',
      label: 'STRESS',
      sub: 'What-If Testing',
      icon: Sliders,
      activeColor: 'text-[#ffb300]',
      activeIndicator: 'bg-[#ffb300]',
      activeBg: 'bg-[#ffb300]/[0.08]',
    },
    {
      id: 'IMPROVE' as WorkstationTab,
      code: '04',
      label: 'IMPROVE',
      sub: 'Model Comparison',
      icon: GitCompare,
      activeColor: 'text-[#00f0ff]',
      activeIndicator: 'bg-[#00f0ff]',
      activeBg: 'bg-[#00f0ff]/[0.08]',
    },
    {
      id: 'REPLAY' as WorkstationTab,
      code: '05',
      label: 'REPLAY',
      sub: 'Failure Timeline',
      icon: History,
      activeColor: 'text-[#ff007f]',
      activeIndicator: 'bg-[#ff007f]',
      activeBg: 'bg-[#ff007f]/[0.08]',
    },
    {
      id: 'EXPERIMENTS' as WorkstationTab,
      code: '06',
      label: 'ENGINE',
      sub: 'Experiment Telemetry',
      icon: Terminal,
      activeColor: 'text-[#b388ff]',
      activeIndicator: 'bg-[#b388ff]',
      activeBg: 'bg-[#b388ff]/[0.08]',
    },
    {
      id: 'RESEARCH' as WorkstationTab,
      code: '07',
      label: 'RESEARCH',
      sub: 'Analysis Assistant',
      icon: Brain,
      activeColor: 'text-[#00f0ff]',
      activeIndicator: 'bg-[#00f0ff]',
      activeBg: 'bg-[#00f0ff]/[0.08]',
    },
  ];

  return (
    <aside className="w-[245px] shrink-0 bg-[#0e051c]/95 border-r border-white/10 flex flex-col justify-between select-none z-20">
      {/* Navigation List */}
      <div className="py-4">
        <div className="px-5 pb-3 text-[11px] font-sans font-semibold text-white/35 uppercase tracking-wider">
          Diagnostic Workflow
        </div>

        <nav className="space-y-1 px-2.5">
          {navItems.map((item) => {
            const Icon = item.icon;
            const isActive = activeTab === item.id;

            return (
              <button
                key={item.id}
                onClick={() => onSelectTab(item.id)}
                className={`w-full text-left px-3 py-2.5 rounded-md flex items-center gap-3 transition-all duration-150 relative ${
                  isActive
                    ? `${item.activeBg} text-white`
                    : 'text-white/60 hover:text-white hover:bg-white/[0.04]'
                }`}
              >
                {/* Active Indicator Bar */}
                {isActive && (
                  <span
                    className={`absolute left-0 top-1.5 bottom-1.5 w-1 rounded-r ${item.activeIndicator}`}
                  />
                )}

                <span
                  className={`font-mono text-[11px] font-semibold tracking-wider ${
                    isActive ? item.activeColor : 'text-white/35'
                  }`}
                >
                  {item.code}
                </span>

                <Icon
                  className={`w-4 h-4 shrink-0 transition-colors ${
                    isActive ? item.activeColor : 'text-white/40'
                  }`}
                />

                <div className="min-w-0 flex-1">
                  <div className="font-display font-semibold text-xs tracking-wider text-white">
                    {item.label}
                  </div>
                  <div className="text-[11px] font-sans text-white/45 truncate leading-tight mt-0.5">
                    {item.sub}
                  </div>
                </div>
              </button>
            );
          })}
        </nav>
      </div>

      {/* Footer Info */}
      <div className="p-4 border-t border-white/10 text-xs text-white/40 font-mono flex items-center justify-between">
        <span className="flex items-center gap-1.5">
          <span className="w-1.5 h-1.5 rounded-full bg-[#00ff88]" />
          <span>PyTorch v2.4</span>
        </span>
        <span className="text-[#00ff88] text-[11px] font-medium">READY</span>
      </div>
    </aside>
  );
};
