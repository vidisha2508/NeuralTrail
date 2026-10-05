import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard,
  Search,
  Activity,
  ShieldAlert,
  GitBranch,
  Wand2,
  GitCompare,
  Database,
  Bot,
  ChevronLeft,
  ChevronRight,
  Cpu,
} from 'lucide-react';

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export const Sidebar: React.FC<SidebarProps> = ({ collapsed, onToggle }) => {
  const navSections = [
    {
      group: 'OVERVIEW',
      items: [
        { name: 'Dashboard', path: '/', icon: LayoutDashboard },
      ],
    },
    {
      group: 'INVESTIGATION',
      items: [
        { name: 'Blind Spot Explorer', path: '/blind-spots', icon: Search },
        { name: 'Neural X-Ray', path: '/neural-xray', icon: Activity },
        { name: 'Robustness Lab', path: '/robustness', icon: ShieldAlert },
        { name: 'What-If Lab', path: '/what-if', icon: GitBranch },
        { name: 'Intervention Lab', path: '/intervention', icon: Wand2 },
      ],
    },
    {
      group: 'ANALYSIS',
      items: [
        { name: 'Model Comparison', path: '/comparison', icon: GitCompare },
        { name: 'Dataset Analysis', path: '/dataset', icon: Database },
      ],
    },
    {
      group: 'RESEARCH',
      items: [
        { name: 'AI Assistant', path: '/research', icon: Bot },
      ],
    },
  ];

  return (
    <aside
      className={`relative flex flex-col bg-bg-panel border-r border-border-subtle transition-all duration-300 ease-in-out select-none z-30 ${
        collapsed ? 'w-16' : 'w-60'
      }`}
    >
      {/* Brand Header */}
      <div className="h-14 flex items-center px-4 border-b border-border-subtle justify-between overflow-hidden">
        {!collapsed ? (
          <div className="flex items-center gap-2.5 min-w-0">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-accent-cyan/20 to-vaporwave-purple/20 border border-accent-cyan/40 flex items-center justify-center flex-shrink-0 shadow-glow-cyan">
              <Cpu className="w-4 h-4 text-accent-cyan" />
            </div>
            <div className="min-w-0">
              <div className="font-heading font-bold text-sm tracking-wider text-text-primary uppercase flex items-center gap-1.5">
                <span>NEURAL TRAIL</span>
              </div>
              <div className="text-[10px] font-mono text-text-dim tracking-tight truncate">
                model debugging lab
              </div>
            </div>
          </div>
        ) : (
          <div className="mx-auto">
            <div className="w-8 h-8 rounded bg-gradient-to-br from-accent-cyan/20 to-vaporwave-purple/20 border border-accent-cyan/40 flex items-center justify-center shadow-glow-cyan">
              <Cpu className="w-4 h-4 text-accent-cyan" />
            </div>
          </div>
        )}

        <button
          onClick={onToggle}
          className="p-1 rounded text-text-dim hover:text-text-primary hover:bg-bg-subtle transition-colors focus:outline-none"
          title={collapsed ? 'Expand Sidebar' : 'Collapse Sidebar'}
        >
          {collapsed ? <ChevronRight className="w-4 h-4" /> : <ChevronLeft className="w-4 h-4" />}
        </button>
      </div>

      {/* Nav Groups */}
      <div className="flex-1 overflow-y-auto overflow-x-hidden py-3 px-2 space-y-4">
        {navSections.map((section) => (
          <div key={section.group} className="space-y-1">
            {!collapsed && (
              <div className="px-2 py-1 text-[10px] font-mono tracking-wider text-text-dim font-medium uppercase">
                {section.group}
              </div>
            )}
            {section.items.map((item) => {
              const Icon = item.icon;
              return (
                <NavLink
                  key={item.path}
                  to={item.path}
                  className={({ isActive }) =>
                    `flex items-center gap-2.5 px-2.5 py-1.5 rounded text-xs transition-all duration-150 group relative ${
                      isActive
                        ? 'bg-accent-cyan/10 text-accent-cyan font-medium border border-accent-cyan/30 shadow-[0_0_12px_rgba(101,214,255,0.15)]'
                        : 'text-text-secondary hover:text-text-primary hover:bg-bg-subtle/70'
                    }`
                  }
                  title={collapsed ? item.name : undefined}
                >
                  {({ isActive }) => (
                    <>
                      <Icon
                        className={`w-4 h-4 flex-shrink-0 transition-colors ${
                          isActive
                            ? 'text-accent-cyan drop-shadow-[0_0_6px_rgba(101,214,255,0.8)]'
                            : 'text-text-dim group-hover:text-text-primary'
                        }`}
                      />
                      {!collapsed && (
                        <span className="truncate tracking-wide font-sans">{item.name}</span>
                      )}
                      {isActive && (
                        <div className="absolute right-0 top-1/2 -translate-y-1/2 w-1 h-3.5 bg-accent-cyan rounded-l shadow-glow-cyan" />
                      )}
                    </>
                  )}
                </NavLink>
              );
            })}
          </div>
        ))}
      </div>

      {/* Bottom Status Panel */}
      <div className="p-3 border-t border-border-subtle bg-bg-primary/50 text-[11px] font-mono">
        {!collapsed ? (
          <div className="space-y-2">
            <div className="flex items-center gap-1.5 text-stable">
              <span className="w-2 h-2 rounded-full bg-stable shadow-glow-stable animate-pulse" />
              <span className="tracking-wider uppercase text-[10px] font-semibold">ANALYSIS ENGINE READY</span>
            </div>
            <div className="space-y-1 text-text-dim text-[10px]">
              <div className="flex justify-between">
                <span>Model:</span>
                <span className="text-text-secondary font-mono">Active Vision Core</span>
              </div>
              <div className="flex justify-between">
                <span>Dataset:</span>
                <span className="text-text-secondary font-mono truncate max-w-[100px]" title="Demo Vision Set">Demo Vision</span>
              </div>
            </div>
          </div>
        ) : (
          <div className="flex justify-center" title="Analysis Engine Ready">
            <span className="w-2.5 h-2.5 rounded-full bg-stable shadow-glow-stable animate-pulse" />
          </div>
        )}
      </div>
    </aside>
  );
};
