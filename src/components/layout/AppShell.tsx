import React, { useState } from 'react';
import { Outlet } from 'react-router-dom';
import { Sidebar } from './Sidebar';
import { TopBar } from './TopBar';
import { AnalysisStage } from '../../types/neuralTrail';

export const AppShell: React.FC = () => {
  const [collapsed, setCollapsed] = useState(false);
  const [analysisStage, setAnalysisStage] = useState<AnalysisStage>('IDLE');

  return (
    <div className="flex h-screen w-screen overflow-hidden bg-bg-primary text-text-primary">
      {/* Sidebar */}
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed(!collapsed)} />

      {/* Main Content Area */}
      <div className="flex-1 flex flex-col min-w-0 overflow-hidden">
        <TopBar onRunAnalysis={setAnalysisStage} />

        <main className="flex-1 overflow-y-auto overflow-x-hidden relative">
          <Outlet context={{ analysisStage }} />
        </main>
      </div>
    </div>
  );
};
