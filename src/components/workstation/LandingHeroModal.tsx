import React, { useState } from 'react';
import { Play, Cpu, Layers, Upload, CheckCircle2, AlertCircle, X, Sparkles, FolderCode } from 'lucide-react';
import { mlApiClient } from '../../services/mlApiClient';

interface LandingHeroModalProps {
  isOpen: boolean;
  onClose: () => void;
  activeModelName?: string;
  isLoaded: boolean;
  onModelSelected: (modelType: string, customData?: any) => Promise<void>;
}

export const LandingHeroModal: React.FC<LandingHeroModalProps> = ({
  isOpen,
  onClose,
  activeModelName = 'No Model Loaded',
  isLoaded,
  onModelSelected,
}) => {
  const [selectedTab, setSelectedTab] = useState<'BUILTIN' | 'CUSTOM_TEST' | 'UPLOAD'>('BUILTIN');
  const [loadingModel, setLoadingModel] = useState<string | null>(null);
  const [errorMsg, setErrorMsg] = useState<string | null>(null);

  // Upload fields
  const [weightFile, setWeightFile] = useState<File | null>(null);
  const [modelCode, setModelCode] = useState<string>('');
  const [className, setClassName] = useState<string>('');
  const [inputShape, setInputShape] = useState<string>('3, 224, 224');
  const [classesList, setClassesList] = useState<string>('');

  if (!isOpen) return null;

  const handleSelectBuiltin = async (modelType: 'resnet18' | 'mobilenet') => {
    setLoadingModel(modelType);
    setErrorMsg(null);
    try {
      await onModelSelected(modelType);
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load model.');
    } finally {
      setLoadingModel(null);
    }
  };

  const handleSelectCustomTest = async () => {
    setLoadingModel('custom_test');
    setErrorMsg(null);
    try {
      await onModelSelected('custom_test');
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Failed to load custom test model.');
    } finally {
      setLoadingModel(null);
    }
  };

  const handleUploadSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!weightFile) {
      setErrorMsg('Please select a PyTorch weight file (.pt or .pth).');
      return;
    }

    setLoadingModel('upload');
    setErrorMsg(null);

    try {
      const formData = new FormData();
      formData.append('file', weightFile);
      if (modelCode.trim()) formData.append('model_code', modelCode.trim());
      if (className.trim()) formData.append('class_name', className.trim());
      if (inputShape.trim()) formData.append('shape', inputShape.trim());
      if (classesList.trim()) formData.append('classes', classesList.trim());

      await onModelSelected('upload', formData);
      onClose();
    } catch (err: any) {
      setErrorMsg(err.message || 'Model upload and validation failed.');
    } finally {
      setLoadingModel(null);
    }
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/85 backdrop-blur-md animate-fade-in font-mono">
      <div className="w-full max-w-2xl rounded-xl border border-white/20 bg-[#100224] shadow-2xl overflow-hidden flex flex-col max-h-[90vh]">
        {/* Header Bar */}
        <div className="px-5 py-3.5 border-b border-white/10 bg-[#160430] flex items-center justify-between shrink-0">
          <div className="flex items-center gap-2">
            <span className="w-2.5 h-2.5 rounded-full bg-[#ff007f] cursor-pointer" onClick={onClose} />
            <span className="w-2.5 h-2.5 rounded-full bg-[#ffb300] cursor-pointer" onClick={onClose} />
            <span className="w-2.5 h-2.5 rounded-full bg-[#00ff88] cursor-pointer" onClick={onClose} />
            <span className="font-display font-bold text-xs text-white uppercase tracking-wider ml-2">
              Model Selection Registry
            </span>
          </div>
          <button
            onClick={onClose}
            className="text-white/40 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Modal Body */}
        <div className="p-6 space-y-6 overflow-y-auto">
          {/* Header Title */}
          <div className="text-center space-y-1">
            <h1 className="font-display font-black text-2xl tracking-wide text-white">
              SELECT PYTORCH VISION MODEL
            </h1>
            <p className="text-xs text-[#00ff88] font-semibold uppercase tracking-wider">
              Connected PyTorch Diagnostics Engine
            </p>
            <p className="text-xs text-white/60 font-sans max-w-lg mx-auto pt-1">
              Select one of the genuine vision architectures below. Neural Trail will load the model weights,
              run empirical evaluation, and connect the diagnostics pipeline end-to-end.
            </p>
          </div>

          {/* Current Status Strip */}
          <div className="p-3 rounded-md bg-white/[0.03] border border-white/10 flex items-center justify-between text-xs">
            <div className="flex items-center gap-2">
              <span className="text-white/40 uppercase text-[10px]">CURRENT MODEL:</span>
              <span className="font-bold text-white">{activeModelName}</span>
            </div>
            <div className="flex items-center gap-1.5 font-mono">
              <span className={`w-2 h-2 rounded-full ${isLoaded ? 'bg-[#00ff88]' : 'bg-[#ffb300]'}`} />
              <span className={isLoaded ? 'text-[#00ff88]' : 'text-[#ffb300]'}>
                {isLoaded ? 'LOADED & ACTIVE' : 'NO MODEL LOADED'}
              </span>
            </div>
          </div>

          {errorMsg && (
            <div className="p-3 rounded bg-[#ff007f]/10 border border-[#ff007f]/40 text-xs text-[#ff007f] flex items-center gap-2">
              <AlertCircle className="w-4 h-4 shrink-0" />
              <span>{errorMsg}</span>
            </div>
          )}

          {/* Tabs for Model Types */}
          <div className="flex border-b border-white/10 gap-2">
            <button
              onClick={() => setSelectedTab('BUILTIN')}
              className={`pb-2.5 px-3 text-xs font-semibold tracking-wider transition-colors border-b-2 -mb-px ${
                selectedTab === 'BUILTIN'
                  ? 'border-[#00ff88] text-[#00ff88]'
                  : 'border-transparent text-white/50 hover:text-white'
              }`}
            >
              STANDARD MODELS
            </button>
            <button
              onClick={() => setSelectedTab('CUSTOM_TEST')}
              className={`pb-2.5 px-3 text-xs font-semibold tracking-wider transition-colors border-b-2 -mb-px ${
                selectedTab === 'CUSTOM_TEST'
                  ? 'border-[#d500f9] text-[#d500f9]'
                  : 'border-transparent text-white/50 hover:text-white'
              }`}
            >
              TEST MODEL (custom/NeuralTrailCNN)
            </button>
            <button
              onClick={() => setSelectedTab('UPLOAD')}
              className={`pb-2.5 px-3 text-xs font-semibold tracking-wider transition-colors border-b-2 -mb-px ${
                selectedTab === 'UPLOAD'
                  ? 'border-[#00f0ff] text-[#00f0ff]'
                  : 'border-transparent text-white/50 hover:text-white'
              }`}
            >
              UPLOAD YOUR PYTORCH MODEL
            </button>
          </div>

          {/* Tab 1: Standard Torchvision Models */}
          {selectedTab === 'BUILTIN' && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* ResNet18 Card */}
              <div className="p-4 rounded-lg bg-white/[0.02] border border-white/10 hover:border-[#00ff88]/50 transition-all flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-display font-bold text-sm text-white">ResNet-18</span>
                    <span className="text-[10px] text-[#00ff88] px-2 py-0.5 rounded bg-[#00ff88]/10 border border-[#00ff88]/30">
                      TORCHVISION
                    </span>
                  </div>
                  <p className="text-xs text-white/60 font-sans">
                    Deep Residual Convolutional Network (18 Layers). ImageNet pre-trained weights with 512-dim penultimate embeddings.
                  </p>
                  <div className="text-[10px] text-white/40 space-y-0.5 pt-1">
                    <div>Input Resolution: 3 × 224 × 224</div>
                    <div>Parameters: 11,689,512</div>
                  </div>
                </div>

                <button
                  onClick={() => handleSelectBuiltin('resnet18')}
                  disabled={loadingModel !== null}
                  className="mt-4 vapor-btn-green w-full py-2 text-xs uppercase font-bold flex items-center justify-center gap-2"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{loadingModel === 'resnet18' ? 'Loading...' : 'Load ResNet-18'}</span>
                </button>
              </div>

              {/* MobileNet Card */}
              <div className="p-4 rounded-lg bg-white/[0.02] border border-white/10 hover:border-[#00f0ff]/50 transition-all flex flex-col justify-between">
                <div className="space-y-2">
                  <div className="flex items-center justify-between">
                    <span className="font-display font-bold text-sm text-white">MobileNet-V2</span>
                    <span className="text-[10px] text-[#00f0ff] px-2 py-0.5 rounded bg-[#00f0ff]/10 border border-[#00f0ff]/30">
                      LIGHTWEIGHT
                    </span>
                  </div>
                  <p className="text-xs text-white/60 font-sans">
                    Inverted Residual Mobile Architecture. High efficiency on mobile and edge vision tasks with depthwise separable convolutions.
                  </p>
                  <div className="text-[10px] text-white/40 space-y-0.5 pt-1">
                    <div>Input Resolution: 3 × 224 × 224</div>
                    <div>Parameters: 3,504,872</div>
                  </div>
                </div>

                <button
                  onClick={() => handleSelectBuiltin('mobilenet')}
                  disabled={loadingModel !== null}
                  className="mt-4 w-full py-2 rounded text-xs uppercase font-bold bg-[#00f0ff]/10 hover:bg-[#00f0ff]/20 text-[#00f0ff] border border-[#00f0ff]/40 flex items-center justify-center gap-2 transition-all"
                >
                  <Play className="w-3.5 h-3.5 fill-current" />
                  <span>{loadingModel === 'mobilenet' ? 'Loading...' : 'Load MobileNet'}</span>
                </button>
              </div>
            </div>
          )}

          {/* Tab 2: Custom Test Model (NeuralTrailCNN from custom/) */}
          {selectedTab === 'CUSTOM_TEST' && (
            <div className="p-4 rounded-lg bg-white/[0.02] border border-[#d500f9]/30 space-y-4">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <FolderCode className="w-4 h-4 text-[#d500f9]" />
                  <span className="font-display font-bold text-sm text-white">NeuralTrailCNN Test Model</span>
                </div>
                <span className="text-[10px] text-[#d500f9] px-2 py-0.5 rounded bg-[#d500f9]/10 border border-[#d500f9]/30">
                  custom/ test suite
                </span>
              </div>

              <p className="text-xs text-white/70 font-sans leading-relaxed">
                Validates the Generic PyTorch Adapter pipeline using the repository's test files:
                <code className="text-[#00ff88] mx-1">custom/model.py</code>,
                <code className="text-[#00ff88] mx-1">custom/neural_trail_cnn.pth</code>, and
                <code className="text-[#00ff88] mx-1">custom/dataset.py</code>.
              </p>

              <div className="grid grid-cols-2 gap-3 text-xs bg-black/40 p-3 rounded border border-white/10 font-mono">
                <div>
                  <span className="text-white/40 block text-[10px]">CLASSES</span>
                  <span className="text-white">circle, square, triangle, cross</span>
                </div>
                <div>
                  <span className="text-white/40 block text-[10px]">INPUT SHAPE</span>
                  <span className="text-white">3 × 32 × 32</span>
                </div>
              </div>

              <button
                onClick={handleSelectCustomTest}
                disabled={loadingModel !== null}
                className="w-full py-2.5 rounded text-xs uppercase font-bold bg-[#d500f9] hover:bg-[#b000d1] text-white flex items-center justify-center gap-2 transition-all shadow-neon-pink"
              >
                <Play className="w-3.5 h-3.5 fill-current" />
                <span>{loadingModel === 'custom_test' ? 'Evaluating Custom Model...' : 'Load & Evaluate NeuralTrailCNN'}</span>
              </button>
            </div>
          )}

          {/* Tab 3: Upload Your PyTorch Model */}
          {selectedTab === 'UPLOAD' && (
            <form onSubmit={handleUploadSubmit} className="space-y-4 text-xs font-mono">
              <div className="space-y-1.5">
                <label className="text-white/70 block font-semibold">1. WEIGHT CHECKPOINT (.pth or .pt) *</label>
                <input
                  type="file"
                  accept=".pt,.pth"
                  onChange={(e) => setWeightFile(e.target.files?.[0] || null)}
                  className="w-full p-2 bg-black/50 border border-white/20 rounded text-white file:mr-3 file:py-1 file:px-2 file:rounded file:border-0 file:bg-white/10 file:text-white file:font-mono file:text-xs hover:border-[#00f0ff]"
                  required
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div className="space-y-1.5">
                  <label className="text-white/70 block font-semibold">INPUT SHAPE (C, H, W)</label>
                  <input
                    type="text"
                    value={inputShape}
                    onChange={(e) => setInputShape(e.target.value)}
                    placeholder="3, 224, 224"
                    className="w-full p-2 bg-black/50 border border-white/20 rounded text-white focus:outline-none focus:border-[#00f0ff]"
                  />
                </div>

                <div className="space-y-1.5">
                  <label className="text-white/70 block font-semibold">CLASS NAMES (COMMA-SEPARATED)</label>
                  <input
                    type="text"
                    value={classesList}
                    onChange={(e) => setClassesList(e.target.value)}
                    placeholder="cat, dog, car, truck"
                    className="w-full p-2 bg-black/50 border border-white/20 rounded text-white focus:outline-none focus:border-[#00f0ff]"
                  />
                </div>
              </div>

              <div className="space-y-1.5">
                <label className="text-white/70 block font-semibold">
                  2. OPTIONAL PYTHON ARCHITECTURE CODE (if state_dict only)
                </label>
                <textarea
                  rows={4}
                  value={modelCode}
                  onChange={(e) => setModelCode(e.target.value)}
                  placeholder={`class MyNet(nn.Module):\n    def __init__(self, num_classes=10):\n        super().__init__()\n        self.features = ...\n    def forward(self, x):\n        ...`}
                  className="w-full p-2 bg-black/50 border border-white/20 rounded text-white font-mono text-xs focus:outline-none focus:border-[#00f0ff]"
                />
              </div>

              <button
                type="submit"
                disabled={loadingModel !== null || !weightFile}
                className="w-full py-2.5 rounded text-xs uppercase font-bold bg-[#00f0ff] hover:bg-[#00c8d6] text-black flex items-center justify-center gap-2 transition-all disabled:opacity-50"
              >
                <Upload className="w-3.5 h-3.5" />
                <span>{loadingModel === 'upload' ? 'Loading & Validating...' : 'Load & Initialize PyTorch Model'}</span>
              </button>
            </form>
          )}
        </div>
      </div>
    </div>
  );
};
