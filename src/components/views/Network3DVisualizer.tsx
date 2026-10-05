import React, { useRef, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Line, Edges } from '@react-three/drei';
import * as THREE from 'three';
import { ActivationLayer } from '../../types/neuralTrail';

interface Network3DVisualizerProps {
  layers: ActivationLayer[];
  selectedLayerId: string;
  onSelectLayer: (id: string) => void;
  showDifferenceMap: boolean;
}

const LayerNode = ({
  layer,
  position,
  isSelected,
  onClick,
  showDifferenceMap,
}: {
  layer: ActivationLayer;
  position: [number, number, number];
  isSelected: boolean;
  onClick: () => void;
  showDifferenceMap: boolean;
}) => {
  const meshRef = useRef<THREE.Group>(null);
  
  // Extract features
  const features = useMemo(() => {
    let mapData = layer.normalFeatureMap;
    if (showDifferenceMap) {
      // Calculate difference if differenceMap is not available directly
      mapData = (layer as any).differenceMap || layer.normalFeatureMap.map((row, i) => 
        row.map((val, j) => Math.abs(val - layer.perturbedFeatureMap[i][j]))
      );
    } else {
      mapData = layer.perturbedFeatureMap || layer.normalFeatureMap;
    }
    return mapData;
  }, [layer, showDifferenceMap]);

  const gridSize = features.length;
  const size = 3; // Total width/height of the layer representation
  const cellSize = size / gridSize;

  useFrame((state) => {
    if (meshRef.current) {
      if (isSelected) {
        meshRef.current.rotation.y = Math.sin(state.clock.elapsedTime * 0.5) * 0.1;
      } else {
        meshRef.current.rotation.y = 0;
      }
    }
  });

  const baseColor = isSelected ? '#00f0ff' : '#4a2c68';
  const highlightColor = showDifferenceMap ? '#ffb300' : (layer.anomalyScore > 0.5 ? '#ff007f' : '#00ff88');

  return (
    <group position={position} ref={meshRef} onClick={(e) => {
      e.stopPropagation();
      onClick();
    }}>
      {/* Container Plane */}
      <mesh>
        <planeGeometry args={[size + 0.5, size + 0.5]} />
        <meshBasicMaterial color={isSelected ? "#1a0b2e" : "#0d0417"} transparent opacity={0.6} side={THREE.DoubleSide} />
        <Edges scale={1} threshold={15} color={isSelected ? "#00f0ff" : "#4a2c68"} />
      </mesh>

      {/* Grid of Activations */}
      {features.map((row, i) => 
        row.map((val, j) => {
          const x = (j - gridSize / 2 + 0.5) * cellSize;
          const y = (-(i - gridSize / 2) - 0.5) * cellSize;
          const height = Math.max(0.1, val * 2);
          return (
            <mesh key={`${i}-${j}`} position={[x, y, height / 2]}>
              <boxGeometry args={[cellSize * 0.8, cellSize * 0.8, height]} />
              <meshBasicMaterial 
                color={val > 0.2 ? highlightColor : '#221133'} 
                transparent 
                opacity={Math.max(0.3, val)}
                wireframe={val <= 0.2}
              />
            </mesh>
          );
        })
      )}

      {/* Layer Label */}
      <Text
        position={[0, size / 2 + 0.5, 0]}
        fontSize={0.25}
        color={isSelected ? "#ffffff" : "#888888"}
        anchorX="center"
        anchorY="bottom"
      >
        {layer.name}
      </Text>
      <Text
        position={[0, -size / 2 - 0.5, 0]}
        fontSize={0.15}
        color={layer.anomalyScore > 0.5 ? "#ff007f" : "#00ff88"}
        anchorX="center"
        anchorY="top"
      >
        {`DIVERGENCE: ${(layer.anomalyScore * 100).toFixed(0)}%`}
      </Text>
    </group>
  );
};

// Connections between layers
const Connections = ({ positions }: { positions: [number, number, number][] }) => {
  return (
    <group>
      {positions.map((pos, i) => {
        if (i === positions.length - 1) return null;
        const nextPos = positions[i + 1];
        return (
          <Line
            key={`line-${i}`}
            points={[pos, nextPos]}
            color="#ff007f"
            opacity={0.3}
            transparent
            lineWidth={1}
            dashed
          />
        );
      })}
    </group>
  );
};

const CameraController = ({ targetPosition }: { targetPosition: [number, number, number] }) => {
  const { camera } = useThree();
  
  useFrame(() => {
    // Smoothly interpolate camera target and position if needed
    // For now, OrbitControls handles most of the interaction
  });
  
  return null;
};

export const Network3DVisualizer: React.FC<Network3DVisualizerProps> = ({
  layers,
  selectedLayerId,
  onSelectLayer,
  showDifferenceMap,
}) => {
  const layerSpacing = 5;
  const positions: [number, number, number][] = layers.map((_, i) => [
    0,
    0,
    -i * layerSpacing + (layers.length * layerSpacing) / 2
  ]);

  return (
    <div className="w-full h-full bg-[#0a0314] rounded-lg border border-white/10 overflow-hidden relative">
      <Canvas camera={{ position: [6, 4, 10], fov: 45 }}>
        <color attach="background" args={['#0a0314']} />
        <ambientLight intensity={0.5} />
        <pointLight position={[10, 10, 10]} intensity={1} />
        
        <group rotation={[0, -Math.PI / 8, 0]}>
          {layers.map((layer, i) => (
            <LayerNode
              key={layer.id}
              layer={layer}
              position={positions[i]}
              isSelected={layer.id === selectedLayerId}
              onClick={() => onSelectLayer(layer.id)}
              showDifferenceMap={showDifferenceMap}
            />
          ))}
          <Connections positions={positions} />
        </group>

        <OrbitControls 
          enablePan={true}
          enableZoom={true}
          enableRotate={true}
          autoRotate={false}
          autoRotateSpeed={0.5}
        />
        <gridHelper args={[50, 50, '#ff007f', '#221133']} position={[0, -3, 0]} />
        <fog attach="fog" args={['#0a0314', 10, 40]} />
      </Canvas>

      <div className="absolute bottom-4 right-4 pointer-events-none">
        <div className="font-mono text-xs text-white/40 bg-black/50 px-2 py-1 rounded backdrop-blur border border-white/5">
          LEFT CLICK + DRAG to rotate • SCROLL to zoom
        </div>
      </div>
    </div>
  );
};
