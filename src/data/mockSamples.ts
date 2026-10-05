import { Sample } from '../types/neuralTrail';

// Deterministic pseudorandom number generator for consistent clusters
function createSeededRandom(seed: number) {
  let s = seed;
  return () => {
    s = (s * 9301 + 49297) % 233280;
    return s / 233280;
  };
}

// Generate stylized 8x8 digit pixel intensity grid for the digit grid panel preview
function generateDigitGrid(digit: number, perturbed: boolean): number[][] {
  const grid: number[][] = Array(8).fill(0).map(() => Array(8).fill(0));
  // Archetypal patterns
  if (digit === 8) {
    for (let r = 1; r < 7; r++) {
      grid[r][2] = 0.8;
      grid[r][5] = 0.8;
    }
    grid[1][3] = 0.9; grid[1][4] = 0.9;
    grid[4][3] = 0.9; grid[4][4] = 0.9;
    grid[6][3] = 0.9; grid[6][4] = 0.9;
    if (perturbed) {
      grid[2][2] = 0.1; // Broken edge causing 8 -> 3 confusion
      grid[3][2] = 0.05;
    }
  } else if (digit === 3) {
    grid[1][2] = 0.9; grid[1][3] = 0.9; grid[1][4] = 0.9; grid[1][5] = 0.9;
    grid[2][5] = 0.9; grid[3][5] = 0.9;
    grid[4][3] = 0.9; grid[4][4] = 0.9; grid[4][5] = 0.9;
    grid[5][5] = 0.9; grid[6][5] = 0.9;
    grid[6][2] = 0.9; grid[6][3] = 0.9; grid[6][4] = 0.9;
  } else if (digit === 5) {
    grid[1][2] = 0.9; grid[1][3] = 0.9; grid[1][4] = 0.9; grid[1][5] = 0.9;
    grid[2][2] = 0.9; grid[3][2] = 0.9;
    grid[3][3] = 0.9; grid[3][4] = 0.9; grid[3][5] = 0.9;
    grid[4][5] = 0.9; grid[5][5] = 0.9;
    grid[6][2] = 0.9; grid[6][3] = 0.9; grid[6][4] = 0.9;
  } else {
    // Generic recognizable stroke
    for (let i = 1; i < 7; i++) {
      grid[i][4] = 0.9;
    }
    grid[1][3] = 0.7;
    grid[6][3] = 0.8;
    grid[6][5] = 0.8;
  }
  return grid;
}

export function generateMockSamples(): Sample[] {
  const rand = createSeededRandom(42);
  const samples: Sample[] = [];
  const TOTAL_COUNT = 1842;

  // Spatial cluster definitions for classes 0 to 9 plus failure clusters
  const classCenters: Record<number, { x: number; y: number; r: number }> = {
    0: { x: 180, y: 190, r: 85 },
    1: { x: 380, y: 150, r: 70 },
    2: { x: 580, y: 160, r: 80 },
    3: { x: 740, y: 220, r: 75 }, // Near cluster 1 (Rotated) & cluster 3
    4: { x: 190, y: 440, r: 85 },
    5: { x: 380, y: 430, r: 80 },
    6: { x: 550, y: 460, r: 80 },
    7: { x: 760, y: 450, r: 85 },
    8: { x: 420, y: 690, r: 85 },
    9: { x: 620, y: 680, r: 80 },
  };

  // Specific failure regions
  // #01: Rotated Inputs (Center 740, 320)
  // #02: Low-light Inputs (Center 310, 680)
  // #03: Class 8/3 confusion (Center 820, 710)
  const failureClusters = [
    { id: 'bs-01', center: { x: 740, y: 320 }, radius: 65, trueClass: 8, predClass: 3, sens: 'Rotation (+15°)' },
    { id: 'bs-02', center: { x: 310, y: 680 }, radius: 55, trueClass: 5, predClass: 6, sens: 'Contrast (-40%)' },
    { id: 'bs-03', center: { x: 820, y: 710 }, radius: 60, trueClass: 8, predClass: 3, sens: 'Scale (-20%)' },
  ];

  // Distribution: ~75% stable (1382), ~15% uncertain (276), ~10% failure (184)
  const failureTarget = 184;
  const uncertainTarget = 276;

  let currentId = 1000;

  // 1. Generate Failure points clustered tightly around the 3 failure regions
  for (let i = 0; i < failureTarget; i++) {
    const fCluster = failureClusters[i % failureClusters.length];
    const angle = rand() * Math.PI * 2;
    const dist = Math.pow(rand(), 0.7) * fCluster.radius;
    const x = Math.round(fCluster.center.x + Math.cos(angle) * dist);
    const y = Math.round(fCluster.center.y + Math.sin(angle) * dist);
    
    // Low confidence for failures
    const conf = Math.round((0.38 + rand() * 0.28) * 1000) / 10;

    samples.push({
      id: `SMP-${currentId++}`,
      x,
      y,
      trueClass: fCluster.trueClass,
      predictedClass: fCluster.predClass,
      confidence: conf,
      correct: false,
      stability: 'failure',
      clusterId: fCluster.id,
      sensitivity: fCluster.sens,
      thumbnailGrid: generateDigitGrid(fCluster.trueClass, true),
      features: {
        rotation: fCluster.id === 'bs-01' ? 15.4 : (rand() * 4 - 2),
        brightness: fCluster.id === 'bs-02' ? -42.0 : (rand() * 4 - 2),
        blur: fCluster.id === 'bs-03' ? 2.8 : 0.4,
      }
    });
  }

  // 2. Generate Uncertain points (on cluster peripheries and transition boundaries)
  for (let i = 0; i < uncertainTarget; i++) {
    const classNum = Math.floor(rand() * 10);
    const c = classCenters[classNum];
    const angle = rand() * Math.PI * 2;
    // Outer rim of cluster
    const dist = c.r * (0.85 + rand() * 0.45);
    const x = Math.round(c.x + Math.cos(angle) * dist);
    const y = Math.round(c.y + Math.sin(angle) * dist);

    const isCorrect = rand() > 0.45;
    const predClass = isCorrect ? classNum : ((classNum + Math.floor(rand() * 8) + 1) % 10);
    const conf = Math.round((0.55 + rand() * 0.22) * 1000) / 10;

    // Nearest failure cluster association if close
    let assignedCluster: string | undefined;
    for (const fc of failureClusters) {
      const d = Math.hypot(x - fc.center.x, y - fc.center.y);
      if (d < fc.radius * 1.5) {
        assignedCluster = fc.id;
        break;
      }
    }

    const sensitivities = ['Jitter (+4px)', 'Blur (σ=1.2)', 'Compression (Q=60)', 'Noise (+8dB)'];

    samples.push({
      id: `SMP-${currentId++}`,
      x,
      y,
      trueClass: classNum,
      predictedClass: predClass,
      confidence: conf,
      correct: isCorrect,
      stability: 'uncertain',
      clusterId: assignedCluster,
      sensitivity: sensitivities[Math.floor(rand() * sensitivities.length)],
      thumbnailGrid: generateDigitGrid(classNum, false),
      features: {
        rotation: Math.round((rand() * 10 - 5) * 10) / 10,
        brightness: Math.round((rand() * 20 - 10) * 10) / 10,
        blur: Math.round(rand() * 1.5 * 10) / 10,
      }
    });
  }

  // 3. Generate Stable points (~75% of total) in solid cohesive cluster cores
  const remaining = TOTAL_COUNT - samples.length;
  for (let i = 0; i < remaining; i++) {
    const classNum = Math.floor(rand() * 10);
    const c = classCenters[classNum];
    const angle = rand() * Math.PI * 2;
    const dist = Math.pow(rand(), 0.6) * (c.r * 0.82);
    const x = Math.round(c.x + Math.cos(angle) * dist);
    const y = Math.round(c.y + Math.sin(angle) * dist);

    // High confidence
    const conf = Math.round((0.88 + rand() * 0.119) * 1000) / 10;

    samples.push({
      id: `SMP-${currentId++}`,
      x,
      y,
      trueClass: classNum,
      predictedClass: classNum,
      confidence: conf,
      correct: true,
      stability: 'stable',
      sensitivity: 'Nominal (Invariance > 99%)',
      thumbnailGrid: generateDigitGrid(classNum, false),
      features: {
        rotation: Math.round((rand() * 4 - 2) * 10) / 10,
        brightness: Math.round((rand() * 6 - 3) * 10) / 10,
        blur: 0.1,
      }
    });
  }

  return samples;
}

export const mockSamples = generateMockSamples();
