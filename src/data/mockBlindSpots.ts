import { BlindSpot } from '../types/neuralTrail';

export const mockBlindSpots: BlindSpot[] = [
  {
    id: 'bs-01',
    clusterNumber: '#01',
    name: 'Rotated Inputs',
    severity: 'HIGH',
    sampleCount: 1284,
    accuracy: 61.8,
    confidence: 43.2,
    commonConfusion: '8 → 3',
    sensitivity: 'Rotation (+15°)',
    clusterCenter: { x: 740, y: 320 },
    radius: 78,
    description: 'Angular perturbations beyond ±12° break spatial symmetry in feature extraction layers conv2_x and conv3_x, triggering acute classification breakdown.',
    mitigationSuggestion: 'Inject ±25° affine rotation augments during fine-tuning phase with spatial transformer regularization.'
  },
  {
    id: 'bs-02',
    clusterNumber: '#02',
    name: 'Low-light Inputs',
    severity: 'MEDIUM',
    sampleCount: 742,
    accuracy: 74.2,
    confidence: 58.1,
    commonConfusion: '5 → 6',
    sensitivity: 'Contrast (-40%)',
    clusterCenter: { x: 310, y: 680 },
    radius: 65,
    description: 'Luminance compression causes high-frequency stroke edges to sink below ReLU activation thresholds, resulting in severe label drift toward curved loops.',
    mitigationSuggestion: 'Add histogram equalization layer and low-contrast gamma jitter synthesis to pre-processing pipeline.'
  },
  {
    id: 'bs-03',
    clusterNumber: '#03',
    name: 'Class 8/3 Confusion',
    severity: 'HIGH',
    sampleCount: 513,
    accuracy: 68.4,
    confidence: 51.0,
    commonConfusion: '8 → 3',
    sensitivity: 'Scale (-20%)',
    clusterCenter: { x: 820, y: 710 },
    radius: 70,
    description: 'When stroke width narrows or input digit is downscaled, the left enclosure bridge of digit 8 drops feature representation, collapsing into numeral 3.',
    mitigationSuggestion: 'Introduce multi-scale boundary loss penalty and targeted hard-negative mining between classes 8 and 3.'
  }
];
