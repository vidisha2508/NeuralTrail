import { GemmaInvestigation } from '../types/neuralTrail';

export const mockGemmaInvestigations: Record<string, GemmaInvestigation> = {
  'bs-01': {
    clusterId: 'bs-01',
    clusterName: 'Rotated Inputs (#01)',
    observation: 'Model confidence drops from 94.2% to 43.2% when input tensors exhibit rotation >12.4°. Stroke connectivity dissolves in residual layer conv2_x.',
    hypothesis: 'The network relies on fixed absolute horizontal stroke coordinates rather than invariant topological loop enclosures. As orientation skews, the left boundary loop of digit 8 falls into the decision attractor for numeral 3.',
    experiment: 'Synthesize ±25° continuous affine rotation perturbations with spatial transformer network (STN) regularization applied to layer2_conv.',
    expectedSignal: '+22.4% local accuracy recovery on rotated subset without inducing class drift or degradation in baseline canonical classes.',
    result: 'Simulated trial achieved 89.2% accuracy on cluster #01. Decision boundary hysteresis stabilized. No regression on classes 0-7.',
    suggestedPatch: 'Sequential(RandomRotation(degrees=(-25, 25)), ColorJitter(brightness=0.1))'
  },
  'bs-02': {
    clusterId: 'bs-02',
    clusterName: 'Low-light Inputs (#02)',
    observation: 'Luminance attenuation below 0.35 leads to 41.9% of activations in conv1 entering dead ReLU saturation.',
    hypothesis: 'Early bias terms lack dynamic headroom for dark background suppression, allowing noise floor artifacts to mimic circular strokes of digit 6.',
    experiment: 'Apply adaptive histogram normalization and introduce synthetic Poisson contrast jitter to 15% of training batches.',
    expectedSignal: 'Sparsity in conv1 activations drops from 68% to 19% under low-contrast illumination.',
    result: 'Accuracy on low-light inputs recovered from 74.2% to 92.6%. False transition 5→6 eliminated.',
    suggestedPatch: 'transforms.RandomEqualize(p=0.4)'
  },
  'bs-03': {
    clusterId: 'bs-03',
    clusterName: 'Class 8/3 Confusion (#03)',
    observation: 'Scale downsampling (-20%) narrows bridge stroke width below receptive field kernel threshold (3x3).',
    hypothesis: 'The model fails to differentiate between open and closed curvature topologies when line thickness is sub-pixel quantized.',
    experiment: 'Inject hard-negative triplet loss penalty explicitly between class 8 and class 3 representations in penultimate feature space.',
    expectedSignal: 'Cosine distance between Class 8 and Class 3 centroids increases from 0.18 to > 0.65 in 512-dim embedding.',
    result: 'Cosine distance reached 0.72. Confusion rate dropped from 31.6% down to 2.8%.',
    suggestedPatch: 'losses.TripletMarginLoss(margin=1.0, p=2)'
  },
  'default': {
    clusterId: 'global',
    clusterName: 'Global Manifold Telemetry',
    observation: 'Aggregate accuracy of 98.2% masks three localized manifold failures in specific affine transformation subspaces.',
    hypothesis: 'Convolutional feature kernels have learned spurious position-dependent heuristics rather than intrinsic geometric topological invariants.',
    experiment: 'Trigger full-manifold stress suite across 6 perturbation dimensions and synthesize targeted hard-negative augmentation batches.',
    expectedSignal: 'Identification of minimal counterfactual perturbation vectors that flip predictions with confidence delta > 50%.',
    result: '3 primary blind spot clusters mapped. Automated mitigation protocols ready for deployment.',
    suggestedPatch: 'python train_robustness.py --checkpoint active_model --manifold evaluation_set --augment targeted'
  }
};
