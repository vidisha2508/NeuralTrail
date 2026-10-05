import math
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans


class BlindSpotDetector:
    """
    Unsupervised detection of failure clusters / blind spots in deep representation space.
    Analyzes real model inference outputs, identifies high-error density regions,
    and derives actionable diagnostic insights.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def detect(
        self,
        samples: List[Dict[str, Any]],
        coords_2d: np.ndarray,
        embeddings: Optional[np.ndarray] = None,
        k_clusters: Optional[int] = None,
    ) -> Tuple[List[Dict[str, Any]], List[int]]:
        """
        Clusters samples and identifies failure-prone regions.

        Returns:
            blind_spots: List of structured BlindSpot dictionaries.
            cluster_assignments: Cluster ID index per sample.
        """
        n_samples = len(samples)
        if n_samples == 0:
            return [], []

        # Determine reasonable number of clusters based on sample count
        if k_clusters is None:
            k = max(3, min(8, n_samples // 12))
        else:
            k = max(2, min(k_clusters, n_samples))

        # We cluster using the 2D projection or embedding space
        features_to_cluster = coords_2d
        kmeans = KMeans(n_clusters=k, random_state=self.random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(features_to_cluster)

        clusters_data: List[Dict[str, Any]] = []

        for c_idx in range(k):
            indices = np.where(cluster_labels == c_idx)[0]
            if len(indices) == 0:
                continue

            c_samples = [samples[i] for i in indices]
            c_coords = coords_2d[indices]

            sample_count = len(c_samples)
            correct_count = sum(1 for s in c_samples if s["correct"])
            failure_count = sample_count - correct_count

            accuracy = float(correct_count) / float(sample_count)
            failure_rate = float(failure_count) / float(sample_count)

            confidences = [s["confidence"] for s in c_samples]
            avg_confidence = float(np.mean(confidences)) if confidences else 0.0

            center_x = float(np.mean(c_coords[:, 0]))
            center_y = float(np.mean(c_coords[:, 1]))

            # Compute radius (90th percentile distance from center)
            dists = np.sqrt((c_coords[:, 0] - center_x) ** 2 + (c_coords[:, 1] - center_y) ** 2)
            radius = float(np.percentile(dists, 90)) if len(dists) > 1 else 15.0
            radius = max(8.0, min(35.0, radius))

            # Analyze error patterns & class confusion
            confusions = []
            perturbation_counts = Counter()
            true_class_counts = Counter()

            for s in c_samples:
                true_class_counts[s.get("true_class_name", "unknown")] += 1
                perturbation_counts[s.get("perturbation", "none")] += 1
                if not s["correct"]:
                    confusions.append(f"{s.get('true_class_name', '')} -> {s.get('predicted_label', '')}")

            dominant_class = true_class_counts.most_common(1)[0][0] if true_class_counts else "General"
            dominant_perturbation = perturbation_counts.most_common(1)[0][0] if perturbation_counts else "clean"

            if confusions:
                common_error, err_count = Counter(confusions).most_common(1)[0]
                common_error_str = f"{common_error} ({err_count} samples)"
            else:
                common_error_str = "No frequent error (high stability)"

            # Severity calculation
            if failure_rate >= 0.50:
                severity = "HIGH"
            elif failure_rate >= 0.25:
                severity = "MEDIUM"
            else:
                severity = "LOW"

            # Diagnostic descriptive name
            if confusions:
                most_common_pair = Counter(confusions).most_common(1)[0][0]
                parts = most_common_pair.split(" -> ")
                from_c = parts[0].strip().title()
                to_c = parts[1].strip().title() if len(parts) > 1 else ""
                name = f"{from_c} / {to_c} Boundary" if to_c else f"{from_c} Error Zone"
            else:
                name = f"{dominant_class.title()} Manifold Cluster"

            # Diagnostic description based on REAL model outputs
            if failure_count > 0:
                desc = (
                    f"Cluster of {sample_count} samples shows a {failure_rate * 100:.1f}% failure rate "
                    f"(accuracy: {accuracy * 100:.1f}%). Primary confusion observed: {common_error_str}. "
                    f"Prominently affected under '{dominant_perturbation}' perturbation."
                )
            else:
                desc = (
                    f"Cluster of {sample_count} samples demonstrates 100% accuracy and high stability "
                    f"with average confidence of {avg_confidence * 100:.1f}%."
                )

            # Actionable mitigation suggestions based on perturbation & confusion
            if "blur" in dominant_perturbation:
                mitigation = (
                    "Apply multi-scale Gaussian blur augmentation and edge-preserving loss regularization "
                    "to maintain high-frequency discriminative features."
                )
            elif "rotation" in dominant_perturbation:
                mitigation = (
                    "Incorporate random rotation augmentation (±30° to ±90°) and test-time rotation averaging "
                    "to improve spatial invariance."
                )
            elif "contrast" in dominant_perturbation or "lighting" in dominant_perturbation:
                mitigation = (
                    "Add dynamic color jittering, histogram equalization, and adaptive gamma normalization "
                    "in the preprocessing pipeline."
                )
            elif "noise" in dominant_perturbation:
                mitigation = (
                    "Inject Gaussian and salt-and-pepper noise into training batches; evaluate bilateral filter preconditioning."
                )
            elif failure_count > 0:
                mitigation = (
                    f"Fine-tune with contrastive margin loss focusing on hard negatives between "
                    f"{dominant_class} and adjacent manifold classes."
                )
            else:
                mitigation = "Baseline representation is robust; monitor for distribution shift."

            # Find representative samples (closest to center + worst misclassifications)
            sorted_by_dist = np.argsort(dists)
            central_sample_ids = [c_samples[idx]["id"] for idx in sorted_by_dist[:4]]

            # Failure exemplars
            failure_samples = [s for s in c_samples if not s["correct"]]
            # Sort failure samples by high confidence (confident errors are most dangerous)
            failure_samples.sort(key=lambda s: s["confidence"], reverse=True)
            failure_sample_ids = [s["id"] for s in failure_samples[:4]]

            representative_ids = list(dict.fromkeys(failure_sample_ids + central_sample_ids))[:6]

            cluster_id = f"cluster-{c_idx}"
            cluster_num = f"#{c_idx + 1:02d}"

            clusters_data.append({
                "id": cluster_id,
                "cluster_number": cluster_num,
                "name": name,
                "severity": severity,
                "sample_count": sample_count,
                "correct_count": correct_count,
                "failure_count": failure_count,
                "accuracy": round(accuracy * 100.0, 2),
                "failure_rate": round(failure_rate, 4),
                "average_confidence": round(avg_confidence, 4),
                "common_prediction_error": common_error_str,
                "dominant_class": dominant_class,
                "sensitivity": dominant_perturbation.replace("_", " ").title(),
                "cluster_center": {"x": round(center_x, 2), "y": round(center_y, 2)},
                "radius": round(radius, 2),
                "description": desc,
                "mitigation_suggestion": mitigation,
                "representative_sample_ids": representative_ids,
            })

        # Sort clusters by failure_rate descending, then sample_count descending
        clusters_data.sort(key=lambda c: (c["failure_rate"], c["sample_count"]), reverse=True)

        return clusters_data, cluster_labels.tolist()
