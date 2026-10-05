from typing import List, Dict, Any
import numpy as np


class MetricsCalculator:
    """
    Computes diagnostic health scores and aggregate metrics for ResNet-18
    across real dataset inference and identified blind spots.
    """

    @staticmethod
    def calculate(
        samples: List[Dict[str, Any]],
        blind_spots: List[Dict[str, Any]],
        model_name: str = "ResNet18",
        architecture: str = "Deep Residual Convolutional Network (18 Layers)",
        dataset_name: str = "ImageNet-1K Benchmark Suite",
    ) -> Dict[str, Any]:
        total_samples = len(samples)
        if total_samples == 0:
            return {
                "name": model_name,
                "architecture": architecture,
                "dataset": dataset_name,
                "health": 100,
                "healthStatus": "HEALTHY",
                "accuracy": 0.0,
                "accuracyDelta": "0.0% vs baseline",
                "avgConfidence": 0.0,
                "blindSpotCount": 0,
                "highSeverityCount": 0,
                "totalSamples": 0,
                "overallVsBlindSpotComparison": {
                    "overallAccuracy": 0.0,
                    "blindSpotAccuracy": 0.0,
                    "annotation": "No samples evaluated.",
                },
            }

        correct_count = sum(1 for s in samples if s["correct"])
        overall_accuracy = (float(correct_count) / float(total_samples)) * 100.0

        confidences = [s["confidence"] for s in samples]
        avg_confidence = float(np.mean(confidences))

        # Filter clusters that are true blind spots (failure_rate >= 0.20 or severity != 'LOW')
        flagged_blind_spots = [b for b in blind_spots if b["failure_rate"] >= 0.20]
        blind_spot_count = len(flagged_blind_spots)
        high_severity_count = sum(1 for b in flagged_blind_spots if b["severity"] == "HIGH")

        # Blind spot accuracy vs overall accuracy
        if flagged_blind_spots:
            bs_accuracies = [b["accuracy"] for b in flagged_blind_spots]
            blind_spot_accuracy = float(np.mean(bs_accuracies))
            drop = overall_accuracy - blind_spot_accuracy
            annotation = (
                f"Model accuracy drops by {drop:.1f}% inside the {blind_spot_count} identified failure regions "
                f"compared to {overall_accuracy:.1f}% global accuracy."
            )
        else:
            blind_spot_accuracy = overall_accuracy
            annotation = f"High stability across all representation clusters (global accuracy: {overall_accuracy:.1f}%)."

        # Health score (0 - 100):
        # Base is overall accuracy, penalized by high severity failure clusters and low confidence
        penalty = (high_severity_count * 12.0) + (max(0, blind_spot_count - 1) * 5.0)
        health_score = max(5.0, min(100.0, overall_accuracy - penalty))

        if health_score >= 80.0:
            health_status = "HEALTHY"
        elif health_score >= 50.0:
            health_status = "DEGRADED"
        else:
            health_status = "CRITICAL"

        accuracy_delta = f"{(overall_accuracy - 85.0):+.1f}% vs baseline"

        return {
            "name": model_name,
            "architecture": architecture,
            "dataset": dataset_name,
            "health": round(health_score, 1),
            "healthStatus": health_status,
            "accuracy": round(overall_accuracy, 2),
            "accuracyDelta": accuracy_delta,
            "avgConfidence": round(avg_confidence, 4),
            "blindSpotCount": blind_spot_count,
            "highSeverityCount": high_severity_count,
            "totalSamples": total_samples,
            "overallVsBlindSpotComparison": {
                "overallAccuracy": round(overall_accuracy, 2),
                "blindSpotAccuracy": round(blind_spot_accuracy, 2),
                "annotation": annotation,
            },
        }
