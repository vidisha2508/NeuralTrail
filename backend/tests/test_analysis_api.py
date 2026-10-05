import sys
from fastapi.testclient import TestClient

from app.main import app


def test_analysis_api_suite():
    print("\n=======================================================")
    print("       NEURAL TRAIL ANALYSIS ENGINE API TEST SUITE     ")
    print("=======================================================")

    with TestClient(app) as client:
        # 1. Health check includes standby status
        print("\n[TEST 1] GET /api/health (initial standby)")
        resp = client.get("/api/health")
        assert resp.status_code == 200
        health_data = resp.json()
        print(f"         Health: {health_data}")
        assert health_data["status"] in ["ok", "healthy"]

        # Load ResNet-18 model
        print("\n[TEST 1b] POST /api/model/load (loading resnet18)")
        load_resp = client.post("/api/model/load", json={"model_type": "resnet18"})
        assert load_resp.status_code == 200

        resp = client.get("/api/health")
        health_data = resp.json()
        assert health_data["model_loaded"] is True

        # 2. Check Analysis Status
        print("\n[TEST 2] GET /api/analysis/status")
        resp = client.get("/api/analysis/status")
        assert resp.status_code == 200
        status_data = resp.json()
        print(f"         Status: stage={status_data['stage']}, total_samples={status_data['total_samples']}")
        assert status_data["has_results"] is True
        assert status_data["total_samples"] > 0

        # 3. Check Analysis Summary & Model Metrics
        print("\n[TEST 3] GET /api/analysis/summary")
        resp = client.get("/api/analysis/summary")
        assert resp.status_code == 200
        summary = resp.json()
        print(f"         Dataset: {summary['dataset_name']}")
        print(f"         Total Samples: {summary['total_samples']}")
        print(f"         Reducer: {summary['reducer']}")
        print(f"         Blind Spots Count: {summary['blind_spots_count']}")
        metrics = summary["metrics"]
        print(f"         Model Metrics: Accuracy={metrics['accuracy']}%, Health={metrics['health']} ({metrics['healthStatus']})")
        print(f"         Avg Confidence: {metrics['avgConfidence']:.4f}")
        print(f"         Comparison: {metrics['overallVsBlindSpotComparison']['annotation']}")
        assert metrics["totalSamples"] == summary["total_samples"]
        assert "overallAccuracy" in metrics["overallVsBlindSpotComparison"]
        assert "blindSpotAccuracy" in metrics["overallVsBlindSpotComparison"]

        # 4. Check Samples & 2D Embedding Representation
        print("\n[TEST 4] GET /api/analysis/samples")
        resp = client.get("/api/analysis/samples?limit=10")
        assert resp.status_code == 200
        samples_data = resp.json()
        print(f"         Total Samples Available: {samples_data['total']}")
        print(f"         Returned: {len(samples_data['samples'])}")
        assert len(samples_data["samples"]) > 0

        s0 = samples_data["samples"][0]
        print(f"         Sample 0: ID={s0['id']}, 2D coords=({s0['x']}, {s0['y']})")
        print(f"         True: {s0['trueClassName']} (idx {s0['trueClass']}) | Pred: {s0['predictedClassName']} (idx {s0['predictedClass']})")
        print(f"         Confidence: {s0['confidence']}, Correct: {s0['correct']}, Stability: {s0['stability']}")
        assert "id" in s0
        assert "x" in s0 and "y" in s0
        assert "trueClass" in s0 and "predictedClass" in s0
        assert "confidence" in s0
        assert "correct" in s0
        assert "stability" in s0
        assert "clusterId" in s0

        # Test filtering by status: 'failure'
        resp_failures = client.get("/api/analysis/samples?status=failure")
        assert resp_failures.status_code == 200
        failures = resp_failures.json()["samples"]
        print(f"         Filtered Failures Count: {len(failures)}")
        for f in failures[:3]:
            assert f["correct"] is False
            assert f["stability"] == "failure"

        # 5. Check Detected Blind Spots / High-Failure Clusters
        print("\n[TEST 5] GET /api/analysis/blind-spots")
        resp = client.get("/api/analysis/blind-spots")
        assert resp.status_code == 200
        blind_spots = resp.json()
        print(f"         Detected Blind Spots / Clusters: {len(blind_spots)}")
        assert len(blind_spots) > 0

        first_spot = blind_spots[0]
        print(f"         Cluster {first_spot['cluster_number']} ({first_spot['id']}): '{first_spot['name']}'")
        print(f"         Severity: {first_spot['severity']} | Samples: {first_spot['sample_count']}")
        print(f"         Accuracy: {first_spot['accuracy']}% | Failure Rate: {first_spot['failure_rate']}")
        print(f"         Avg Confidence: {first_spot['average_confidence']}")
        print(f"         Common Error: {first_spot['common_prediction_error']}")
        print(f"         Center: {first_spot['cluster_center']}, Radius: {first_spot['radius']}")
        print(f"         Description: {first_spot['description']}")
        print(f"         Mitigation: {first_spot['mitigation_suggestion']}")

        assert "id" in first_spot
        assert "sample_count" in first_spot
        assert "accuracy" in first_spot
        assert "failure_rate" in first_spot
        assert "average_confidence" in first_spot
        assert "common_prediction_error" in first_spot
        assert "severity" in first_spot
        assert "cluster_center" in first_spot
        assert "x" in first_spot["cluster_center"] and "y" in first_spot["cluster_center"]

        # 6. Check Individual Blind-Spot Details
        print(f"\n[TEST 6] GET /api/analysis/blind-spots/{first_spot['id']}")
        resp = client.get(f"/api/analysis/blind-spots/{first_spot['id']}")
        assert resp.status_code == 200
        detail = resp.json()
        print(f"         Detail Cluster: {detail['name']}")
        print(f"         Representative Samples Count: {len(detail['representative_samples'])}")
        print(f"         Full Cluster Samples Count: {len(detail['samples'])}")
        print(f"         Error Breakdown: {detail['error_distribution']}")
        assert len(detail["representative_samples"]) > 0
        assert len(detail["samples"]) == detail["sample_count"]

        # 7. Test Triggering a Fresh Analysis Run via POST /api/analysis/start
        print("\n[TEST 7] POST /api/analysis/start (run with 40 samples, PCA)")
        start_payload = {
            "dataset_source": "benchmark",
            "sample_count": 40,
            "reducer_type": "pca",
            "async_run": False,
        }
        resp = client.post("/api/analysis/start", json=start_payload)
        assert resp.status_code == 200
        run_res = resp.json()
        print(f"         Start result: {run_res['message']}")
        assert run_res["success"] is True

        # Verify summary updated
        resp = client.get("/api/analysis/summary")
        summary_after = resp.json()
        print(f"         New total samples evaluated: {summary_after['total_samples']}")
        assert summary_after["total_samples"] == 40

        # 8. Test Activation Pathway Visualization (5 Stages)
        print("\n[TEST 8] GET /api/analysis/xray/pathway (Activation Pathway Visualization)")
        resp = client.get("/api/analysis/xray/pathway")
        assert resp.status_code == 200
        pathway = resp.json()
        print(f"         Feature Name: {pathway['feature_name']}")
        print(f"         Sample A (Canonical): {pathway['sample_a']['id']} ({pathway['sample_a']['trueClassName']})")
        print(f"         Sample B (Degraded): {pathway['sample_b']['id']} ({pathway['sample_b']['trueClassName']} -> {pathway['sample_b']['predictedClassName']})")
        print(f"         5 Stages Extracted:")
        expected_stages = ["INPUT", "EARLY", "MIDDLE", "DEEP", "OUTPUT"]
        extracted_stages = [l["stage"] for l in pathway["layers"]]
        assert extracted_stages == expected_stages
        for l in pathway["layers"]:
            print(f"          - 0{extracted_stages.index(l['stage'])+1}. {l['stage']} ({l['name']}): Anomaly={l['anomalyScore']*100:.0f}%, Shape={l['tensorShape']}")
            assert len(l["normalFeatureMap"]) == 6
            assert len(l["normalFeatureMap"][0]) == 6
            assert len(l["perturbedFeatureMap"]) == 6
            assert len(l["differenceMap"]) == 6
        assert "causality_disclaimer" in pathway
        print(f"         Disclaimer: {pathway['causality_disclaimer']}")

        # 9. Test Activation Pathway with Sample Comparison & Layer Selection
        print("\n[TEST 9] POST /api/analysis/xray/compare")
        compare_payload = {
            "sample_a_id": pathway["sample_a"]["id"],
            "sample_b_id": pathway["sample_b"]["id"],
            "selected_layer_id": "layer-deep",
        }
        resp = client.post("/api/analysis/xray/compare", json=compare_payload)
        assert resp.status_code == 200
        comp_res = resp.json()
        print(f"         Selected Layer: {comp_res['selected_layer_id']}")
        assert comp_res["selected_layer_id"] == "layer-deep"
        assert len(comp_res["layers"]) == 5

        # 10. Test What-If Candidate Samples
        print("\n[TEST 10] GET /api/analysis/whatif/samples")
        resp = client.get("/api/analysis/whatif/samples")
        assert resp.status_code == 200
        candidates = resp.json()
        print(f"         Available Candidates: {len(candidates)}")
        assert len(candidates) > 0

        # 11. Test Real-Time What-If Stress Perturbation Experiment
        print("\n[TEST 11] POST /api/analysis/whatif/experiment (rotation, noise, blur, crop, brightness, occlusion)")
        exp_payload = {
            "sample_id": candidates[0]["id"],
            "rotation": 25.0,
            "noise": 20.0,
            "blur": 3.0,
            "crop": 15.0,
            "brightness": -15.0,
            "occlusion": 20.0,
            "include_images": True,
        }
        resp = client.post("/api/analysis/whatif/experiment", json=exp_payload)
        assert resp.status_code == 200
        exp_res = resp.json()
        print(f"         Original: '{exp_res['original_prediction']}' ({exp_res['original_confidence_percentage']}%)")
        print(f"         Perturbed: '{exp_res['perturbed_prediction']}' ({exp_res['perturbed_confidence_percentage']}%)")
        print(f"         Confidence Change: {exp_res['confidence_change_percentage']}%")
        print(f"         Prediction Flip: {exp_res['prediction_flip']}")
        print(f"         Robustness Score: {exp_res['robustness_score']} / 100")
        print(f"         Inference Latency: {exp_res['latency_ms']} ms")
        assert "original_prediction" in exp_res
        assert "perturbed_prediction" in exp_res
        assert "confidence_change" in exp_res
        assert "prediction_flip" in exp_res
        assert "robustness_score" in exp_res
        assert 0 <= exp_res["robustness_score"] <= 100
        assert exp_res["original_image_url"].startswith("data:image/")
        assert exp_res["perturbed_image_url"].startswith("data:image/")

    print("\n=======================================================")
    print("    ALL ANALYSIS ENGINE, X-RAY & WHAT-IF CHECKS PASSED!")
    print("=======================================================")


if __name__ == "__main__":
    test_analysis_api_suite()


