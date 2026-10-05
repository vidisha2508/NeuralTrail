from fastapi.testclient import TestClient
from app.main import app

with TestClient(app) as client:
    print("\n--- TEST CUSTOM MODEL (acceptance test) ---")
    res = client.post("/api/model/load-custom-test")
    assert res.status_code == 200, res.text
    print("Custom model loaded:", res.json()["metadata"]["model_name"])

    health = client.get("/api/health").json()
    print("Health:", health)
    assert health["model_loaded"] is True
    assert "NeuralTrail" in health["active_model"]

    summary = client.get("/api/analysis/summary").json()
    metrics = summary["metrics"]
    print(f"Accuracy: {metrics['accuracy']}%, Total samples: {summary['total_samples']}, Blind spots: {summary['blind_spots_count']}")

    evidence = client.get("/api/researcher/evidence").json()
    print("Evidence:", evidence)
    assert "NeuralTrailCNN" in evidence["model"]

    hyp = client.post("/api/researcher/hypothesize", json={"evidence": evidence}).json()
    print("Gemma hypothesis:", hyp["research_output"]["hypothesis"])
    assert "hypothesis" in hyp["research_output"]

    cid = "cluster-0" if summary["blind_spots_count"] > 0 else None
    exp_payload = {"cluster_id": cid, "experiment_protocol": "data_augmentation_rotation"}
    exp_res = client.post("/api/researcher/experiment", json=exp_payload).json()
    print("Experiment result:", exp_res)
    assert "observed_accuracy" in exp_res
    assert "accuracy_delta" in exp_res

    replay_data = client.get("/api/analysis/replay").json()
    events = replay_data.get("events", replay_data) if isinstance(replay_data, dict) else replay_data
    print(f"Replay recorded {len(events)} events")
    for e in events:
        print(f"  - [{e.get('type', 'EVENT')}] {e.get('title')}: {e.get('desc', e.get('summary', ''))}")
    assert len(events) >= 2

    print("\n--- ALL CUSTOM TEST STEPS PASSED SUCCESSFULLY! ---")
