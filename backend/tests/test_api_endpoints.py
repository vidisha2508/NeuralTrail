import io
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from app.main import app
from app.config import settings

def make_sample_image() -> bytes:
    img = Image.new("RGB", (224, 224), color=(30, 80, 150))
    d = ImageDraw.Draw(img)
    d.ellipse([50, 50, 174, 174], fill=(240, 200, 50))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()

def test_api_suite():
    print("\n==================================================")
    print("       FASTAPI TEST CLIENT ENDPOINT SUITE         ")
    print("==================================================")

    with TestClient(app) as client:
        # 1. Health Endpoint
        print("\n[API 1] GET /api/health")
        resp = client.get("/api/health")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        print(f"        Response: {data}")
        assert data["status"] == "healthy"
        assert data["model_loaded"] is True

        # 2. Model Info Endpoint
        print("\n[API 2] GET /api/model/info")
        resp = client.get("/api/model/info")
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        meta = resp.json()
        print(f"        Model: {meta['model_name']} | Architecture: {meta['architecture']}")
        print(f"        Device: {meta['device']} | Checkpoint: {meta['checkpoint_path']}")
        assert meta["model_name"] == "ResNet18"
        assert meta["total_parameters"] > 11_000_000

        # 3. Model Predict Endpoint (Multipart Form Data Upload)
        print("\n[API 3] POST /api/model/predict")
        img_bytes = make_sample_image()
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        resp = client.post("/api/model/predict?top_k=5", files=files)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        pred = resp.json()
        print(f"        Top Label: {pred['top_label']} (index: {pred['top_index']})")
        print(f"        Confidence: {pred['confidence']:.4f}")
        print(f"        Inference Latency: {pred['inference_time_ms']} ms")
        print(f"        Predictions Count: {len(pred['predictions'])}")
        assert pred["confidence"] > 0.0
        assert len(pred["predictions"]) == 5

        # 4. Latent Feature Embedding Endpoint
        print("\n[API 4] POST /api/model/embedding")
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        resp = client.post("/api/model/embedding", files=files)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        emb = resp.json()
        print(f"        Embedding Dimension: {emb['dimension']}")
        print(f"        Vector Norm: {emb['norm']}")
        assert emb["dimension"] == 512
        assert len(emb["embedding"]) == 512

        # 5. Layer Activations Endpoint
        print("\n[API 5] POST /api/model/activations")
        files = {"file": ("sample.jpg", img_bytes, "image/jpeg")}
        resp = client.post("/api/model/activations", files=files)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        act = resp.json()
        print(f"        Extracted Layers: {[l['layer_name'] for l in act['layers']]}")
        assert len(act["layers"]) == 4

        # 6. Model Reload Endpoint
        print("\n[API 6] POST /api/model/load")
        load_payload = {
            "checkpoint_path": settings.MODEL_CHECKPOINT_PATH,
            "model_type": "resnet18",
            "device": "cpu"
        }
        resp = client.post("/api/model/load", json=load_payload)
        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        load_data = resp.json()
        print(f"        Reload status: {load_data['message']}")
        assert load_data["success"] is True

    print("\n==================================================")
    print("      ALL API TEST CLIENT CHECKS COMPLETED        ")
    print("==================================================")

if __name__ == "__main__":
    test_api_suite()
