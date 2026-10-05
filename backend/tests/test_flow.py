import os
import io
import time
from pathlib import Path
from PIL import Image, ImageDraw
import numpy as np

def generate_test_image() -> bytes:
    """Generates a synthetic RGB test image with distinct geometric patterns."""
    img = Image.new("RGB", (256, 256), color=(40, 10, 60))
    draw = ImageDraw.Draw(img)
    draw.rectangle([40, 40, 216, 216], outline=(0, 255, 136), width=4)
    draw.ellipse([80, 80, 176, 176], fill=(255, 0, 127), outline=(0, 240, 255), width=3)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def run_tests():
    print("==================================================")
    print("  NEURAL TRAIL ML BACKEND PIPELINE VERIFICATION   ")
    print("==================================================")

    from app.config import settings
    from app.services.model_loader import model_manager
    from app.adapters.resnet18 import ResNet18Adapter
    from app.services.preprocessor import preprocessor

    checkpoint_path = Path(settings.get_checkpoint_path())
    print(f"\n[1] Target Checkpoint Path: {checkpoint_path}")

    # Ensure models directory exists
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Load or Initialize ResNet-18 Checkpoint (.pth)
    print("\n[2] Loading ResNet18 from .pth checkpoint...")
    start_load = time.perf_counter()
    adapter = model_manager.load_model(
        model_type="resnet18",
        checkpoint_path=str(checkpoint_path),
        device="cpu"
    )
    load_time = (time.perf_counter() - start_load) * 1000.0
    print(f"    Loaded in {load_time:.2f} ms")
    assert checkpoint_path.exists(), f"Checkpoint file {checkpoint_path} should exist on disk."
    print(f"    Verified local .pth file exists: {checkpoint_path} ({checkpoint_path.stat().st_size / (1024*1024):.2f} MB)")

    # 2. Verify Metadata
    print("\n[3] Checking Model Metadata...")
    meta = adapter.metadata()
    print(f"    Model Name:     {meta['model_name']}")
    print(f"    Architecture:   {meta['architecture']}")
    print(f"    Total Params:   {meta['total_parameters']:,}")
    print(f"    Input Res:      {meta['input_resolution']}")
    print(f"    Device:         {meta['device']}")
    print(f"    Layers:         {', '.join(meta['layer_names'])}")
    assert meta["total_parameters"] > 11_000_000, "ResNet18 should have ~11.7M parameters"
    assert meta["is_loaded"] is True, "Model should report loaded = True"

    # 3. Preprocess Test Image
    print("\n[4] Generating and Preprocessing Test Image...")
    img_bytes = generate_test_image()
    tensor = preprocessor.preprocess(img_bytes)
    print(f"    Tensor shape:   {list(tensor.shape)} (Expected: [1, 3, 224, 224])")
    assert list(tensor.shape) == [1, 3, 224, 224]

    # 4. Run Prediction Flow
    print("\n[5] Running Prediction (.pth -> ResNet18 -> Image -> Prediction + Confidence)...")
    pred_res = adapter.predict(img_bytes, top_k=5)
    print(f"    Top Label:      {pred_res['top_label']}")
    print(f"    Top Index:      {pred_res['top_index']}")
    print(f"    Confidence:     {pred_res['confidence']:.4f} ({pred_res['predictions'][0]['confidence_percentage']}%)")
    print(f"    Latency:        {pred_res['inference_time_ms']} ms")
    print("    Top 5 Predictions:")
    for p in pred_res["predictions"]:
        print(f"      - [{p['index']}] {p['label']}: {p['confidence_percentage']}% (prob: {p['probability']:.4f})")

    assert "confidence" in pred_res
    assert "predictions" in pred_res
    assert len(pred_res["predictions"]) == 5
    assert pred_res["confidence"] > 0.0

    # 5. Extract Feature Embedding
    print("\n[6] Extracting Latent Feature Embedding...")
    emb_res = adapter.get_embedding(img_bytes)
    print(f"    Dimension:      {emb_res['dimension']} (Expected: 512 for ResNet18)")
    print(f"    Norm:           {emb_res['norm']}")
    print(f"    Latency:        {emb_res['inference_time_ms']} ms")
    print(f"    Embedding Head: {emb_res['embedding'][:6]}...")
    assert emb_res["dimension"] == 512
    assert len(emb_res["embedding"]) == 512
    assert emb_res["norm"] > 0.0

    # 6. Extract Intermediate Layer Activations
    print("\n[7] Extracting Layer Activations & Spatial Maps...")
    act_res = adapter.get_activations(img_bytes)
    print(f"    Captured Layers: {len(act_res['layers'])}")
    for layer in act_res["layers"]:
        print(f"      - {layer['layer_name']}: shape={layer['shape']}, mean={layer['mean']}, max={layer['max']}, sparsity={layer['sparsity']}")
        assert layer["heatmap_2d"] is not None
        assert len(layer["heatmap_2d"]) > 0
    assert len(act_res["layers"]) == 4

    print("\n==================================================")
    print("  ALL BACKEND PIPELINE CHECKS PASSED SUCCESSFULLY ")
    print("==================================================")

if __name__ == "__main__":
    run_tests()
