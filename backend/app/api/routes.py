import io
import torch
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query, status
from fastapi.responses import JSONResponse, StreamingResponse
from typing import Optional, List

from ..schemas import (
    PredictionResponse,
    EmbeddingResponse,
    ActivationsResponse,
    ModelMetadataResponse,
    ModelLoadRequest,
    ModelLoadResponse,
    AnalysisStartRequest,
    AnalysisStatusResponse,
    AnalysisSummaryResponse,
    SamplesResponse,
    BlindSpotItem,
    BlindSpotDetailResponse,
    ActivationPathwayResponse,
    ActivationPathwayCompareRequest,
    WhatIfExperimentRequest,
    WhatIfExperimentResponse,
    WhatIfSampleCandidate,
    StructuredEvidence,
    GemmaHypothesizeRequest,
    GemmaHypothesizeResponse,
    ResearcherExperimentRunRequest,
    ResearcherExperimentRunResponse,
    ExperimentCreateRequest,
    ExperimentCancelResponse,
    GemmaInterpretationRequest,
    GemmaInterpretationResponse,
)
from ..services.model_loader import model_manager
from ..services.analysis_manager import analysis_manager
from ..services.gemma_researcher import evidence_engine, gemma_researcher, experiment_execution_engine
from ..compute.manager import compute_manager
from ..compute.base import Experiment
from ..config import settings

router = APIRouter(prefix="/api", tags=["Neural Trail Inference & Analysis"])


# ==========================================
# Core Model & Health Endpoints
# ==========================================

@router.get("/health", summary="Service Health & Active Model State")
async def health_check():
    meta = model_manager.active_adapter.metadata() if model_manager.is_loaded else None
    return {
        "status": "healthy",
        "service": "neural-trail-backend",
        "model_loaded": model_manager.is_loaded,
        "active_model": meta["model_name"] if meta else None,
        "active_model_type": model_manager.active_model_type,
        "device": settings.DEVICE,
        "analysis_ready": analysis_manager.get_status()["has_results"],
    }


@router.get("/model/info", response_model=ModelMetadataResponse, summary="Get Active Model Metadata")
async def get_model_info():
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No model is currently loaded. Select a model from the model selector to initiate investigation.",
        )
    return model_manager.active_adapter.metadata()


@router.get("/model/checkpoints", summary="List Available Local .pth Checkpoints")
async def list_checkpoints():
    checkpoints = model_manager.list_checkpoints()
    return {
        "available_checkpoints": checkpoints,
        "active_checkpoint": settings.get_checkpoint_path(),
    }


@router.post("/model/load", response_model=ModelLoadResponse, summary="Load or Switch Built-in / Checkpoint Model")
async def load_model(request: ModelLoadRequest):
    try:
        adapter = model_manager.load_model(
            model_type=request.model_type or settings.MODEL_TYPE,
            checkpoint_path=request.checkpoint_path,
            device=request.device or settings.DEVICE,
        )
        meta = adapter.metadata()
        return ModelLoadResponse(
            success=True,
            message=f"Successfully loaded {meta['model_name']}",
            metadata=meta,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}",
        )


@router.post("/model/load-custom-test", response_model=ModelLoadResponse, summary="Load Custom Test Model (NeuralTrailCNN from custom/)")
async def load_custom_test_model():
    """
    Loads custom/model.py + custom/neural_trail_cnn.pth with ShapeDataset (custom/dataset.py).
    Validates generic PyTorch adapter without hardcoding it as product architecture.
    """
    try:
        adapter = model_manager.load_custom_test_model()
        meta = adapter.metadata()
        return ModelLoadResponse(
            success=True,
            message=f"Successfully loaded {meta['model_name']}",
            metadata=meta,
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load custom test model: {str(e)}",
        )


@router.post("/model/unload", summary="Unload Active Model and Reset Investigation State")
async def unload_model():
    model_manager.unload_model()
    return {"success": True, "message": "Model unloaded. Neural Trail returned to standby state."}


@router.post("/model/export", summary="Export Active Model (.pth Checkpoint)")
@router.get("/model/export", summary="Export Active Model (.pth Checkpoint)")
async def export_model():
    """
    Exports the currently active model's PyTorch state dictionary as a downloadable .pth checkpoint.
    """
    if not model_manager.is_loaded or not model_manager.active_adapter:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No model is currently loaded to export.",
        )
    try:
        adapter = model_manager.active_adapter
        buffer = io.BytesIO()
        if hasattr(adapter, "model") and adapter.model is not None:
            torch.save(adapter.model.state_dict(), buffer)
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Active adapter does not contain an exportable PyTorch model.",
            )
        buffer.seek(0)
        meta = adapter.metadata()
        raw_name = meta.get("model_name", "model").lower()
        clean_name = "".join(c if c.isalnum() or c in ("_", "-") else "_" for c in raw_name)
        filename = f"{clean_name}_checkpoint.pth"

        return StreamingResponse(
            buffer,
            media_type="application/octet-stream",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Access-Control-Expose-Headers": "Content-Disposition",
            },
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Model export failed: {str(e)}",
        )


@router.post("/model/upload", response_model=ModelLoadResponse, summary="Upload & Validate PyTorch (.pt / .pth) Model")
async def upload_model(
    file: UploadFile = File(..., description="PyTorch model file (.pt or .pth)"),
    model_code: Optional[str] = Form(None, description="Optional Python source code for model definition"),
    class_name: Optional[str] = Form(None, description="Optional class name of nn.Module"),
    classes: Optional[str] = Form(None, description="Comma-separated class names"),
    shape: Optional[str] = Form(None, description="Input shape formatted as C,H,W (e.g. 3,32,32 or 3,224,224)"),
):
    if not file.filename.endswith((".pt", ".pth")):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file format. Please upload a PyTorch model file with extension .pt or .pth",
        )

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        shape_list = None
        if shape:
            shape_list = [int(x.strip()) for x in shape.split(",") if x.strip()]

        classes_list = None
        if classes:
            classes_list = [c.strip() for c in classes.split(",") if c.strip()]

        adapter = model_manager.load_uploaded_model(
            file_bytes=contents,
            filename=file.filename,
            model_code=model_code,
            class_name=class_name,
            input_shape=shape_list,
            classes=classes_list,
        )
        meta = adapter.metadata()

        return ModelLoadResponse(
            success=True,
            message=f"Successfully loaded uploaded PyTorch model '{meta['model_name']}'",
            metadata=meta,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Model validation error: {str(e)}",
        )



@router.post("/model/predict", response_model=PredictionResponse, summary="Run Inference on Single Image")
async def predict_image(
    file: UploadFile = File(..., description="Image file (JPEG, PNG, WEBP, etc.)"),
    top_k: int = Query(default=5, ge=1, le=50, description="Number of top predictions to return"),
):
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model is not loaded. Ensure a .pth checkpoint is loaded before inferencing.",
        )

    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Uploaded file is empty.")

        result = model_manager.active_adapter.predict(image_bytes, top_k=top_k)
        return result
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference error: {str(e)}",
        )


@router.post("/model/embedding", response_model=EmbeddingResponse, summary="Extract Latent Feature Embedding")
async def get_image_embedding(
    file: UploadFile = File(..., description="Image file to extract embedding from")
):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        image_bytes = await file.read()
        result = model_manager.active_adapter.get_embedding(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction error: {str(e)}")


@router.post("/model/activations", response_model=ActivationsResponse, summary="Extract Layer Activations & Spatial Maps")
async def get_image_activations(
    file: UploadFile = File(..., description="Image file to extract layer activations from")
):
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        image_bytes = await file.read()
        result = model_manager.active_adapter.get_activations(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Activation extraction error: {str(e)}")


# ==========================================
# Neural Trail Analysis Engine Endpoints
# ==========================================

@router.post("/analysis/start", summary="Start Dataset Inference & Blind-Spot Analysis")
async def start_analysis(request: AnalysisStartRequest):
    """
    Executes the real Neural Trail analysis pipeline:
    dataset → ResNet18 → predictions + embeddings + activations → analysis.
    Computes PCA 2D projections, failure clustering, and diagnostic metrics.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        res = analysis_manager.start_analysis(
            dataset_source=request.dataset_source or "benchmark",
            sample_count=request.sample_count or 100,
            reducer_type=request.reducer_type or "pca",
            k_clusters=request.k_clusters,
            custom_dir=request.custom_dir,
            async_run=request.async_run or False,
        )
        return res
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Analysis failed: {str(e)}",
        )


@router.get("/analysis/status", response_model=AnalysisStatusResponse, summary="Get Analysis Execution Status")
async def get_analysis_status():
    """Returns current analysis stage, progress percentage, and state."""
    return analysis_manager.get_status()


@router.get("/analysis/summary", response_model=AnalysisSummaryResponse, summary="Get Aggregate Model Metrics & Summary")
async def get_analysis_summary():
    """Returns aggregate accuracy, health score, blind spot count, and comparative metrics."""
    return analysis_manager.get_summary()


@router.get("/analysis/samples", response_model=SamplesResponse, summary="Get 2D Representation Data for Visualization")
async def get_samples(
    cluster_id: Optional[str] = Query(default=None, description="Filter by cluster ID (e.g. cluster-0)"),
    status: Optional[str] = Query(default=None, description="Filter by stability: all, stable, uncertain, failure"),
    min_confidence: Optional[float] = Query(default=None, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    class_id: Optional[int] = Query(default=None, description="Filter by true or predicted class index"),
    limit: int = Query(default=500, ge=1, le=1000, description="Max samples to return"),
    offset: int = Query(default=0, ge=0, description="Offset for pagination"),
):
    """
    Returns sample tracks with true/predicted labels, confidences, correctness,
    stability, cluster assignments, and 2D embedding coordinates.
    """
    return analysis_manager.get_samples(
        cluster_id=cluster_id,
        status=status,
        min_confidence=min_confidence,
        class_id=class_id,
        limit=limit,
        offset=offset,
    )


@router.get("/analysis/blind-spots", response_model=List[BlindSpotItem], summary="Get Identified Blind Spots / Failure Clusters")
async def get_blind_spots():
    """
    Returns detected representation clusters ranked by failure rate and severity.
    Includes cluster center, radius, accuracy, failure rate, and common prediction errors.
    """
    return analysis_manager.get_blind_spots()


@router.get("/analysis/blind-spots/{cluster_id}", response_model=BlindSpotDetailResponse, summary="Get Individual Blind-Spot Details")
async def get_blind_spot_detail(cluster_id: str):
    """
    Returns deep diagnostic details for a specific blind spot, including its
    representative samples, full cluster sample list, and error confusion breakdown.
    """
    spot = analysis_manager.get_blind_spot_by_id(cluster_id)
    if not spot:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Blind spot with ID '{cluster_id}' not found.",
        )
    return spot


@router.get("/analysis/replay", summary="Get Actual Investigation Events for Replay Timeline")
async def get_investigation_replay():
    """
    Returns actual investigation events recorded chronologically during the session.
    No mock or static events.
    """
    events = analysis_manager.get_replay_events()
    return {"total": len(events), "events": events}


@router.post("/analysis/select-sample", summary="Select Active Sample in Shared Investigation State")
async def select_investigation_sample(sample_id: str = Query(..., description="ID of sample to inspect")):
    """
    Sets active sample across FIND, TRACE, and STRESS tabs, recording the inspection in the timeline.
    """
    analysis_manager.set_active_sample(sample_id)
    return {"success": True, "active_sample_id": sample_id}


# ==========================================
# Neural X-Ray: Activation Pathway Endpoints
# ==========================================

@router.get("/analysis/xray/pathway", response_model=ActivationPathwayResponse, summary="Get Activation Pathway Visualization (5 Stages)")
async def get_activation_pathway(
    sample_a_id: Optional[str] = Query(default=None, description="Canonical / Successful sample ID"),
    sample_b_id: Optional[str] = Query(default=None, description="Degraded / Failed sample ID"),
    selected_layer_id: str = Query(default="layer-middle", description="Selected layer identifier"),
):
    """
    Activation Pathway Visualization:
    Extracts real ResNet-18 activations across 5 stages: INPUT → EARLY → MIDDLE → DEEP → OUTPUT.
    Enables comparative analysis between canonical and failure pathways.
    Note: Visualizing representation divergence does not prove causal mechanisms.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        return analysis_manager.get_activation_pathway(
            sample_a_id=sample_a_id,
            sample_b_id=sample_b_id,
            selected_layer_id=selected_layer_id,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Activation pathway extraction failed: {str(e)}",
        )


@router.post("/analysis/xray/compare", response_model=ActivationPathwayResponse, summary="Compare Activation Pathways of Two Samples")
async def compare_activation_pathways(request: ActivationPathwayCompareRequest):
    """
    Compares intermediate activation pathways of two specified samples across 5 layers.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        return analysis_manager.get_activation_pathway(
            sample_a_id=request.sample_a_id,
            sample_b_id=request.sample_b_id,
            selected_layer_id=request.selected_layer_id or "layer-middle",
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Activation comparison failed: {str(e)}",
        )


# ==========================================
# What-If / Perturbation Lab Endpoints
# ==========================================

@router.get("/analysis/whatif/samples", response_model=List[WhatIfSampleCandidate], summary="Get Candidate Samples for What-If Testing")
async def get_whatif_samples():
    """
    Returns candidate evaluation samples across distinct classes
    ready for What-If visual stress testing.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")
    return analysis_manager.get_whatif_samples()


@router.post("/analysis/whatif/experiment", response_model=WhatIfExperimentResponse, summary="Run Real-Time What-If Stress Perturbation")
async def run_whatif_experiment(request: WhatIfExperimentRequest):
    """
    Executes real-time What-If experiment:
    original image → original prediction → perturbation → new prediction.
    Supports modular transformations: rotation, noise, blur, crop, brightness, occlusion.
    Returns original/perturbed predictions, confidence change, prediction flip, and robustness score.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")

    try:
        return analysis_manager.run_whatif_experiment(
            sample_id=request.sample_id,
            rotation=request.rotation or 0.0,
            noise=request.noise or 0.0,
            blur=request.blur or 0.0,
            crop=request.crop or 0.0,
            brightness=request.brightness or 0.0,
            occlusion=request.occlusion or 0.0,
            include_images=request.include_images if request.include_images is not None else True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"What-If experiment failed: {str(e)}",
        )


# ==========================================
# Gemma 4 AI Researcher Endpoints
# DETECT → HYPOTHESIZE → EXPERIMENT → OBSERVE
# ==========================================

@router.get("/researcher/evidence", response_model=StructuredEvidence, summary="Generate Structured Evidence from Analysis Engine (DETECT)")
async def get_researcher_evidence(
    cluster_id: Optional[str] = Query(default=None, description="Blind spot cluster ID to extract evidence for"),
):
    """
    DETECT Stage:
    The Analysis Engine generates structured numerical evidence from real ResNet18 model inference:
    {
      "model": "ResNet18",
      "overall_accuracy": 0.81,
      "blind_spot_accuracy": 0.42,
      "common_error": "cat_to_dog",
      "rotation_flip_rate": 0.37
    }
    All measurements originate strictly from the Analysis Engine. Zero hallucination.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")
    try:
        return evidence_engine.generate_evidence(cluster_id=cluster_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate structured evidence: {str(e)}",
        )


@router.post("/researcher/hypothesize", response_model=GemmaHypothesizeResponse, summary="Gemma 4 Formulates Research Hypothesis (HYPOTHESIZE)")
async def gemma_hypothesize(request: GemmaHypothesizeRequest):
    """
    HYPOTHESIZE Stage:
    Gemma 4 acts as the AI Researcher (not model under test, not numerical calculator).
    Consumes structured evidence and returns:
    {
      "observation": "...",
      "hypothesis": "...",
      "recommended_experiment": "...",
      "expected_signal": "...",
      "priority": "high"
    }
    Gemma strictly never invents measurements. Prompts are optimized to minimize token usage.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")
    try:
        evidence = request.evidence
        if not evidence:
            evidence = evidence_engine.generate_evidence(cluster_id=request.cluster_id)
        
        return gemma_researcher.hypothesize(evidence)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemma 4 hypothesizing failed: {str(e)}",
        )


@router.post("/researcher/experiment/run", response_model=ResearcherExperimentRunResponse, summary="Execute Recommended Experiment (EXPERIMENT → OBSERVE)")
@router.post("/researcher/experiment", response_model=ResearcherExperimentRunResponse, include_in_schema=False)
async def run_researcher_experiment(request: ResearcherExperimentRunRequest):
    """
    EXPERIMENT → OBSERVE Stage:
    User approves and executes Gemma 4's recommended experiment protocol on the model.
    Collects real post-intervention telemetry and returns empirical confirmation/refutation
    of Gemma's hypothesis.
    """
    if not model_manager.is_loaded:
        raise HTTPException(status_code=400, detail="Model is not loaded.")
    try:
        return experiment_execution_engine.run_experiment(
            cluster_id=request.cluster_id,
            experiment_protocol=request.experiment_protocol,
            perturbation_type=request.perturbation_type or "rotation",
            intensity=request.intensity or 20.0,
            apply_mitigation=request.apply_mitigation if request.apply_mitigation is not None else True,
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Experiment execution failed: {str(e)}",
        )


# ==========================================
# Compute & Experiment Execution Layer Endpoints
# (Unified Local & DigitalOcean Compute Provider)
# ==========================================

@router.get("/compute/info", summary="Get Active Compute Provider Status")
async def get_compute_info():
    """
    Returns information about active compute provider (Local or DigitalOcean),
    cloud region, and available execution modes.
    """
    prov = compute_manager.get_provider()
    return {
        "active_provider": prov.name,
        "default_provider": compute_manager.default_provider_name,
        "supported_providers": ["local", "digitalocean"],
        "is_cloud_configured": prov.name == "digitalocean" or getattr(compute_manager.get_provider("digitalocean"), "is_configured", False),
        "region": getattr(prov, "region", "local-system"),
        "capabilities": [
            "batch_inference",
            "perturbation_sweeps",
            "large_experiments",
            "gemma_inference",
            "invariant_mitigation",
        ],
    }


@router.post("/experiments", response_model=Experiment, summary="Submit New Experiment Job (Non-blocking)")
async def submit_experiment(request: ExperimentCreateRequest):
    """
    Submits an experiment job to the execution layer.
    Executes asynchronously in a background thread pool without blocking FastAPI.
    Supports Local and DigitalOcean compute providers via unified API.
    """
    try:
        exp = compute_manager.create_and_submit_experiment(
            exp_type=request.type or "invariant_mitigation",
            parameters=request.parameters or {},
            provider_name=request.provider,
        )
        return exp
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to submit experiment: {str(e)}",
        )


@router.get("/experiments", response_model=List[Experiment], summary="List All Experiment Jobs")
async def list_experiments():
    """
    Lists all queued, running, completed, and failed experiments across providers.
    """
    return compute_manager.list_experiments()


@router.get("/experiments/{experiment_id}", response_model=Experiment, summary="Get Experiment Details, Progress, & Results")
async def get_experiment(experiment_id: str):
    """
    Retrieves real-time status, progress (0-100%), results, and live logs for an experiment.
    """
    exp = compute_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    return exp


@router.post("/experiments/{experiment_id}/cancel", response_model=ExperimentCancelResponse, summary="Cancel Running Experiment")
async def cancel_experiment(experiment_id: str):
    """
    Cancels an active running or queued experiment.
    """
    success = compute_manager.cancel_experiment(experiment_id)
    if not success:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment '{experiment_id}' could not be cancelled or does not exist.",
        )
    return ExperimentCancelResponse(
        success=True,
        message=f"Experiment '{experiment_id}' has been cancelled.",
        experiment_id=experiment_id,
    )


@router.post("/experiments/{experiment_id}/interpret", response_model=GemmaInterpretationResponse, summary="Gemma 4 Evaluates Completed Experiment Results")
async def interpret_experiment(experiment_id: str):
    """
    Gemma 4 interprets completed experiment results, evaluates empirical delta against original hypothesis,
    confirms/refutes hypothesis, and prepares Before/After remediation patch.
    """
    exp = compute_manager.get_experiment(experiment_id)
    if not exp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Experiment '{experiment_id}' not found.",
        )
    if exp.status != "COMPLETED" or not exp.results:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Experiment '{experiment_id}' is not yet completed (current status: {exp.status}).",
        )

    try:
        res = gemma_researcher.interpret_experiment_results(experiment_id, exp.results)
        return GemmaInterpretationResponse(**res)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Gemma interpretation failed: {str(e)}",
        )




