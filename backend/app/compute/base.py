import time
from abc import ABC, abstractmethod
from enum import Enum
from typing import Dict, Any, Optional, List
from pydantic import BaseModel, Field


class ExperimentStatus(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class ExperimentType(str, Enum):
    INVARIANT_MITIGATION = "invariant_mitigation"
    PERTURBATION_SWEEP = "perturbation_sweep"
    BATCH_INFERENCE = "batch_inference"
    GEMMA_REASONING = "gemma_reasoning"


class ExperimentTimestamps(BaseModel):
    created_at: float = Field(default_factory=time.time)
    started_at: Optional[float] = None
    completed_at: Optional[float] = None


class Experiment(BaseModel):
    id: str
    type: str = "invariant_mitigation"
    parameters: Dict[str, Any] = Field(default_factory=dict)
    status: ExperimentStatus = ExperimentStatus.QUEUED
    progress: int = 0
    results: Optional[Dict[str, Any]] = None
    timestamps: ExperimentTimestamps = Field(default_factory=ExperimentTimestamps)
    logs: List[str] = Field(default_factory=list)
    provider: str = "local"
    error: Optional[str] = None


class ComputeProvider(ABC):
    """
    Abstract interface for compute backends (Local execution or DigitalOcean cloud execution).
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the compute provider (e.g. 'local', 'digitalocean')."""
        pass

    @abstractmethod
    def submit_experiment(self, experiment: Experiment) -> Experiment:
        """Submits an experiment for execution. Non-blocking."""
        pass

    @abstractmethod
    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        """Returns the current state, progress, and results of an experiment."""
        pass

    @abstractmethod
    def list_experiments(self) -> List[Experiment]:
        """Lists all recorded experiments."""
        pass

    @abstractmethod
    def cancel_experiment(self, experiment_id: str) -> bool:
        """Cancels a running experiment."""
        pass
