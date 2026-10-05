import os
import uuid
import logging
from typing import Dict, Optional, List, Any

from .base import ComputeProvider, Experiment, ExperimentStatus, ExperimentType
from .local_provider import LocalComputeProvider
from .digitalocean_provider import DigitalOceanComputeProvider

logger = logging.getLogger("neural-trail-backend")


class ComputeManager:
    """
    Unified manager for compute providers.
    Supports switching between LocalComputeProvider and DigitalOceanComputeProvider
    via environment variable COMPUTE_PROVIDER=local|digitalocean or per-request override.
    The frontend interacts with the exact same unified API regardless of the provider!
    """

    def __init__(self):
        self._local_provider = LocalComputeProvider()
        self._do_provider = DigitalOceanComputeProvider()
        logger.info("Initialized ComputeManager with Local and DigitalOcean compute providers.")

    @property
    def default_provider_name(self) -> str:
        return os.getenv("COMPUTE_PROVIDER", "local").lower()

    def get_provider(self, provider_name: Optional[str] = None) -> ComputeProvider:
        name = (provider_name or self.default_provider_name).lower()
        if name in ("digitalocean", "do", "cloud"):
            return self._do_provider
        return self._local_provider

    def create_and_submit_experiment(
        self,
        exp_type: str = "invariant_mitigation",
        parameters: Optional[Dict[str, Any]] = None,
        provider_name: Optional[str] = None,
    ) -> Experiment:
        exp_id = f"EXP-{uuid.uuid4().hex[:6].upper()}"
        provider = self.get_provider(provider_name)

        experiment = Experiment(
            id=exp_id,
            type=exp_type,
            parameters=parameters or {},
            provider=provider.name,
        )

        return provider.submit_experiment(experiment)

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        # Check both providers
        exp = self._local_provider.get_experiment(experiment_id)
        if exp:
            return exp
        return self._do_provider.get_experiment(experiment_id)

    def list_experiments(self) -> List[Experiment]:
        local_exps = self._local_provider.list_experiments()
        do_exps = self._do_provider.list_experiments()
        all_exps = local_exps + do_exps
        # Sort by creation timestamp descending
        all_exps.sort(key=lambda e: e.timestamps.created_at, reverse=True)
        return all_exps

    def cancel_experiment(self, experiment_id: str) -> bool:
        if self._local_provider.cancel_experiment(experiment_id):
            return True
        return self._do_provider.cancel_experiment(experiment_id)


# Global singleton compute manager
compute_manager = ComputeManager()
