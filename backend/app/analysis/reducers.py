from abc import ABC, abstractmethod
from typing import Dict, Type, Optional
import numpy as np
from sklearn.decomposition import PCA


class BaseReducer(ABC):
    """
    Abstract Base Class for latent feature dimensionality reduction.
    Maps high-dimensional embeddings (e.g. 512-dim ResNet18) to 2D coordinates.
    """

    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Takes (N, D) float array of embeddings and returns (N, 2) array of coordinates.
        """
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class PCAReducer(BaseReducer):
    """
    Principal Component Analysis (PCA) 2D projection.
    Fast, deterministic, and preserves maximal global linear variance.
    Coordinates are normalized to a clean visual canvas range [-100, 100].
    """

    def __init__(self, target_scale: float = 85.0):
        self.target_scale = target_scale
        self.pca = PCA(n_components=2, random_state=42)
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    @property
    def name(self) -> str:
        return "pca"

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        if len(embeddings) < 2:
            return np.zeros((len(embeddings), 2), dtype=np.float32)

        coords_2d = self.pca.fit_transform(embeddings)
        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_

        # Normalize and scale to visual canvas range [-target_scale, target_scale]
        mins = coords_2d.min(axis=0)
        maxs = coords_2d.max(axis=0)
        ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)

        # Scale to [-1, 1] then multiply by target_scale
        normalized = 2.0 * (coords_2d - mins) / ranges - 1.0
        scaled = normalized * self.target_scale

        return np.round(scaled, 2)


class UMAPReducer(BaseReducer):
    """
    Uniform Manifold Approximation and Projection (UMAP) 2D projection.
    Preserves both local and global manifold structure.
    Falls back gracefully to PCA if umap-learn is not installed in the environment.
    """

    def __init__(self, n_neighbors: int = 15, min_dist: float = 0.1, target_scale: float = 85.0):
        self.n_neighbors = n_neighbors
        self.min_dist = min_dist
        self.target_scale = target_scale

    @property
    def name(self) -> str:
        return "umap"

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        try:
            import umap
            reducer = umap.UMAP(
                n_components=2,
                n_neighbors=min(self.n_neighbors, max(2, len(embeddings) - 1)),
                min_dist=self.min_dist,
                random_state=42,
            )
            coords_2d = reducer.fit_transform(embeddings)

            mins = coords_2d.min(axis=0)
            maxs = coords_2d.max(axis=0)
            ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
            normalized = 2.0 * (coords_2d - mins) / ranges - 1.0
            scaled = normalized * self.target_scale
            return np.round(scaled, 2)
        except ImportError:
            # Fall back to PCA if umap-learn package is not installed
            fallback = PCAReducer(target_scale=self.target_scale)
            return fallback.fit_transform(embeddings)


class TSNEReducer(BaseReducer):
    """
    t-Distributed Stochastic Neighbor Embedding (t-SNE) 2D projection.
    """

    def __init__(self, perplexity: float = 30.0, target_scale: float = 85.0):
        self.perplexity = perplexity
        self.target_scale = target_scale

    @property
    def name(self) -> str:
        return "tsne"

    def fit_transform(self, embeddings: np.ndarray) -> np.ndarray:
        from sklearn.manifold import TSNE
        perp = min(self.perplexity, max(2.0, float(len(embeddings) - 1) / 3.0))
        tsne = TSNE(n_components=2, perplexity=perp, random_state=42, init="pca", learning_rate="auto")
        coords_2d = tsne.fit_transform(embeddings)

        mins = coords_2d.min(axis=0)
        maxs = coords_2d.max(axis=0)
        ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        normalized = 2.0 * (coords_2d - mins) / ranges - 1.0
        scaled = normalized * self.target_scale
        return np.round(scaled, 2)


_REDUCER_REGISTRY: Dict[str, Type[BaseReducer]] = {
    "pca": PCAReducer,
    "umap": UMAPReducer,
    "tsne": TSNEReducer,
}


def get_reducer(name: str = "pca", **kwargs) -> BaseReducer:
    """Factory method to instantiate a dimensionality reducer."""
    reducer_cls = _REDUCER_REGISTRY.get(name.lower())
    if not reducer_cls:
        reducer_cls = PCAReducer
    return reducer_cls(**kwargs)
