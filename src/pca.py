"""
Dimensionality reduction with PCA.

The number of components is the smallest one (between 2 and the number of
features) whose cumulative explained variance reaches ``cvr_threshold``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from src.utils import get_logger, load_config, load_frame, resolve, save_parquet

logger = get_logger(__name__)

_CONFIG = load_config("pca")
DEFAULT_IN_PATH = resolve(_CONFIG["ingest_path"])
DEFAULT_OUT_PATH = resolve(_CONFIG["output_path"])
NOT_COLUMNS = _CONFIG["columns_not_considered"]
CVR_THRESHOLD = _CONFIG["cvr_threshold"]


def choose_n_components(data: pd.DataFrame, thresh: float, min_components: int = 2) -> int:
    """Smallest number of components reaching ``thresh`` cumulative explained variance."""
    max_components = min(data.shape)
    cumulative = np.cumsum(PCA(n_components=max_components).fit(data).explained_variance_ratio_)
    n = int(np.searchsorted(cumulative, thresh) + 1)
    n = max(min_components, min(n, max_components))
    logger.info("PCA: %d components explain %.1f%% of variance", n, 100 * cumulative[n - 1])
    return n


def pc_analyzer(
    in_path: str = DEFAULT_IN_PATH,
    out_path: str = DEFAULT_OUT_PATH,
    drop_cols=NOT_COLUMNS,
    cvr_thresh: float = CVR_THRESHOLD,
) -> str:
    """
    Run PCA on the scaled customer features and save the principal components.

    :raises KeyError: If a column in ``drop_cols`` does not exist.
    :raises ValueError: If fewer than 4 features remain or the threshold is not in ]0, 1[.
    """
    data = load_frame(in_path)
    missing = [c for c in drop_cols if c not in data.columns]
    if missing:
        raise KeyError(f"Column not found in Dataframe: {missing}")
    if not 0 < cvr_thresh < 1:
        raise ValueError("cvr_thresh should lie between 0 and 1.")

    features = data.set_index("CustomerID").drop(columns=list(drop_cols))
    if features.shape[1] < 4:
        raise ValueError("Columns less than 4. Need not proceed with PCA.")
    features = features.astype(float)

    pca = PCA(n_components=choose_n_components(features, cvr_thresh)).fit(features)
    reduced = pd.DataFrame(
        pca.transform(features),
        columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        index=features.index,
    )
    return save_parquet(reduced, out_path)
