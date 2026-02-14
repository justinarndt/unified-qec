"""
Hugging Face Hub Integration for Model Zoo

Handles upload/download of model artifacts to/from the Hugging Face Hub,
with revision pinning, smart caching, and auto-generated Model Cards.

Section 7 §3.3: Distribution and versioning via Hugging Face Hub.
"""

import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from huggingface_hub import (
        HfApi,
        hf_hub_download,
        snapshot_download,
        upload_folder,
    )
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


def _check_hf():
    if not HF_AVAILABLE:
        raise ImportError(
            "huggingface-hub is required for Model Zoo distribution. "
            "Install with: pip install unified-qec[zoo]"
        )


class ZooManager:
    """Upload/download model artifacts from Hugging Face Hub.

    Provides git-versioned model distribution with SHA-pinned
    revision support and auto-generated Model Cards.

    Parameters
    ----------
    repo_id : str
        Hugging Face repository ID (e.g., ``'justinarndt/unified-qec-zoo'``).
    token : str, optional
        Hugging Face API token. If None, uses cached credentials.
    cache_dir : str, optional
        Local cache directory. Default: ``~/.cache/huggingface``.

    Examples
    --------
    >>> zoo = ZooManager("justinarndt/unified-qec-zoo")
    >>> zoo.push("./checkpoints/step_00001000", commit_message="d=5 BP+OSD model")
    >>> local_path = zoo.pull("step_00001000/model.npz", revision="abc123")
    """

    def __init__(
        self,
        repo_id: str = "justinarndt/unified-qec-zoo",
        token: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        _check_hf()
        self.repo_id = repo_id
        self.token = token
        self.cache_dir = cache_dir
        self._api = HfApi(token=token)

    def create_repo(self, private: bool = False, exist_ok: bool = True):
        """Create the HuggingFace repository if it doesn't exist.

        Parameters
        ----------
        private : bool
            Whether the repo should be private.
        exist_ok : bool
            If True, don't raise if repo already exists.
        """
        self._api.create_repo(
            repo_id=self.repo_id,
            repo_type="model",
            private=private,
            exist_ok=exist_ok,
            token=self.token,
        )

    def push(
        self,
        local_path: str,
        commit_message: str = "Update model checkpoint",
        revision: Optional[str] = None,
        generate_card: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Upload a checkpoint directory to the Hub.

        Parameters
        ----------
        local_path : str or Path
            Local directory containing checkpoint files.
        commit_message : str
            Git commit message.
        revision : str, optional
            Branch or tag name. Default: ``'main'``.
        generate_card : bool
            If True, auto-generate a Model Card from metadata.
        metadata : dict, optional
            Model metadata for the Model Card.

        Returns
        -------
        str
            Commit URL on the Hub.
        """
        local_path = Path(local_path)

        # Auto-generate Model Card
        if generate_card and metadata:
            card_content = self.generate_model_card(metadata)
            card_path = local_path / "README.md"
            with open(card_path, "w") as f:
                f.write(card_content)

        # Upload
        commit_info = upload_folder(
            folder_path=str(local_path),
            repo_id=self.repo_id,
            repo_type="model",
            commit_message=commit_message,
            revision=revision,
            token=self.token,
        )

        return str(commit_info)

    def pull(
        self,
        filename: str,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ) -> Path:
        """Download a single file from the Hub.

        Uses smart caching: if the file is already cached and matches
        the requested revision, no network request is made.

        Parameters
        ----------
        filename : str
            Relative path within the repo (e.g., ``'step_00001000/model.npz'``).
        revision : str, optional
            Git revision (SHA, branch, or tag). Default: ``'main'``.
        cache_dir : str, optional
            Override local cache directory.

        Returns
        -------
        Path
            Local path to the downloaded file.
        """
        local = hf_hub_download(
            repo_id=self.repo_id,
            filename=filename,
            revision=revision,
            cache_dir=cache_dir or self.cache_dir,
            token=self.token,
        )
        return Path(local)

    def pull_snapshot(
        self,
        revision: Optional[str] = None,
        cache_dir: Optional[str] = None,
        allow_patterns: Optional[list] = None,
    ) -> Path:
        """Download the entire repository snapshot.

        Parameters
        ----------
        revision : str, optional
            Git revision. Default: ``'main'``.
        cache_dir : str, optional
            Override local cache directory.
        allow_patterns : list, optional
            Glob patterns to include (e.g., ``['*.npz', '*.json']``).

        Returns
        -------
        Path
            Local path to the snapshot directory.
        """
        local = snapshot_download(
            repo_id=self.repo_id,
            revision=revision,
            cache_dir=cache_dir or self.cache_dir,
            allow_patterns=allow_patterns,
            token=self.token,
        )
        return Path(local)

    def list_revisions(self) -> list:
        """List all commits (revisions) for the repo.

        Returns
        -------
        list of dict
            Each dict contains commit SHA, message, and date.
        """
        commits = self._api.list_repo_commits(
            repo_id=self.repo_id,
            repo_type="model",
            token=self.token,
        )
        return [
            {
                "sha": c.commit_id,
                "message": c.title,
                "date": str(c.created_at),
            }
            for c in commits
        ]

    def generate_model_card(self, metadata: Dict[str, Any]) -> str:
        """Generate a Hugging Face Model Card from metadata.

        Parameters
        ----------
        metadata : dict
            Must contain keys like ``code_distance``, ``noise_model``,
            ``hidden_dim``, ``num_layers``, ``threshold``, etc.

        Returns
        -------
        str
            Markdown content for the Model Card with YAML header.
        """
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d")

        # YAML frontmatter
        tags = metadata.get("tags", [
            "quantum-error-correction",
            "jax",
            "flax",
            "sinter-compatible",
            "unified-qec",
        ])
        tags_yaml = "\n".join(f"  - {t}" for t in tags)

        card = textwrap.dedent(f"""\
        ---
        tags:
        {tags_yaml}
        library_name: flax
        license: mit
        datasets:
          - custom
        metrics:
          - accuracy
        ---

        # {metadata.get('model_name', 'Unified-QEC Neural Decoder')}

        ## Model Description

        A Transformer-based neural syndrome decoder for quantum error correction,
        trained on Stim-generated surface code data with the unified-qec toolkit.

        - **Architecture:** {metadata.get('num_layers', 4)}-layer Transformer, \
        {metadata.get('hidden_dim', 256)}-dim, {metadata.get('num_heads', 8)} heads
        - **Code distance:** {metadata.get('code_distance', 'N/A')}
        - **Noise model:** {metadata.get('noise_model', 'depolarizing')}
        - **Physical error rate:** {metadata.get('physical_error_rate', 'N/A')}
        - **Training shots:** {metadata.get('training_shots', 'N/A')}
        - **Date:** {now}

        ## Performance

        | Metric | Value |
        |---|---|
        | Logical error rate | {metadata.get('logical_error_rate', 'N/A')} |
        | Threshold | {metadata.get('threshold', 'N/A')} |
        | Training loss (final) | {metadata.get('final_loss', 'N/A')} |
        | Decode latency (μs/shot) | {metadata.get('decode_latency_us', 'N/A')} |

        ## Usage

        ```python
        from unified_qec.zoo import ModelCheckpoint, NeuralSyndromeDecoder
        from unified_qec.zoo.hub import ZooManager

        zoo = ZooManager("{self.repo_id}")
        local = zoo.pull_snapshot()
        ckpt = ModelCheckpoint(local)
        state, meta = ckpt.restore()
        ```

        ## Training

        ```python
        from unified_qec.zoo.neural_decoder import create_train_state, train_step
        state = create_train_state(rng, num_detectors={metadata.get('num_detectors', 'N')}, \\
        num_observables={metadata.get('num_observables', 'M')})
        ```

        ## Citation

        Trained with [unified-qec](https://github.com/justinarndt/unified-qec).
        """)

        return card
