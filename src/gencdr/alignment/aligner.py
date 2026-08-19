"""CSV-driven weighted DPO alignment runner.

Loads a policy model and a frozen reference from local checkpoint directories,
trains on one or more reward CSVs with the scaffold-grouped weighted DPO objective,
and saves the best model (by validation Spearman correlation) plus the tokenizer to
an output directory.

This is intentionally standalone: no experiment-tracker, config framework, or
reward-file auto-discovery — just ``reward CSV(s) -> aligned model``. Requires the
``align`` extra.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast

from gencdr.alignment.dataset import FrameworkCDRRewardCollator, FrameworkCDRRewardDataset
from gencdr.alignment.trainer import WeightedDPOModule
from gencdr.checkpoints import resolve_checkpoint

logger = logging.getLogger(__name__)


def _reward_bins(rewards: pd.Series) -> np.ndarray:
    """Bin rewards by percentile: 0 (top 10%), 1 (median to 90%), 2 (below median)."""
    values = pd.to_numeric(rewards, errors="coerce").to_numpy()
    if np.isnan(values).any():
        raise ValueError("Rewards contain non-numeric or NaN values.")
    p50, p90 = np.percentile(values, 50), np.percentile(values, 90)
    labels = np.full(values.shape, 2, dtype=np.int64)
    labels[values >= p50] = 1
    labels[values >= p90] = 0
    return labels


def _train_val_split(
    df: pd.DataFrame, val_size: float, seed: int, stratified: bool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Split a reward table into train/validation frames (optionally reward-stratified)."""
    rng = np.random.default_rng(seed)
    n = len(df)
    if n < 2:
        raise ValueError(f"Need at least 2 rows to split; got {n}.")

    if stratified:
        bins = _reward_bins(df["reward"])
        val_idx: List[int] = []
        for b in np.unique(bins):
            idx = np.where(bins == b)[0]
            rng.shuffle(idx)
            k = int(round(len(idx) * val_size))
            val_idx.extend(idx[:k].tolist())
        val_mask = np.zeros(n, dtype=bool)
        val_mask[val_idx] = True
    else:
        perm = rng.permutation(n)
        k = int(round(n * val_size))
        val_mask = np.zeros(n, dtype=bool)
        val_mask[perm[:k]] = True

    val_df = df.iloc[val_mask].reset_index(drop=True)
    tr_df = df.iloc[~val_mask].reset_index(drop=True)
    if len(tr_df) == 0 or len(val_df) == 0:
        # Degenerate split (tiny input): fall back to a single held-out row.
        tr_df, val_df = df.iloc[1:].reset_index(drop=True), df.iloc[:1].reset_index(drop=True)
    return tr_df, val_df


class WeightedDPOAligner:
    """Run scaffold-grouped weighted DPO alignment from reward CSV(s)."""

    def __init__(
        self,
        model: str,
        output_dir: str,
        ref_model: Optional[str] = None,
        mode: str = "single",
        order: str = "L-first",
        group_by: str = "source",
        include_scheme_token: bool = False,
        exp_name: str = "gencdr-wdpo",
        beta: float = 0.15,
        learning_rate: float = 2e-6,
        weight_decay: float = 0.01,
        batch_size: int = 16,
        max_epochs: int = 5,
        val_size: float = 0.2,
        stratified_val: bool = True,
        max_length: int = 256,
        gradient_clip_val: float = 1.0,
        precision: str = "32-true",
        num_workers: int = 4,
        seed: int = 42,
        num_devices: Optional[int] = None,
    ) -> None:
        """Store model references and training hyperparameters.

        Parameters
        ----------
        model : str
            Policy checkpoint: a directory or a short name (see gencdr.checkpoints).
        output_dir : str
            Directory for the aligned model, tokenizer, checkpoints and logs.
        ref_model : str, optional
            Frozen reference checkpoint. Defaults to ``model`` (the standard choice:
            the pre-alignment policy is the reference).
        mode : str
            ``"single"`` or ``"paired"``.
        beta : float
            KL-regularisation strength (manuscript default 0.15).
        """
        self.model = model
        self.ref_model = ref_model if ref_model is not None else model
        self.output_dir = output_dir
        self.mode = str(mode).lower()
        self.order = order
        self.group_by = group_by
        self.include_scheme_token = bool(include_scheme_token)
        self.exp_name = exp_name
        self.beta = float(beta)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.val_size = float(val_size)
        self.stratified_val = bool(stratified_val)
        self.max_length = int(max_length)
        self.gradient_clip_val = float(gradient_clip_val)
        self.precision = precision
        self.num_workers = int(num_workers)
        self.seed = int(seed)
        self.num_devices = num_devices

    def run(self, reward_csvs: List[str]) -> str:
        """Train on the given reward CSV(s) and return the best checkpoint path.

        Parameters
        ----------
        reward_csvs : list of str
            One or more reward CSV paths (concatenated before splitting).

        Returns
        -------
        str
            Path to the best checkpoint (highest validation Spearman correlation).
        """
        out_dir = Path(self.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        pl.seed_everything(self.seed, workers=True)

        logger.info("Loading %d reward CSV(s): %s", len(reward_csvs), reward_csvs)
        df = pd.concat([pd.read_csv(f).fillna("") for f in reward_csvs], ignore_index=True)
        tr_df, va_df = _train_val_split(df, self.val_size, self.seed, self.stratified_val)
        logger.info("Train rows: %d | Val rows: %d", len(tr_df), len(va_df))

        model_dir = str(resolve_checkpoint(self.model))
        ref_dir = str(resolve_checkpoint(self.ref_model))
        logger.info("Policy: %s | Reference: %s", model_dir, ref_dir)

        tokenizer = PreTrainedTokenizerFast.from_pretrained(model_dir, local_files_only=True)
        policy = AutoModelForCausalLM.from_pretrained(model_dir, local_files_only=True)
        reference = AutoModelForCausalLM.from_pretrained(ref_dir, local_files_only=True)

        collator = FrameworkCDRRewardCollator(
            tokenizer=tokenizer,
            mode=self.mode,
            order=self.order,
            include_scheme_token=self.include_scheme_token,
            max_length=self.max_length,
        )
        train_set = FrameworkCDRRewardDataset(tr_df, mode=self.mode, group_by=self.group_by)
        val_set = FrameworkCDRRewardDataset(va_df, mode=self.mode, group_by=self.group_by)

        train_loader = DataLoader(
            train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=collator,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            drop_last=False,
            collate_fn=collator,
        )

        module = WeightedDPOModule(
            model=policy,
            ref_model=reference,
            lr=self.learning_rate,
            beta=self.beta,
            weight_decay=self.weight_decay,
        )

        (out_dir / f"{self.exp_name}_config.json").write_text(json.dumps(self._config(), indent=2))

        checkpoint_cb = ModelCheckpoint(
            monitor="val_spearman_corr",
            mode="max",
            save_top_k=1,
            dirpath=str(out_dir),
            filename=f"{self.exp_name}" + "-epoch={epoch}-spearman={val_spearman_corr:.3f}",
            auto_insert_metric_name=False,
            save_last=True,
            every_n_epochs=1,
        )
        lr_monitor = LearningRateMonitor(logging_interval="step")

        devices = self.num_devices if self.num_devices is not None else "auto"
        trainer = pl.Trainer(
            accelerator="auto",
            devices=devices,
            max_epochs=self.max_epochs,
            default_root_dir=str(out_dir),
            precision=self.precision,
            log_every_n_steps=1,
            logger=CSVLogger(save_dir=str(out_dir), name=self.exp_name),
            callbacks=[checkpoint_cb, lr_monitor],
            check_val_every_n_epoch=1,
            gradient_clip_val=self.gradient_clip_val,
        )

        trainer.validate(module, val_loader)
        trainer.fit(module, train_loader, val_loader)

        logger.info("Saving aligned model + tokenizer to %s", out_dir)
        module.model.save_pretrained(str(out_dir))
        tokenizer.save_pretrained(str(out_dir))

        best = checkpoint_cb.best_model_path
        logger.info("Best checkpoint: %s", best)
        return best

    def _config(self) -> dict:
        """Return a JSON-serialisable snapshot of the run configuration."""
        return {k: v for k, v in vars(self).items() if isinstance(v, (str, int, float, bool, type(None)))}
