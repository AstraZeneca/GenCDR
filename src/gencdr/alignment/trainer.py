"""PyTorch Lightning module for scaffold-grouped weighted DPO alignment.

Implements exactly the published objective: a completion-only, beta-scaled mean
log-ratio against a frozen reference policy, combined with a scaffold-grouped
listwise cross-entropy over softmax-normalised rewards (see ``gencdr.alignment.loss``).

Requires the ``align`` extra (``pip install "gencdr[align]"``), which brings in
``pytorch-lightning`` and ``torchmetrics``. Import this module lazily so the core
package stays free of the training dependencies.
"""

from typing import Any, Dict, List

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import SGD, Adam, AdamW, Optimizer, RMSprop
from torchmetrics.regression import SpearmanCorrCoef
from torchmetrics.utilities.data import dim_zero_cat

from gencdr.alignment.loss import completion_mask_unshifted, grouped_weighted_dpo_loss


class WeightedDPOModule(pl.LightningModule):
    """Scaffold-grouped weighted DPO with a completion-only log-ratio.

    The sequence-level implicit reward is the beta-scaled mean per-token log-ratio
    over completion tokens only (those after ``<HPRED>``/``<LPRED>`` through ``<EOS>``,
    including inter-CDR ``<SEP>``). Framework prompt tokens are excluded. The batch is
    partitioned by scaffold group and the per-group listwise cross-entropy losses are
    aggregated weighted by group size.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        ref_model: torch.nn.Module,
        lr: float = 2e-6,
        beta: float = 0.15,
        opt_name: str = "adamw",
        weight_decay: float = 0.01,
    ) -> None:
        """Store the policy/frozen-reference models and optimisation hyperparameters."""
        super().__init__()
        # Avoid pickling the two large models into hyperparameters/checkpoints.
        self.save_hyperparameters(ignore=["model", "ref_model"])

        self.model = model
        self.model.train()

        self.ref_model = ref_model
        self.ref_model.eval()
        for p in self.ref_model.parameters():
            p.requires_grad = False

        self.lr = float(lr)
        self.beta = float(beta)
        self.opt_name = str(opt_name).lower()
        self.weight_decay = float(weight_decay)

        self.spearman_metric = SpearmanCorrCoef()

    # ------------------------------------------------------------------
    # Completion-only log-probabilities
    # ------------------------------------------------------------------

    def _token_logps_over_completion(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_spans: torch.Tensor,
        model: torch.nn.Module,
    ) -> tuple:
        """Return ``(token_logp, include)`` over shifted positions ``[B, L-1]``.

        ``token_logp`` is ``log p(y_t | y_<t)``; ``include`` is a boolean mask
        selecting completion tokens (including EOS) and excluding pads.
        """
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        log_probs = F.log_softmax(outputs.logits, dim=-1)

        shift_labels = input_ids[:, 1:]  # [B, L-1]
        shift_log_probs = log_probs[:, :-1, :]  # [B, L-1, V]
        shift_attn = attention_mask[:, 1:].bool()

        token_logp = shift_log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)  # [B, L-1]

        comp_unshifted = completion_mask_unshifted(completion_spans, seq_len=input_ids.size(1))  # [B, L]
        comp_shift = comp_unshifted[:, 1:]  # [B, L-1]

        include = shift_attn & comp_shift
        return token_logp, include

    def compute_completion_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        completion_spans: torch.Tensor,
        model: torch.nn.Module,
    ) -> torch.Tensor:
        """Return the mean per-sequence completion log-probability ``[B]``."""
        token_logp, include = self._token_logps_over_completion(input_ids, attention_mask, completion_spans, model)
        lengths = include.sum(dim=1).clamp(min=1)
        return (token_logp * include.float()).sum(dim=1) / lengths

    def _seq_log_ratio(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Return the per-sequence beta-scaled mean completion log-ratio ``[B]``."""
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        completion_spans = batch["completion_spans"]

        tok_logp_model, include = self._token_logps_over_completion(
            input_ids, attention_mask, completion_spans, self.model
        )
        with torch.no_grad():
            tok_logp_ref, _ = self._token_logps_over_completion(
                input_ids, attention_mask, completion_spans, self.ref_model
            )

        lengths = include.sum(dim=1).clamp(min=1).float()
        tok_log_ratio = tok_logp_model - tok_logp_ref  # [B, L-1]
        return self.beta * ((tok_log_ratio * include.float()).sum(dim=1) / lengths)  # [B]

    def compute_loss(self, batch: Dict[str, Any]) -> torch.Tensor:
        """Compute the scaffold-grouped weighted DPO loss for a batch."""
        seq_log_ratio = self._seq_log_ratio(batch)
        return grouped_weighted_dpo_loss(seq_log_ratio, batch["rewards"], batch["group_labels"])

    # ------------------------------------------------------------------
    # Train / validation hooks
    # ------------------------------------------------------------------

    def training_step(self, batch: Dict[str, Any], _: int) -> torch.Tensor:
        """Run one training step and log the loss."""
        loss = self.compute_loss(batch)
        self.log(
            "training_loss", loss, on_step=True, on_epoch=False, sync_dist=True, batch_size=batch["input_ids"].size(0)
        )
        return loss

    def validation_step(self, batch: Dict[str, Any], _: int) -> torch.Tensor:
        """Run one validation step; log the loss and update the Spearman metric."""
        loss = self.compute_loss(batch)
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_epoch=True,
            on_step=False,
            sync_dist=True,
            batch_size=batch["input_ids"].size(0),
        )

        logps = self.compute_completion_logps(
            batch["input_ids"], batch["attention_mask"], batch["completion_spans"], self.model
        )
        self.spearman_metric.update(logps, batch["rewards"].to(logps.device))
        return loss

    def on_validation_epoch_end(self) -> None:
        """Log the overall validation Spearman correlation (checkpoint selection metric)."""
        spearman = self.spearman_metric.compute()
        n_samples = dim_zero_cat(self.spearman_metric.preds).shape[0]
        self.log("val_spearman_corr", spearman, sync_dist=True, prog_bar=True)
        self.log("val_spearman_corr_n", float(n_samples), sync_dist=True, prog_bar=True)
        self.spearman_metric.reset()

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def get_optimizer(self, module: torch.nn.Module, opts: Dict[str, Any]) -> Optimizer:
        """Build an optimizer, excluding bias/norm parameters from weight decay."""
        assert opts["opt_name"] in {"adam", "adamw", "rmsprop", "sgd"}
        if "init_lr" not in opts:
            raise ValueError("get_optimizer requires 'init_lr' in opts (no default).")
        init_lr = float(opts["init_lr"])
        weight_decay = float(opts.get("weight_decay", 0.0))

        if weight_decay > 0.0:
            decay = [
                p for n, p in module.named_parameters() if p.requires_grad and ("bias" not in n) and ("norm" not in n)
            ]
            no_decay = [p for n, p in module.named_parameters() if p.requires_grad and (("bias" in n) or ("norm" in n))]
            params: List[Any] = [
                {"params": decay, "weight_decay": weight_decay},
                {"params": no_decay, "weight_decay": 0.0},
            ]
        else:
            params = [p for p in module.parameters() if p.requires_grad]

        opt_name = opts["opt_name"].lower()
        if opt_name == "adam":
            return Adam(params, lr=init_lr)
        if opt_name == "adamw":
            return AdamW(params, lr=init_lr)
        if opt_name == "sgd":
            return SGD(
                params, lr=init_lr, nesterov=opts.get("sgd_nesterov", False), momentum=opts.get("sgd_momentum", 0.0)
            )
        return RMSprop(params, alpha=opts.get("rmsprop_alpha", 0.95), lr=init_lr)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer for the policy model."""
        return self.get_optimizer(
            self.model,
            {"opt_name": self.opt_name, "init_lr": self.lr, "weight_decay": self.weight_decay},
        )
