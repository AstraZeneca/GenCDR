"""Scaffold-grouped weighted DPO loss (paper-faithful).

This is the alignment objective from the GenCDR manuscript (weighted DPO,
listwise cross-entropy over softmax-normalised scalar rewards, aggregated across
framework scaffolds). The two functions here are pure (no model, no Lightning) so
the loss math can be unit-tested in isolation; the model forward pass and the
completion-only log-ratio live in ``gencdr.alignment.trainer``.

Given a batch of prompt/response/reward tuples partitioned into scaffold groups
``G``, with per-sequence beta-scaled mean completion log-ratios ``s`` used as the
listwise logits and softmax-normalised rewards ``w`` as the soft targets:

    L = sum_{g in G} (|g| / B) * CrossEntropy( s[g], softmax(reward[g]) )

Only the default weighted cross-entropy objective is implemented. The internal
PPO-style ratio clipping and the ACCEPT-vs-reject preference-ranking variant are
deliberately excluded, as they are not part of the published method.
"""

from typing import List

import torch
import torch.nn.functional as F


def completion_mask_unshifted(completion_spans: torch.Tensor, seq_len: int) -> torch.Tensor:
    """Build a boolean completion mask from ``[B, 2]`` (start, end-exclusive) spans.

    Parameters
    ----------
    completion_spans : torch.Tensor
        ``[B, 2]`` long tensor of unshifted (start, end-exclusive) token positions.
    seq_len : int
        Sequence length ``L`` the mask is built for.

    Returns
    -------
    torch.Tensor
        ``[B, L]`` boolean mask, True on completion positions. Invalid ranges
        (end <= start) and out-of-bounds values yield all-False rows.
    """
    device = completion_spans.device
    positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, L]
    starts = completion_spans[:, 0].unsqueeze(1).clamp(min=0, max=seq_len)  # [B, 1]
    ends = completion_spans[:, 1].unsqueeze(1).clamp(min=0, max=seq_len)  # [B, 1]
    return (positions >= starts) & (positions < ends)


def grouped_weighted_dpo_loss(
    seq_logits: torch.Tensor,
    rewards: torch.Tensor,
    group_labels: List[str],
) -> torch.Tensor:
    """Compute the scaffold-grouped weighted DPO cross-entropy loss.

    Parameters
    ----------
    seq_logits : torch.Tensor
        ``[B]`` per-sequence beta-scaled mean completion log-ratios (the listwise logits).
    rewards : torch.Tensor
        ``[B]`` scalar rewards.
    group_labels : list of str
        Length-``B`` scaffold group label per sequence. Sequences sharing a label
        form one listwise cross-entropy; groups are aggregated weighted by size.

    Returns
    -------
    torch.Tensor
        Scalar loss.
    """
    batch_size = int(seq_logits.shape[0])
    if batch_size == 0:
        return torch.zeros((), device=seq_logits.device, dtype=seq_logits.dtype)

    rewards = rewards.to(seq_logits.device)

    groups: dict[str, List[int]] = {}
    for i, g in enumerate(group_labels):
        groups.setdefault(g, []).append(i)

    loss = torch.zeros((), device=seq_logits.device, dtype=seq_logits.dtype)
    for idxs in groups.values():
        k = len(idxs)
        idx_t = torch.tensor(idxs, device=seq_logits.device, dtype=torch.long)
        g_logits = seq_logits.index_select(0, idx_t)  # [k]
        g_rewards = rewards.index_select(0, idx_t)  # [k]
        g_weights = torch.softmax(g_rewards, dim=0)  # [k]
        g_loss = F.cross_entropy(g_logits.unsqueeze(0), g_weights.unsqueeze(0), reduction="mean")
        loss = loss + (k / batch_size) * g_loss

    return loss
