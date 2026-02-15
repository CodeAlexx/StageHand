"""Numeric guards — BF16 enforcement and NaN/Inf detection.

These guards run in the transfer engine and after each block's forward
pass to ensure strict dtype discipline and catch numeric instability
early.  The goal is zero dtype promotions in strict mode and immediate
NaN/Inf visibility.

First target model: WAN 2.2.
"""
from __future__ import annotations

import logging

import torch

from stagehand.errors import DtypeMismatchError

__all__ = ["NumericGuard"]

logger = logging.getLogger(__name__)

# Dtype "precision" ordering for promotion detection.
# Higher value = higher precision.
_DTYPE_PRECISION: dict[torch.dtype, int] = {
    torch.float8_e5m2: 0,
    torch.float8_e4m3fn: 0,
    torch.bfloat16: 1,
    torch.float16: 1,
    torch.float32: 2,
    torch.float64: 3,
}


class NumericGuard:
    """BF16 enforcement and NaN/Inf detection.

    Parameters
    ----------
    strict_bf16:
        If True, :meth:`check_dtype` raises on any dtype mismatch.
    fail_on_dtype_promotion:
        If True, :meth:`check_promotion` raises when output precision
        exceeds input precision.
    nan_inf_check:
        If True, :meth:`check_output` scans for NaN/Inf values.
    """

    def __init__(
        self,
        strict_bf16: bool = True,
        fail_on_dtype_promotion: bool = True,
        nan_inf_check: bool = True,
    ) -> None:
        self._strict_bf16 = strict_bf16
        self._fail_on_dtype_promotion = fail_on_dtype_promotion
        self._nan_inf_check = nan_inf_check

    # ── dtype check ──────────────────────────────────────────────────

    def check_dtype(
        self,
        tensor: torch.Tensor,
        expected_dtype: torch.dtype,
        context: str,
    ) -> None:
        """Assert *tensor* has *expected_dtype*.

        Parameters
        ----------
        tensor:
            Tensor to check.
        expected_dtype:
            The dtype the tensor should have.
        context:
            Human-readable label for error messages (e.g. block_id).

        Raises
        ------
        DtypeMismatchError
            In strict mode when the dtype does not match.
        """
        if self._strict_bf16 and tensor.dtype != expected_dtype:
            raise DtypeMismatchError(
                f"[{context}] expected dtype {expected_dtype}, "
                f"got {tensor.dtype}"
            )

    # ── NaN / Inf check ──────────────────────────────────────────────

    def check_output(
        self,
        tensor: torch.Tensor,
        block_id: str,
        step: int,
    ) -> tuple[int, int]:
        """Scan *tensor* for NaN and Inf values.

        Returns
        -------
        (nan_count, inf_count)
            Number of NaN and Inf elements found.  Both are 0 when
            ``nan_inf_check`` is disabled.
        """
        if not self._nan_inf_check:
            return (0, 0)

        finite_mask = torch.isfinite(tensor)
        if finite_mask.all():
            return (0, 0)

        nan_count = int(torch.isnan(tensor).sum().item())
        inf_count = int(torch.isinf(tensor).sum().item())

        if nan_count > 0 or inf_count > 0:
            logger.warning(
                "Numeric instability in block %r at step %d: "
                "nan_count=%d, inf_count=%d, tensor shape=%s",
                block_id,
                step,
                nan_count,
                inf_count,
                list(tensor.shape),
            )

        return (nan_count, inf_count)

    # ── promotion check ──────────────────────────────────────────────

    def check_promotion(
        self,
        input_dtype: torch.dtype,
        output_dtype: torch.dtype,
        context: str,
    ) -> None:
        """Detect unwanted dtype promotion (output higher precision than input).

        Raises
        ------
        DtypeMismatchError
            When *fail_on_dtype_promotion* is True and output precision
            exceeds input precision.
        """
        if not self._fail_on_dtype_promotion:
            return

        input_prec = _DTYPE_PRECISION.get(input_dtype, -1)
        output_prec = _DTYPE_PRECISION.get(output_dtype, -1)

        # Only flag when we know both dtypes and output is strictly higher.
        if input_prec >= 0 and output_prec >= 0 and output_prec > input_prec:
            raise DtypeMismatchError(
                f"[{context}] dtype promotion detected: "
                f"{input_dtype} -> {output_dtype}"
            )

    def __repr__(self) -> str:
        return (
            f"NumericGuard(strict_bf16={self._strict_bf16}, "
            f"fail_on_promotion={self._fail_on_dtype_promotion}, "
            f"nan_inf={self._nan_inf_check})"
        )
