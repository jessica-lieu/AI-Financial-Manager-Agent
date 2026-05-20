"""
tools.py — Pure, deterministic functions and data models.

Nothing in this module maintains state, calls an LLM, or touches the network.
Everything here is fully testable without a GPU.
"""

from __future__ import annotations

import json
from decimal import Decimal
from typing import Optional

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class LineItem(BaseModel):
    """A single purchased item on the receipt."""
    item: str
    qty: float = 1.0
    unit_price: Optional[float] = None
    total_price: Optional[float] = None
    tag: Optional[str] = None

    @model_validator(mode='before')
    @classmethod
    def coerce_qty_none_to_one(cls, values):
        """Replace None with 1.0 for qty."""
        if values.get('qty') is None:
            values['qty'] = 1.0
        return values


class Financials(BaseModel):
    """Aggregate monetary fields printed on the receipt."""
    subtotal: Optional[float] = None
    tax: Optional[float] = 0.0
    tip: Optional[float] = 0.0
    grand_total: float

    @model_validator(mode='before')
    @classmethod
    def coerce_none_to_zero(cls, values):
        """Replace None with 0.0 for tax/tip so downstream math works."""
        for field in ('tax', 'tip'):
            if values.get(field) is None:
                values[field] = 0.0
        return values


class ReceiptData(BaseModel):
    """Full structured payload extracted from a receipt image."""
    raw_transcription: str
    total_items_counted: int
    merchant_name: Optional[str] = None
    date: Optional[str] = None
    line_items: list[LineItem] = Field(default_factory=list)
    financials: Financials


class ValidationResult(BaseModel):
    """Result of comparing the computed total to the printed total."""
    is_match: bool
    calculated_total: float
    printed_total: float
    difference: float


# ---------------------------------------------------------------------------
# Custom exception
# ---------------------------------------------------------------------------

class RequiresHumanInterventionError(Exception):
    """Raised when automated extraction cannot reconcile totals."""

    def __init__(
        self,
        message: str,
        receipt_data: ReceiptData,
        validation: ValidationResult,
    ) -> None:
        super().__init__(message)
        self.receipt_data = receipt_data
        self.validation = validation


# ---------------------------------------------------------------------------
# Pure deterministic functions
# ---------------------------------------------------------------------------

def calculate_actual_total(line_items: list[float]) -> float:
    """Sum line-item prices using Decimal to avoid float rounding errors.

    This is the *tool* the system calls instead of letting the LLM
    hallucinate arithmetic.
    """
    total = sum(Decimal(str(p)) for p in line_items)
    return float(total)


def validate_receipt(receipt: ReceiptData) -> ValidationResult:
    """Compare deterministic sum of line items against the printed total.

    Parameters
    ----------
    receipt:
        The parsed receipt data whose line items will be summed.

    Returns
    -------
    A ``ValidationResult`` indicating whether the computed total matches
    the printed grand total (within a ±$0.01 tolerance).
    """
    prices = [item.total_price for item in receipt.line_items
              if item.total_price is not None]
    computed = Decimal(str(calculate_actual_total(prices)))
    computed += Decimal(str(receipt.financials.tax))
    computed += Decimal(str(receipt.financials.tip))

    printed = Decimal(str(receipt.financials.grand_total))
    diff = float(computed - printed)

    return ValidationResult(
        is_match=abs(diff) < 0.01,
        calculated_total=float(computed),
        printed_total=float(printed),
        difference=diff,
    )
