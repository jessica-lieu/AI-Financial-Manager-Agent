"""
config.py — Prompt templates, model identifiers, and pipeline constants.

All configuration that the agent and Streamlit app reference lives here.
No runtime state, no LLM calls, no side effects.
"""

# ---------------------------------------------------------------------------
# Model identifiers (confirmed available via HF Inference Providers)
# ---------------------------------------------------------------------------
GEMMA_ID = "google/gemma-4-31B-it"
QWEN35_35B_ID = "Qwen/Qwen3.5-35B-A3B"
LLAMA_ID = "meta-llama/Llama-4-Scout-17B-16E-Instruct"

DEFAULT_MODEL_ID = GEMMA_ID

MODEL_CHOICES: dict[str, str] = {
    "Gemma 4 31B": GEMMA_ID,
    "Qwen3.5-35B-A3B": QWEN35_35B_ID,
    "Llama 4 Scout 17B": LLAMA_ID,
}

# Maps model IDs to a known working serverless provider.
MODEL_PROVIDERS: dict[str, str] = {
    GEMMA_ID: "novita",
    QWEN35_35B_ID: "novita",
    LLAMA_ID: "novita",
}

# ---------------------------------------------------------------------------
# Item category tags (used in the extraction prompt and the UI editor)
# ---------------------------------------------------------------------------
ITEM_TAGS: list[str] = [
    "Groceries",
    "Dining",
    "Electronics",
    "Clothing",
    "Office Supplies",
    "Health & Beauty",
    "Household",
    "Beverages",
    "Snacks",
    "Entertainment",
    "Transportation",
    "Other",
]

# ---------------------------------------------------------------------------
# Pipeline constants
# ---------------------------------------------------------------------------
MAX_RETRIES = 1

# ---------------------------------------------------------------------------
# Extraction prompt (sent to the VLM alongside the receipt image)
# ---------------------------------------------------------------------------
EXTRACTION_PROMPT = """\
Analyze the receipt image and return ONLY a JSON object (no markdown fences).
IMPORTANT — follow these steps IN ORDER to avoid skipping line items:

1. Transcribe: Read every line of text on the receipt from top to bottom,
    exactly as printed, and write it into the "raw_transcription" field.
    Include item names, prices, subtotals, tax lines — everything.
2. Count: Count the number of purchased line items (not subtotal/tax/total
    lines) and write it into "total_items_counted".
3. Structure: Using ONLY the transcription above, fill in the
    "line_items" and "financials" fields.  The number of objects in
    "line_items" MUST equal "total_items_counted".
4. Tag: For each line item, assign a financial category "tag" from this
    list: "Groceries", "Dining", "Electronics", "Clothing",
    "Office Supplies", "Health & Beauty", "Household", "Beverages",
    "Snacks", "Entertainment", "Transportation", "Other".
    Choose the single best-fitting tag based on the item name.

Doing the transcription and count first prevents you from accidentally skipping any items.
  
{
  "merchant_name": "",
  "date": "",
  "line_items": [
    {"item": "", "qty": 0, "unit_price": 0.00, "total_price": 0.00, "tag": ""}
  ],
  "financials": {
    "subtotal": 0.00,
    "tax": 0.00,
    "tip": 0.00,
    "grand_total": 0.00
  }
}

Rules:
- Extract every purchased line item with its name, quantity, unit price, and
  total price.
- Pay close attention to quantities. If a line shows "2 @", "x6", "QTY 6",
  or a multiplier before/after the item name, set "qty" to that number and
  "unit_price" to the per-unit price.  "total_price" = qty * unit_price. 
  Do NOT put the line total in "unit_price" — "unit_price" is always the per-single-item price.
- If an item appears with no quantity indicator, default qty to 1.
- Identify the printed grand total from the footer.
- Use null for any field you cannot read.
- Do NOT guess arithmetic; just transcribe what is printed.
- Every line item MUST have a "tag" from the allowed list above.
"""


def build_rescan_prompt(base_prompt: str, difference: float) -> str:
    """Append critic feedback for the retry attempt.

    Parameters
    ----------
    base_prompt:
        The original extraction prompt to augment.
    difference:
        computed_total - printed_total.  Negative means the extraction
        overshoots; positive means it undershoots.

    Returns
    -------
    A new prompt string containing the original instructions plus
    specific guidance on what the model likely got wrong.
    """
    abs_diff = abs(difference)
    if difference < 0:
        guidance = (
            f"Your extracted line-item total is ${abs_diff:.2f} HIGHER than the "
            f"printed grand total. You likely hallucinated an extra item or "
            f"read a price too high. Look for a line item that should be "
            f"removed or whose price should be reduced by ${abs_diff:.2f}."
        )
    else:
        guidance = (
            f"Your extracted line-item total is ${abs_diff:.2f} LOWER than the "
            f"printed grand total. You likely missed a line item. Look "
            f"specifically for an item or a combination of items that cost "
            f"exactly ${abs_diff:.2f}."
        )
    return (
        base_prompt
        + f"\n\nCRITIC FEEDBACK: The previous extraction had a discrepancy "
        + f"of ${abs_diff:.2f}. {guidance} "
        + f"Please re-examine the receipt carefully, redo the full "
        + f"transcription, recount, and correct the structured output."
    )


# ---------------------------------------------------------------------------
# Chat system prompt (Financial Chat Assistant — Feature 4)
# ---------------------------------------------------------------------------
CHAT_SYSTEM_PROMPT = """\
You are a helpful financial advisor assistant. The user has uploaded receipts
that have been digitized into structured data. Use this data to answer their
questions about spending, budgeting, and purchase history.

**Your receipt data is below (JSON):**
{receipt_context}

**Capabilities:**
- Summarise total spending across all receipts or a specific time period.
- Break down spending by category tag (e.g. Groceries, Dining, Electronics).
- Identify the most/least expensive purchases.
- Answer questions like "How much did I spend on groceries this month?"
- Provide simple budgeting advice based on the data.

**Rules:**
- Only reference data that actually exists in the receipt context above.
- If the user asks about data you don't have, say so honestly.
- Format monetary values as $X.XX.
- Be concise but thorough.
"""

