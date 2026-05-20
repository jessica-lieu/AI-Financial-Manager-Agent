"""
agent.py — Core LLM logic for the Planner → Executioner → Critic loop.

All functions accept their dependencies as arguments and return structured
data.  No ``print()`` statements — diagnostics go through the standard
``logging`` module so Streamlit can capture them.

Uses the Hugging Face Serverless Inference API via ``huggingface_hub``
instead of downloading model weights locally.
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Optional

from huggingface_hub import InferenceClient

from config import EXTRACTION_PROMPT, MAX_RETRIES, MODEL_PROVIDERS, build_rescan_prompt
from tools import (
    ReceiptData,
    RequiresHumanInterventionError,
    ValidationResult,
    validate_receipt,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _image_to_data_uri(image_path: str) -> str:
    """Read an image file and return a base64-encoded data URI.

    Parameters
    ----------
    image_path:
        Local file path to the image.

    Returns
    -------
    A ``data:<mime>;base64,<encoded>`` string suitable for the HF
    Inference API ``image_url`` field.
    """
    path = Path(image_path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    raw_bytes = path.read_bytes()
    b64 = base64.b64encode(raw_bytes).decode("utf-8")
    return f"data:{mime_type};base64,{b64}"


# ---------------------------------------------------------------------------
# VLM interaction (Planner / Executioner)
# ---------------------------------------------------------------------------

def ask_vlm(
    image_path: str,
    prompt: str,
    model_id: str,
    hf_token: str,
) -> Optional[dict]:
    """Send an image + prompt to the VLM via the HF Inference API and
    parse the JSON response.

    Parameters
    ----------
    image_path:
        Local file path to the receipt image.
    prompt:
        The text prompt instructing the VLM what to extract.
    model_id:
        The HuggingFace model identifier (e.g.
        ``"Qwen/Qwen2.5-VL-7B-Instruct"``).
    hf_token:
        A valid HuggingFace API token with Inference API access.

    Returns
    -------
    A parsed dict from the model's JSON output, or ``None`` if parsing
    fails.
    """
    provider = MODEL_PROVIDERS.get(model_id)
    client = InferenceClient(provider=provider, api_key=hf_token)

    image_uri = _image_to_data_uri(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": image_uri},
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    logger.info("Sending image to VLM (%s) via Inference API…", model_id)

    # Retry with exponential backoff on 429 (rate-limit) responses.
    max_api_retries = 3
    base_delay = 5  # seconds
    response = None
    for api_attempt in range(max_api_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model_id,
                messages=messages,
                max_tokens=2048,
            )
            break  # success
        except Exception as api_err:
            err_str = str(api_err)
            if "429" in err_str and api_attempt < max_api_retries:
                wait = base_delay * (2 ** api_attempt)
                logger.warning(
                    "Rate-limited (429). Retrying in %ds… (attempt %d/%d)",
                    wait, api_attempt + 1, max_api_retries,
                )
                time.sleep(wait)
            else:
                raise  # re-raise non-429 errors or final attempt

    if response is None:
        logger.warning("VLM call failed after retries.")
        return None

    text = response.choices[0].message.content or ""
    logger.debug("Raw VLM response: %s", text[:500])

    # Extract the first JSON object from the response.
    json_match = re.search(r"\{.*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            logger.warning("VLM returned invalid JSON — will retry if allowed.")
            return None

    logger.warning("No JSON object found in VLM response.")
    return None


# ---------------------------------------------------------------------------
# Orchestration: Planner → Executioner → Critic loop
# ---------------------------------------------------------------------------

def process_receipt(
    image_path: str,
    model_id: str,
    hf_token: str,
    status_callback=None,
) -> ReceiptData:
    """End-to-end receipt processing with closed-loop validation.

    Parameters
    ----------
    image_path:
        Path to the receipt image.
    model_id:
        HuggingFace model identifier for the Inference API.
    hf_token:
        A valid HuggingFace API token.
    status_callback:
        Optional callable ``(message: str) -> None`` that the caller can
        supply to receive progress updates (e.g. for ``st.status``).

    Returns
    -------
    A validated ``ReceiptData`` object.

    Raises
    ------
    RequiresHumanInterventionError
        If the totals cannot be reconciled after retries.
    ValueError
        If the VLM fails to return parseable JSON on all attempts.
    """
    def _update(msg: str) -> None:
        logger.info(msg)
        if status_callback:
            status_callback(msg)

    prompt = EXTRACTION_PROMPT
    receipt: Optional[ReceiptData] = None
    validation: Optional[ValidationResult] = None

    for attempt in range(MAX_RETRIES + 1):
        _update(f"Attempt {attempt + 1}/{MAX_RETRIES + 1}: Extracting data…")

        # --- Step 1: VLM extraction (Planner + Executioner) ---
        raw = ask_vlm(image_path, prompt, model_id, hf_token)
        if raw is None:
            if attempt < MAX_RETRIES:
                _update("VLM returned unparseable output — retrying…")
                prompt = build_rescan_prompt(EXTRACTION_PROMPT, 0.0)
                continue
            raise ValueError(
                "VLM failed to return parseable JSON after all attempts."
            )

        raw["raw_transcription"] = json.dumps(raw)
        raw["total_items_counted"] = len(raw.get("line_items", []))

        receipt = ReceiptData.model_validate(raw)

        # --- Step 2: Deterministic arithmetic via tool (Critic) ---
        validation = validate_receipt(receipt)
        _update(
            f"Validation: calculated=${validation.calculated_total:.2f}, "
            f"printed=${validation.printed_total:.2f}, "
            f"diff=${validation.difference:.2f}"
        )

        # --- Step 3: Check match ---
        if validation.is_match:
            _update("✅ Totals match — extraction complete.")
            return receipt

        # --- Step 4: Prepare retry with critic feedback ---
        if attempt < MAX_RETRIES:
            _update(
                f"Mismatch of ${abs(validation.difference):.2f} — "
                f"retrying with critic feedback…"
            )
            prompt = build_rescan_prompt(
                EXTRACTION_PROMPT, validation.difference
            )

    # --- Step 5: Human intervention fallback ---
    assert receipt is not None and validation is not None
    raise RequiresHumanInterventionError(
        message=(
            f"Discrepancy of ${abs(validation.difference):.2f} persists after "
            f"{MAX_RETRIES + 1} attempts. Requires manual review."
        ),
        receipt_data=receipt,
        validation=validation,
    )
