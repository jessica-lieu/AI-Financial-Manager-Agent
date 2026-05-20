"""
app.py — Streamlit frontend for the AI Receipt Processing Agent.

Imports from ``agent.py`` (LLM orchestration), ``tools.py`` (pure
functions / models), and ``storage.py`` (SQLite persistence).

Uses ``st.session_state`` to persist receipt data across re-runs so the
LLM is not invoked on every widget interaction.

All model inference is offloaded to the Hugging Face Serverless Inference
API — no local model weights are downloaded.
"""

from __future__ import annotations

import json
import logging
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st
from huggingface_hub import InferenceClient

from agent import process_receipt
from config import (
    CHAT_SYSTEM_PROMPT,
    ITEM_TAGS,
    MODEL_CHOICES,
    MODEL_PROVIDERS,
)
from storage import (
    get_all_receipts,
    get_all_receipts_with_items,
    get_receipt_detail,
    init_db,
    save_receipt,
)
from tools import (
    ReceiptData,
    RequiresHumanInterventionError,
    ValidationResult,
    calculate_actual_total,
    validate_receipt,
)

# ---------------------------------------------------------------------------
# Logging — route agent diagnostics into Streamlit's status containers
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)s  %(levelname)s  %(message)s",
)

# ---------------------------------------------------------------------------
# Initialise persistent storage
# ---------------------------------------------------------------------------
init_db()

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="AI Receipt Manager",
    page_icon="🧾",
    layout="wide",
)

st.title("🧾 AI Financial Manager")
st.caption(
    "Upload receipts, track spending by category, review history, "
    "and chat with an AI financial advisor — all in one place."
)


# ---------------------------------------------------------------------------
# Session-state defaults (run once)
# ---------------------------------------------------------------------------
_DEFAULTS: dict[str, object] = {
    # Upload tab
    "receipt_data": None,           # ReceiptData dict (serialised)
    "validation": None,             # ValidationResult dict
    "needs_review": False,          # True ⇒ show the data editor
    "processing_complete": False,
    "uploaded_file_name": None,     # track which file produced current data
    "uploaded_file_bytes": None,    # raw bytes for saving later
    "status_messages": [],          # progress log from the agent
    "receipt_saved": False,         # True after the receipt is saved
    # Chat tab
    "chat_messages": [],            # list of {"role": ..., "content": ...}
}

for key, default in _DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar — API key & model selection
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")

    hf_token = st.text_input(
        "HuggingFace API Token",
        type="password",
        help="Required for the Inference API. "
             "Get yours at https://huggingface.co/settings/tokens",
    )

    model_name = st.selectbox(
        "Model",
        options=list(MODEL_CHOICES.keys()),
        index=0,
    )
    model_id = MODEL_CHOICES[model_name]

    st.divider()
    st.markdown(
        "**How it works**\n\n"
        "1. 📤 **Upload** a receipt image\n"
        "2. The agent extracts & tags line items via HF Inference API\n"
        "3. Arithmetic is verified deterministically (Critic)\n"
        "4. 💾 **Save** receipts to your local database\n"
        "5. 📜 **History** — browse past receipts\n"
        "6. 💬 **Chat** — ask an AI about your spending"
    )


# ═══════════════════════════════════════════════════════════════════════════
# Tab layout
# ═══════════════════════════════════════════════════════════════════════════
tab_upload, tab_history, tab_chat = st.tabs([
    "📤 Upload & Process",
    "📜 Receipt History",
    "💬 Financial Chat",
])


# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — Upload & Process
# ═══════════════════════════════════════════════════════════════════════════
with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a receipt image",
        type=["jpg", "jpeg", "png", "webp"],
        help="Supports JPG, PNG, and WebP receipt photos.",
    )

    if uploaded_file:
        col_img, col_data = st.columns([1, 2])

        with col_img:
            st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)

        # Detect if the user uploaded a *new* file → reset state
        if uploaded_file.name != st.session_state.uploaded_file_name:
            st.session_state.receipt_data = None
            st.session_state.validation = None
            st.session_state.needs_review = False
            st.session_state.processing_complete = False
            st.session_state.uploaded_file_name = uploaded_file.name
            st.session_state.uploaded_file_bytes = uploaded_file.getvalue()
            st.session_state.status_messages = []
            st.session_state.receipt_saved = False

        # ---------------------------------------------------------------
        # "Process Receipt" button
        # ---------------------------------------------------------------
        with col_data:
            if not st.session_state.processing_complete:
                if not hf_token:
                    st.warning(
                        "Enter your HuggingFace API token in the sidebar "
                        "before processing."
                    )
                else:
                    process_btn = st.button(
                        "🔍 Process Receipt",
                        use_container_width=True,
                        type="primary",
                    )

                    if process_btn:
                        # Write uploaded bytes to a temp file for the agent
                        suffix = Path(uploaded_file.name).suffix
                        with tempfile.NamedTemporaryFile(
                            suffix=suffix, delete=False
                        ) as tmp:
                            tmp.write(uploaded_file.getvalue())
                            tmp_path = tmp.name

                        status_msgs: list[str] = []

                        with st.status(
                            "Processing receipt…", expanded=True
                        ) as status:
                            def _on_status(msg: str) -> None:
                                status_msgs.append(msg)
                                st.write(msg)

                            try:
                                receipt = process_receipt(
                                    tmp_path,
                                    model_id,
                                    hf_token,
                                    status_callback=_on_status,
                                )
                                st.session_state.receipt_data = receipt.model_dump()
                                validation = validate_receipt(receipt)
                                st.session_state.validation = validation.model_dump()
                                st.session_state.needs_review = False
                                st.session_state.processing_complete = True
                                status.update(
                                    label="✅ Receipt processed successfully!",
                                    state="complete",
                                )

                            except RequiresHumanInterventionError as exc:
                                st.session_state.receipt_data = (
                                    exc.receipt_data.model_dump()
                                )
                                st.session_state.validation = (
                                    exc.validation.model_dump()
                                )
                                st.session_state.needs_review = True
                                st.session_state.processing_complete = True
                                status.update(
                                    label="⚠️ Needs human review",
                                    state="error",
                                )

                            except Exception as exc:
                                st.error(f"Processing failed: {exc}")
                                status.update(
                                    label="❌ Processing failed",
                                    state="error",
                                )
                                st.session_state.status_messages = status_msgs
                                st.stop()

                        st.session_state.status_messages = status_msgs
                        st.rerun()

        # ---------------------------------------------------------------
        # Display results (persisted in session_state)
        # ---------------------------------------------------------------
        if st.session_state.processing_complete and st.session_state.receipt_data:
            data = st.session_state.receipt_data
            val = st.session_state.validation

            st.divider()

            # Summary metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Merchant", data.get("merchant_name") or "—")
            mcol2.metric("Date", data.get("date") or "—")
            mcol3.metric(
                "Grand Total",
                f"${data['financials']['grand_total']:.2f}",
            )
            mcol4.metric(
                "Items",
                data.get("total_items_counted", len(data.get("line_items", []))),
            )

            # Warning banner if review is needed
            if st.session_state.needs_review:
                st.warning(
                    f"⚠️ **Human review required** — the calculated total "
                    f"(${val['calculated_total']:.2f}) differs from the printed "
                    f"total (${val['printed_total']:.2f}) by "
                    f"${abs(val['difference']):.2f}.  "
                    f"Edit the table below and click **Re-validate**.",
                    icon="✏️",
                )

            # Build a DataFrame for the data editor
            line_items = data.get("line_items", [])
            df = pd.DataFrame(line_items)
            if df.empty:
                df = pd.DataFrame(
                    columns=["item", "qty", "unit_price", "total_price", "tag"]
                )

            st.subheader("Line Items")
            edited_df = st.data_editor(
                df,
                num_rows="dynamic",
                use_container_width=True,
                column_config={
                    "item": st.column_config.TextColumn("Item"),
                    "qty": st.column_config.NumberColumn(
                        "Qty", min_value=0.0, step=0.01, format="%.2f"
                    ),
                    "unit_price": st.column_config.NumberColumn(
                        "Unit Price ($)", min_value=0.0, format="%.2f"
                    ),
                    "total_price": st.column_config.NumberColumn(
                        "Total Price ($)", min_value=0.0, format="%.2f"
                    ),
                    "tag": st.column_config.SelectboxColumn(
                        "Category",
                        options=ITEM_TAGS,
                        required=False,
                    ),
                },
                key="line_items_editor",
            )

            # Financials editor
            fin = data["financials"]
            st.subheader("Financials")
            fcol1, fcol2, fcol3, fcol4 = st.columns(4)
            new_subtotal = fcol1.number_input(
                "Subtotal ($)", value=fin.get("subtotal") or 0.0,
                format="%.2f", key="fin_subtotal",
            )
            new_tax = fcol2.number_input(
                "Tax ($)", value=fin.get("tax", 0.0),
                format="%.2f", key="fin_tax",
            )
            new_tip = fcol3.number_input(
                "Tip ($)", value=fin.get("tip", 0.0),
                format="%.2f", key="fin_tip",
            )
            new_grand = fcol4.number_input(
                "Grand Total ($)", value=fin["grand_total"],
                format="%.2f", key="fin_grand",
            )

            # Action buttons
            btn_col1, btn_col2 = st.columns(2)

            # Re-validate button
            with btn_col1:
                if st.button("🔄 Re-validate", use_container_width=True):
                    # Rebuild receipt data from edited values
                    updated_items = edited_df.to_dict(orient="records")
                    data["line_items"] = updated_items
                    data["total_items_counted"] = len(updated_items)
                    data["financials"]["subtotal"] = new_subtotal
                    data["financials"]["tax"] = new_tax
                    data["financials"]["tip"] = new_tip
                    data["financials"]["grand_total"] = new_grand

                    # Re-run deterministic validation
                    receipt_obj = ReceiptData.model_validate(data)
                    new_val = validate_receipt(receipt_obj)

                    st.session_state.receipt_data = data
                    st.session_state.validation = new_val.model_dump()
                    st.session_state.needs_review = not new_val.is_match
                    st.rerun()

            # Save button
            with btn_col2:
                if st.session_state.receipt_saved:
                    st.success("✅ Receipt saved!", icon="💾")
                else:
                    if st.button(
                        "💾 Save Receipt",
                        use_container_width=True,
                        type="primary",
                    ):
                        # Sync any edits back before saving
                        updated_items = edited_df.to_dict(orient="records")
                        data["line_items"] = updated_items
                        data["total_items_counted"] = len(updated_items)
                        data["financials"]["subtotal"] = new_subtotal
                        data["financials"]["tax"] = new_tax
                        data["financials"]["tip"] = new_tip
                        data["financials"]["grand_total"] = new_grand
                        st.session_state.receipt_data = data

                        receipt_id = save_receipt(
                            receipt_data=data,
                            image_bytes=st.session_state.uploaded_file_bytes,
                            image_name=st.session_state.uploaded_file_name,
                        )
                        st.session_state.receipt_saved = True
                        st.rerun()

            # Validation status
            if val and val.get("is_match"):
                st.success(
                    f"✅ **Validated** — Calculated total "
                    f"(${val['calculated_total']:.2f}) matches the printed "
                    f"total (${val['printed_total']:.2f}).",
                    icon="✅",
                )
            elif val:
                st.error(
                    f"❌ **Mismatch** — Calculated: ${val['calculated_total']:.2f} "
                    f"vs Printed: ${val['printed_total']:.2f} "
                    f"(diff: ${val['difference']:.2f})",
                    icon="❌",
                )

            # Raw JSON expander
            with st.expander("📋 Raw extracted JSON"):
                st.json(data)

    else:
        st.info("👆 Upload a receipt image to get started.", icon="📤")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — Receipt History
# ═══════════════════════════════════════════════════════════════════════════
with tab_history:
    st.subheader("📜 Saved Receipts")

    receipts = get_all_receipts()

    if not receipts:
        st.info(
            "No receipts saved yet. Upload and process a receipt in the "
            "**Upload & Process** tab, then click **💾 Save Receipt**.",
            icon="📭",
        )
    else:
        st.caption(f"{len(receipts)} receipt(s) on file")

        for r in receipts:
            label = (
                f"**{r['merchant'] or 'Unknown Merchant'}** — "
                f"{r['date'] or 'No date'} — "
                f"${r['grand_total']:.2f}  "
                f"({r['item_count']} items)"
            )
            with st.expander(label):
                detail = get_receipt_detail(r["id"])
                if detail is None:
                    st.error("Could not load receipt details.")
                    continue

                detail_col1, detail_col2 = st.columns([1, 2])

                # Show the original image
                with detail_col1:
                    img_bytes = detail["receipt"].get("image_blob")
                    if img_bytes:
                        st.image(
                            img_bytes,
                            caption=detail["receipt"].get("image_name", "Receipt"),
                            use_container_width=True,
                        )
                    else:
                        st.caption("No image stored.")

                # Show line items + financials
                with detail_col2:
                    items_df = pd.DataFrame(detail["line_items"])
                    if not items_df.empty:
                        st.dataframe(
                            items_df,
                            use_container_width=True,
                            column_config={
                                "item": "Item",
                                "qty": "Qty",
                                "unit_price": st.column_config.NumberColumn(
                                    "Unit Price ($)", format="%.2f"
                                ),
                                "total_price": st.column_config.NumberColumn(
                                    "Total ($)", format="%.2f"
                                ),
                                "tag": "Category",
                            },
                        )
                    else:
                        st.caption("No line items recorded.")

                    # Financials summary
                    rcpt = detail["receipt"]
                    fin_cols = st.columns(4)
                    fin_cols[0].metric(
                        "Subtotal", f"${rcpt.get('subtotal') or 0:.2f}"
                    )
                    fin_cols[1].metric(
                        "Tax", f"${rcpt.get('tax') or 0:.2f}"
                    )
                    fin_cols[2].metric(
                        "Tip", f"${rcpt.get('tip') or 0:.2f}"
                    )
                    fin_cols[3].metric(
                        "Grand Total", f"${rcpt.get('grand_total') or 0:.2f}"
                    )

                st.caption(f"Saved on: {r['created_at']}")


# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — Financial Chat
# ═══════════════════════════════════════════════════════════════════════════
with tab_chat:
    st.subheader("💬 Financial Chat Assistant")
    st.caption(
        "Ask questions about your spending, budgets, and purchase history. "
        "The assistant has access to all your saved receipts."
    )

    # Check prerequisites
    all_receipt_data = get_all_receipts_with_items()

    if not all_receipt_data:
        st.info(
            "No receipts saved yet. Upload, process, and **save** at least "
            "one receipt before using the chat assistant.",
            icon="💡",
        )
    elif not hf_token:
        st.warning(
            "Enter your HuggingFace API token in the sidebar to use the "
            "chat assistant."
        )
    else:
        # Display chat history
        for msg in st.session_state.chat_messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        # Chat input
        user_input = st.chat_input("Ask about your spending…")

        if user_input:
            # Append user message
            st.session_state.chat_messages.append(
                {"role": "user", "content": user_input}
            )
            with st.chat_message("user"):
                st.markdown(user_input)

            # Build context-injected system prompt
            receipt_context = json.dumps(all_receipt_data, indent=2)
            system_prompt = CHAT_SYSTEM_PROMPT.format(
                receipt_context=receipt_context
            )

            # Build messages for the API
            api_messages = [
                {"role": "system", "content": system_prompt},
            ]
            for msg in st.session_state.chat_messages:
                api_messages.append(
                    {"role": msg["role"], "content": msg["content"]}
                )

            # Call the LLM
            with st.chat_message("assistant"):
                with st.spinner("Thinking…"):
                    try:
                        provider = MODEL_PROVIDERS.get(model_id)
                        client = InferenceClient(
                            provider=provider, api_key=hf_token
                        )
                        response = client.chat.completions.create(
                            model=model_id,
                            messages=api_messages,
                            max_tokens=2048,
                        )
                        assistant_text = (
                            response.choices[0].message.content or ""
                        )
                    except Exception as exc:
                        assistant_text = (
                            f"Sorry, I encountered an error: {exc}"
                        )

                st.markdown(assistant_text)

            # Save assistant response
            st.session_state.chat_messages.append(
                {"role": "assistant", "content": assistant_text}
            )

        # Clear chat button
        if st.session_state.chat_messages:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.chat_messages = []
                st.rerun()
