# Specs — Version 1

## Overview

Two focused additions to the existing PDF Q&A Streamlit app. No changes to the ingestion pipeline, vector store, or retrieval logic. All changes are in `app.py` and `query.py`.

---

## Feature 1 — Conversation Memory

### What it does
Each question carries the prior N turns of Q&A as additional context so the LLM can answer follow-up questions coherently (e.g. "What did you mean by that?" or "Tell me more about the second point").

### Current state
`st.session_state.messages` stores the chat history for display only. Each call to `query_document` is stateless — it passes only the retrieved chunks and the current question to the LLM.

### What changes

**`query.py`**
- Add a `chat_history` parameter to `query_document`.
- Rewrite the prompt template to include a `{chat_history}` slot above the `{question}` slot. Format it as alternating `Human: …` / `Assistant: …` lines, capped at the last **6 turns** (3 pairs) to stay within context limits.
- No changes to retrieval — chunks are still fetched for the current question only.

**`app.py`**
- On each user submission, pass `st.session_state.messages` (excluding source metadata) into `query_document`.
- Keep the existing display loop unchanged.

### Constraints
- Memory is per-session (in-memory, Streamlit session state). Reloading the tab or uploading a new PDF resets it.
- History is trimmed to the last 6 turns before formatting to avoid prompt bloat.
- No persistent cross-session memory in this version.

---

## Feature 2 — Model Comparison

### What it does
A toggle in the sidebar switches the app into **Compare mode**. In this mode, every question is sent to two models simultaneously — **Claude Haiku** and a user-selected free model — and their answers are displayed side by side with separate token/cost rows.

### Current state
One model is selected from a dropdown and used for every query. No parallel calls.

### What changes

**`app.py`**
- Add a `st.toggle("Compare mode")` in the sidebar under the model selector.
- When compare mode is **off**: existing single-model behaviour, unchanged.
- When compare mode is **on**:
  - Hide the single model selector; instead show a static label "Claude Haiku vs Free model" and a second selectbox listing only the free/cheap tier: `meta-llama/llama-3.1-8b-instruct:free` (default), `google/gemini-flash-1.5`.
  - On each query, build two separate LLM instances (one for `anthropic/claude-3-haiku`, one for the selected free model) and call `query_document` for each. Run them **sequentially** (not in threads) to avoid complexity.
  - Render responses in a two-column layout (`st.columns(2)`):
    - Left column: Claude Haiku answer + its sources expander.
    - Right column: free model answer + its sources expander.
  - Store both responses in `st.session_state.messages` as a single entry with a `"comparison": True` flag and both payloads, so the display loop renders them correctly on re-render.

**`query.py`**
- No new functions needed; `build_rag_chain` and `query_document` already accept an arbitrary model name.

### Constraints
- Compare mode only pairs Claude Haiku against a free model (no arbitrary pair selection in this version).
- Calls are sequential, so the second model's answer appears after the first finishes. A spinner is shown for each.
- Conversation memory in compare mode uses only the Haiku side of prior turns as the shared history (simplest approach; avoids ambiguity about which model's answer to pass forward).

---

## Files changed

| File | Changes |
|---|---|
| `app.py` | Compare mode toggle, dual-column layout, history passed to query |
| `query.py` | `chat_history` param |
| `specs-version-1.md` | This file (new) |

No new dependencies beyond what is already in `requirements.txt`.
