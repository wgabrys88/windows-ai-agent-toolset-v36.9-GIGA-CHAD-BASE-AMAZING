Updated files + a safe “fake execution” mode **by default**, plus a **sticky, relative-to-current-view Focus workflow** that keeps coordinate conversion correct across turns.

---

## What was fixed (minimal, targeted patches)

### 1) `execute.py` — “fake” actions by default + focus-aware coordinate mapping

* **Default = NO real mouse/keyboard movement.**
  Real input only happens if you set `FRANZ_PHYSICAL_EXECUTION=1` (or pass `physical_execution=true` in the JSON input).
* Added `focus_exec` and `focus_capture` inputs:

  * `focus_exec`: the coordinate space the model used when it produced the actions (so clicks/drags map correctly to the real screen).
  * `focus_capture`: the region `capture.py` should crop for the *next screenshot*.
* **Important:** `focus()` lines from model output are **not** forwarded directly to `capture.py` (because model focus is *relative-to-current-view*), instead we compute the correct absolute crop and inject a synthetic `focus(...)` for capture.
* Actions are still recorded/annotated as the model wrote them (so debugging stays faithful).

### 2) `main.py` — narrative-only memory + sticky focus across turns

* Keeps **only the `NARRATIVE:` section** as “memory” sent back to the VLM, while still storing the full raw response for next-turn execution.
* Introduces `PipelineState.focus` as a **sticky view**:

  * `focus()` is interpreted as **relative to the CURRENT screenshot**, and becomes the view for subsequent screenshots until changed.
  * `focus(0,0,1000,1000)` resets to full screen.
* On each turn:

  * `focus_exec = state.focus` (used to remap actions correctly)
  * `focus_capture = next_focus(state.focus, state.last_raw)` (used to crop the next screenshot)

### 3) `panel.py` — UI matches the “1 / 2” expectation + editable hyperparams + dark theme

* **Bottom response editor shows the assistant “content” only**, not raw JSON.
* Raw JSON moved into a **collapsible** section (“Show Raw JSON”).
* Added **hyperparameter inputs** (model, temperature, top_p, max_tokens, freq/presence penalties). Values are injected into the outgoing request.
* Dark theme: better contrast (no “blue-on-blue”).
* Added quick **focus** and **reset_focus** insert buttons (and focus in manual/response helpers).

### 4) `capture.py`

* No behavioral changes needed here (it already supports focus + annotation correctly under Python 3.13).

---

## How to use it

### Safe debugging (no real input)

Do nothing—**fake mode is the default**.

### Enable real input when ready

Set:

* `FRANZ_PHYSICAL_EXECUTION=1`

### Focus usage (works across turns, correctly mapped)

* Narrow view: `focus(100,100,900,900)`
* Reset: `focus(0,0,1000,1000)`
* Focus coords are **relative to the current screenshot** (the one the model is looking at).

---

