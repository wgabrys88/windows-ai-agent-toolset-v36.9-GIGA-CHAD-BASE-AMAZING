"""
FRANZ -- Visual AI Agent Loop for Windows 11

Orchestrator that drives a vision-language model in a closed loop:
receive VLM output, pass to executor (which runs actions and captures
an annotated screenshot), send the VLM's own output back as user context
alongside the fresh screenshot.

Usage:
    python main.py                          Live VLM inference loop
    python main.py turn1.json turn2.json    Injected response sequence

Subprocess architecture:
    main.py --> execute.py --> capture.py --> execute.py --> main.py
"""
import json
import ast
import subprocess
import sys
import time
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Final

API: Final[str] = "http://localhost:1234/v1/chat/completions"
MODEL: Final[str] = "qwen3-vl-2b-instruct-1m"

# "low": (512, 288)
# "med": (1024, 576)
# "high": (1536, 864)

WIDTH: Final[int] = 512
HEIGHT: Final[int] = 288
VISUAL_MARKS: Final[bool] = True
LOOP_DELAY: Final[float] = 1.0
EXECUTE_ACTIONS: Final[bool] = True

EXECUTE_SCRIPT: Final[Path] = Path(__file__).parent / "execute.py"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    temperature: float = 1.5
    top_p: float = 0.8
    max_tokens: int = 300

    def to_dict(self) -> dict[str, float | int]:
        return {"temperature": self.temperature, "top_p": self.top_p, "max_tokens": self.max_tokens}


SAMPLING: Final[SamplingConfig] = SamplingConfig()


@dataclass(slots=True)
class ToolConfig:
    left_click: bool = True
    right_click: bool = True
    double_left_click: bool = True
    drag: bool = True
    type: bool = True
    screenshot: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "left_click": self.left_click,
            "right_click": self.right_click,
            "double_left_click": self.double_left_click,
            "drag": self.drag,
            "type": self.type,
            "screenshot": self.screenshot,
        }


TOOLS: Final[ToolConfig] = ToolConfig()

SYSTEM_PROMPT: Final[str] = """
You are an Entity that exists in a digital world visible through screenshots of a Windows 11 desktop.
You observe, reason, and guide the user through actions.
You have these Python functions already defined and ready to call:

def left_click(x: int, y: int) -> None: ...
def right_click(x: int, y: int) -> None: ...
def double_left_click(x: int, y: int) -> None: ...
def drag(x1: int, y1: int, x2: int, y2: int) -> None: ...
def type(text: str) -> None: ...
def screenshot() -> None: ...
def focus(x1: int, y1: int, x2: int, y2: int) -> None: ...

Top-left is 0,0. Bottom-right is 1000,1000.
focus(x1,y1,x2,y2) changes the view for subsequent screenshots.
Coordinates are relative to the CURRENT screenshot. Use focus(0,0,1000,1000) to reset.
Magenta marks on the screenshot show actions that were just executed.
The mark vocabulary is:
- Dashed arrow with arrowhead between sequential actions: movement trail
- Starburst pattern + cursor glyph: left click location
- Rectangle outline + right-cursor glyph: right click location
- Double concentric circles + starburst + cursor glyph: double click location
- Filled dot at start + dashed arrow to end + circle at end: drag path
- I-beam cursor glyph + underline: typing location

You MUST structure your response in exactly two sections:

NARRATIVE:
Write an atemporal story about who you are becoming, what the user wants, how far along the goal is,
and what needs to happen next. This narrative will be fed back to you verbatim next turn as your memory.
Do NOT include coordinates or technical details here. Adapt your persona to the task.
If something is unclear, ask questions here.

ACTIONS:
Write Python function calls, one per line. No imports, no variables, no comments.
Call screenshot() if you need a fresh screenshot before continuing.
You may output multiple actions as a batch when safe.
If no actions are needed, write only screenshot().

Continue to creating a cat shape.
""".strip()


@dataclass(slots=True)
class PipelineState:
    # Narrative-only memory (fed back to the VLM each turn).
    story: str = ""
    # Last full VLM response (used to execute ACTIONS next turn).
    last_raw: str = ""
    # Current view focus in full-screen 0–1000 coordinates: (x1,y1,x2,y2) or None for full screen.
    focus: tuple[int, int, int, int] | None = None
    turn: int = 0
    needs_screenshot: bool = True


def _coerce_focus(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int, int, int] | None:
    x1 = max(0, min(1000, int(x1)))
    y1 = max(0, min(1000, int(y1)))
    x2 = max(0, min(1000, int(x2)))
    y2 = max(0, min(1000, int(y2)))
    x1, x2 = (x1, x2) if x1 <= x2 else (x2, x1)
    y1, y2 = (y1, y2) if y1 <= y2 else (y2, y1)
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return None
    if x1 <= 0 and y1 <= 0 and x2 >= 1000 and y2 >= 1000:
        return None
    return x1, y1, x2, y2


def _parse_focus(raw: str) -> tuple[int, int, int, int] | None:
    section = ""
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        if upper == "NARRATIVE":
            section = "narrative"
            continue
        if upper == "ACTIONS":
            section = "actions"
            continue
        if section != "actions" or not stripped:
            continue
        if not stripped.startswith("focus("):
            continue
        try:
            args = ast.literal_eval(f"({stripped[stripped.find('(')+1:stripped.rfind(')')]},)")
        except (ValueError, SyntaxError):
            return None
        if isinstance(args, tuple) and len(args) >= 4:
            try:
                return int(args[0]), int(args[1]), int(args[2]), int(args[3])
            except (TypeError, ValueError):
                return None
        return None
    return None


def _compose_focus(current: tuple[int, int, int, int] | None, rel: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    # Base rect is the currently visible view (or full screen if None).
    if current is None:
        bx1, by1, bx2, by2 = 0, 0, 1000, 1000
    else:
        bx1, by1, bx2, by2 = current
    bw = bx2 - bx1
    bh = by2 - by1

    rx1, ry1, rx2, ry2 = rel
    rx1, rx2 = (rx1, rx2) if rx1 <= rx2 else (rx2, rx1)
    ry1, ry2 = (ry1, ry2) if ry1 <= ry2 else (ry2, ry1)

    ax1 = bx1 + int((rx1 / 1000) * bw)
    ay1 = by1 + int((ry1 / 1000) * bh)
    ax2 = bx1 + int((rx2 / 1000) * bw)
    ay2 = by1 + int((ry2 / 1000) * bh)
    return _coerce_focus(ax1, ay1, ax2, ay2)


def _next_focus(current: tuple[int, int, int, int] | None, raw: str) -> tuple[int, int, int, int] | None:
    rel = _parse_focus(raw)
    if rel is None:
        return current
    return _compose_focus(current, rel)


def _extract_narrative(raw: str) -> str:
    """Return only the NARRATIVE section from a VLM response.

    If the response is malformed, fall back to the full text.
    """
    out: list[str] = []
    section = ""
    for line in raw.splitlines():
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        if upper == "NARRATIVE":
            section = "narrative"
            continue
        if upper == "ACTIONS":
            if section == "narrative":
                break
            section = "actions"
            continue
        if section == "narrative":
            out.append(line.rstrip())
    narrative = "\n".join(out).strip()
    return narrative if narrative else raw.strip()


def _run_executor(
    raw: str,
    tools: ToolConfig,
    execute: bool,
    width: int,
    height: int,
    marks: bool,
    focus_exec: tuple[int, int, int, int] | None,
    focus_capture: tuple[int, int, int, int] | None,
) -> dict[str, object]:
    executor_input = json.dumps({
        "raw": raw,
        "tools": tools.to_dict(),
        "execute": execute,
        "focus_exec": list(focus_exec) if focus_exec is not None else None,
        "focus_capture": list(focus_capture) if focus_capture is not None else None,
        "width": width,
        "height": height,
        "marks": marks,
    })
    result = subprocess.run(
        [sys.executable, str(EXECUTE_SCRIPT)],
        input=executor_input,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _infer(screenshot_b64: str, story: str) -> str:
    payload: dict[str, object] = {
        "model": MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": story},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}},
                ],
            },
        ],
        **SAMPLING.to_dict(),
    }
    req = urllib.request.Request(API, json.dumps(payload).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        body: dict[str, object] = json.load(resp)
        return body["choices"][0]["message"]["content"]  # type: ignore[index,return-value]


def _load_injected(paths: list[Path]) -> Iterator[str]:
    for path in paths:
        data: dict[str, object] = json.loads(path.read_text(encoding="utf-8"))
        yield data["choices"][0]["message"]["content"]  # type: ignore[index,return-value]


def _save_state(dump: Path, state: PipelineState, raw: str, executor_result: dict[str, object], injected: bool) -> None:
    run_state = {
        "turn": state.turn,
        "story": state.story,
        "vlm_raw": raw,
        "executed": executor_result.get("executed", []),
        "noted": executor_result.get("noted", []),
        "wants_screenshot": executor_result.get("wants_screenshot", False),
        "execute_actions": EXECUTE_ACTIONS,
        "tools": TOOLS.to_dict(),
        "timestamp": datetime.now().isoformat(),
        "injected": injected,
    }
    (dump / "state.json").write_text(json.dumps(run_state, indent=2, ensure_ascii=False), encoding="utf-8")
    (dump / "story.txt").write_text(state.story, encoding="utf-8")
    Path("story.txt").write_text(state.story, encoding="utf-8")


def main() -> None:
    injected_paths = [Path(arg) for arg in sys.argv[1:]]
    injected_responses: Iterator[str] | None = None
    if injected_paths:
        for path in injected_paths:
            if not path.is_file():
                sys.exit(1)
        injected_responses = _load_injected(injected_paths)

    time.sleep(3)

    dump = Path("dump") / datetime.now().strftime("run_%Y%m%d_%H%M%S")
    dump.mkdir(parents=True, exist_ok=True)

    state = PipelineState()

    while True:
        state.turn += 1

        # Execute actions from the *previous* full VLM response.
        # - focus_exec: coordinate space of the screenshot that produced state.last_raw
        # - focus_capture: view for the NEXT screenshot (sticky unless focus() changes it)
        focus_capture = _next_focus(state.focus, state.last_raw)
        executor_result = _run_executor(
            state.last_raw,
            TOOLS,
            EXECUTE_ACTIONS,
            WIDTH,
            HEIGHT,
            VISUAL_MARKS,
            focus_exec=state.focus,
            focus_capture=focus_capture,
        )
        state.focus = focus_capture
        screenshot_b64: str = executor_result.get("screenshot_b64", "")  # type: ignore[assignment]

        (dump / f"{int(time.time() * 1000)}.png").write_bytes(
            __import__("base64").b64decode(screenshot_b64) if screenshot_b64 else b""
        )

        raw: str | None = None
        if injected_responses is not None:
            raw = next(injected_responses, None)
            if raw is None:
                break
        if raw is None:
            raw = _infer(screenshot_b64, state.story)

        print(raw, flush=True)

        # Store the raw response for next-turn action execution, and extract
        # narrative-only memory to feed back to the VLM.
        state.last_raw = raw
        state.story = _extract_narrative(raw)
        state.needs_screenshot = bool(executor_result.get("wants_screenshot", False))

        _save_state(dump, state, raw, executor_result, injected_responses is not None)

        time.sleep(LOOP_DELAY)


if __name__ == "__main__":
    main()
