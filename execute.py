"""
FRANZ execute -- Parse VLM actions, dispatch Win32 input, call capture.py

Standalone subprocess. Reads JSON from stdin, writes JSON to stdout.

Input (stdin JSON):
    raw         str             Raw VLM response text
    tools       dict[str,bool]  Per-tool active states
    execute     bool            Master execution switch
    focus_exec  list[int]       Optional 4-tuple [x1,y1,x2,y2] (0-1000) describing the
                                coordinate space of the screenshot that produced *raw*
    focus_capture list[int]     Optional 4-tuple [x1,y1,x2,y2] (0-1000) describing the
                                region capture.py should crop for the next screenshot
    width       int             Screenshot width for capture.py
    height      int             Screenshot height for capture.py
    marks       bool            Whether capture.py should draw annotations

Output (stdout JSON):
    executed            list[str]   Action lines that were dispatched
    noted               list[str]   Action lines recorded but not executed
    wants_screenshot    bool        Whether VLM called screenshot()
    screenshot_b64      str         Base64 PNG from capture.py
"""
import ctypes
import ctypes.wintypes
import ast
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Final

_LDOWN: Final[int] = 0x0002
_LUP: Final[int] = 0x0004
_RDOWN: Final[int] = 0x0008
_RUP: Final[int] = 0x0010
_VK_RETURN: Final[int] = 0x0D
_VK_SHIFT: Final[int] = 0x10
_VK_SPACE: Final[int] = 0x20
_KEYUP: Final[int] = 2
_MOVE_STEPS: Final[int] = 20
_STEP_DELAY: Final[float] = 0.01
_CLICK_DELAY: Final[float] = 0.15
_CHAR_DELAY: Final[float] = 0.08
_WORD_DELAY: Final[float] = 0.15

_shcore: Final[ctypes.WinDLL] = ctypes.WinDLL("shcore", use_last_error=True)
_shcore.SetProcessDpiAwareness(2)
_user32: Final[ctypes.WinDLL] = ctypes.WinDLL("user32", use_last_error=True)
_screen_w: Final[int] = _user32.GetSystemMetrics(0)
_screen_h: Final[int] = _user32.GetSystemMetrics(1)

KNOWN_FUNCTIONS: Final[frozenset[str]] = frozenset({
    "left_click", "right_click", "double_left_click", "drag", "type", "screenshot", "focus",
})

CAPTURE_SCRIPT: Final[Path] = Path(__file__).parent / "capture.py"

def _truthy(val: str | None) -> bool:
    if not val:
        return False
    return val.strip().lower() in {"1", "true", "yes", "y", "on"}


# Default to **fake** execution (no real input). Override with:
#   FRANZ_PHYSICAL_EXECUTION=1
PHYSICAL_EXECUTION: bool = _truthy(os.environ.get("FRANZ_PHYSICAL_EXECUTION"))

def _noop(*_args: object, **_kwargs: object) -> None:
    return None


def _to_px(val: int, dim: int) -> int:
    return int(val / 1000 * dim)


def _parse_call(line: str) -> tuple[str, list[object]] | None:
    """Parse a ``name(a, b, ...)`` line without eval."""
    paren = line.find("(")
    if paren == -1:
        return None
    name = line[:paren].strip()
    try:
        args = list(ast.literal_eval(f"({line[paren + 1 : line.rfind(')')]},)"))
    except (ValueError, SyntaxError):
        return None
    return name, args


def _remap_norm(v: int, start_px: int, extent_px: int, screen_dim: int) -> int:
    v = max(0, min(1000, int(v)))
    abs_px = start_px + int((v / 1000) * extent_px)
    return max(0, min(1000, int((abs_px / screen_dim) * 1000)))


def _focus_to_px(focus: tuple[int, int, int, int]) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = focus
    fx1 = _to_px(x1, _screen_w)
    fy1 = _to_px(y1, _screen_h)
    fx2 = _to_px(x2, _screen_w)
    fy2 = _to_px(y2, _screen_h)
    fx1, fx2 = max(0, min(fx1, fx2)), min(_screen_w, max(fx1, fx2))
    fy1, fy2 = max(0, min(fy1, fy2)), min(_screen_h, max(fy1, fy2))
    fw = fx2 - fx1
    fh = fy2 - fy1
    if fw <= 0 or fh <= 0:
        return None
    return fx1, fy1, fw, fh


def _coerce_focus(val: object) -> tuple[int, int, int, int] | None:
    if not isinstance(val, (list, tuple)) or len(val) < 4:
        return None
    try:
        x1, y1, x2, y2 = (int(val[0]), int(val[1]), int(val[2]), int(val[3]))
    except Exception:
        return None
    x1 = max(0, min(1000, x1))
    y1 = max(0, min(1000, y1))
    x2 = max(0, min(1000, x2))
    y2 = max(0, min(1000, y2))
    # Treat full-screen as "no focus".
    if x1 <= 0 and y1 <= 0 and x2 >= 1000 and y2 >= 1000:
        return None
    return x1, y1, x2, y2


def _cursor_pos() -> tuple[int, int]:
    pt = ctypes.wintypes.POINT()
    _user32.GetCursorPos(ctypes.byref(pt))
    return pt.x, pt.y


def _smooth_move(tx: int, ty: int) -> None:
    sx, sy = _cursor_pos()
    dx, dy = tx - sx, ty - sy
    for i in range(_MOVE_STEPS + 1):
        t = i / _MOVE_STEPS
        t = t * t * (3.0 - 2.0 * t)
        _user32.SetCursorPos(int(sx + dx * t), int(sy + dy * t))
        time.sleep(_STEP_DELAY)


def _mouse_click(down: int, up: int) -> None:
    _user32.mouse_event(down, 0, 0, 0, 0)
    time.sleep(0.05)
    _user32.mouse_event(up, 0, 0, 0, 0)


def _key_tap(vk: int) -> None:
    _user32.keybd_event(vk, 0, 0, 0)
    time.sleep(0.02)
    _user32.keybd_event(vk, 0, _KEYUP, 0)


def _type_text(text: str) -> None:
    for ch in text:
        if ch == " ":
            _key_tap(_VK_SPACE)
            time.sleep(_WORD_DELAY)
        elif ch == "\n":
            _key_tap(_VK_RETURN)
            time.sleep(_WORD_DELAY)
        else:
            vk = _user32.VkKeyScanW(ord(ch))
            if vk == -1:
                continue
            need_shift = bool((vk >> 8) & 1)
            if need_shift:
                _user32.keybd_event(_VK_SHIFT, 0, 0, 0)
                time.sleep(0.01)
            _key_tap(vk & 0xFF)
            if need_shift:
                _user32.keybd_event(_VK_SHIFT, 0, _KEYUP, 0)
            time.sleep(_CHAR_DELAY)


def left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(_CLICK_DELAY)
    _mouse_click(_LDOWN, _LUP)


def right_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(_CLICK_DELAY)
    _mouse_click(_RDOWN, _RUP)


def double_left_click(x: int, y: int) -> None:
    _smooth_move(_to_px(x, _screen_w), _to_px(y, _screen_h))
    time.sleep(_CLICK_DELAY)
    _mouse_click(_LDOWN, _LUP)
    time.sleep(0.08)
    _mouse_click(_LDOWN, _LUP)


def drag(x1: int, y1: int, x2: int, y2: int) -> None:
    _smooth_move(_to_px(x1, _screen_w), _to_px(y1, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(_LDOWN, 0, 0, 0, 0)
    time.sleep(0.1)
    _smooth_move(_to_px(x2, _screen_w), _to_px(y2, _screen_h))
    time.sleep(0.1)
    _user32.mouse_event(_LUP, 0, 0, 0, 0)


def screenshot() -> None:
    pass


DISPATCH: Final[dict[str, Callable[..., None]]] = {
    "left_click": left_click,
    "right_click": right_click,
    "double_left_click": double_left_click,
    "drag": drag,
    "type": _type_text,
    "screenshot": screenshot,
}

DISPATCH_FAKE: Final[dict[str, Callable[..., None]]] = {
    "left_click": _noop,
    "right_click": _noop,
    "double_left_click": _noop,
    "drag": _noop,
    "type": _noop,
    "screenshot": _noop,
}


def _parse_actions(raw: str) -> list[str]:
    action_lines: list[str] = []
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
        if section == "actions" and stripped:
            action_lines.append(stripped)
    return action_lines


def _run_capture(actions: list[str], width: int, height: int, marks: bool) -> str:
    capture_input = json.dumps({
        "actions": actions,
        "width": width,
        "height": height,
        "marks": marks,
    })
    result = subprocess.run(
        [sys.executable, str(CAPTURE_SCRIPT)],
        input=capture_input,
        capture_output=True,
        text=True,
    )
    return result.stdout


def main() -> None:
    request = json.loads(sys.stdin.read())
    raw: str = request.get("raw", "")
    tools: dict[str, bool] = request.get("tools", {})
    master_execute: bool = request.get("execute", True)
    # Optional per-call override (lets you keep the process alive and flip modes).
    physical_execute: bool = bool(request.get("physical_execution", PHYSICAL_EXECUTION))
    focus_exec = _coerce_focus(request.get("focus_exec"))
    focus_capture = _coerce_focus(request.get("focus_capture"))
    width: int = request["width"]
    height: int = request["height"]
    marks: bool = request.get("marks", True)

    action_lines = _parse_actions(raw)
    dispatch = DISPATCH if physical_execute else DISPATCH_FAKE

    executed: list[str] = []
    noted: list[str] = []
    wants_screenshot = False

    focus_px = _focus_to_px(focus_exec) if focus_exec is not None else None

    for line in action_lines:
        paren = line.find("(")
        if paren == -1:
            continue
        name = line[:paren].strip()
        if name not in KNOWN_FUNCTIONS:
            continue
        if name == "screenshot":
            wants_screenshot = True
            noted.append(line)
            continue
        # focus(x1,y1,x2,y2) is a capture-only control action consumed by capture.py.
        # It must flow through to capture.py but should never produce real input.
        if name == "focus":
            noted.append(line)
            continue
        if not master_execute or not tools.get(name, True):
            noted.append(line)
            continue
        try:
            line_exec = line
            if physical_execute and focus_px is not None and name in {"left_click", "right_click", "double_left_click", "drag"}:
                parsed = _parse_call(line)
                if parsed is not None:
                    _, args = parsed
                    fx, fy, fw, fh = focus_px
                    match name:
                        case "left_click" | "right_click" | "double_left_click" if len(args) >= 2:
                            x = _remap_norm(int(args[0]), fx, fw, _screen_w)
                            y = _remap_norm(int(args[1]), fy, fh, _screen_h)
                            line_exec = f"{name}({x},{y})"
                        case "drag" if len(args) >= 4:
                            x1 = _remap_norm(int(args[0]), fx, fw, _screen_w)
                            y1 = _remap_norm(int(args[1]), fy, fh, _screen_h)
                            x2 = _remap_norm(int(args[2]), fx, fw, _screen_w)
                            y2 = _remap_norm(int(args[3]), fy, fh, _screen_h)
                            line_exec = f"drag({x1},{y1},{x2},{y2})"
            eval(line_exec, {"__builtins__": {}}, dispatch)
            executed.append(line)
        except Exception:
            noted.append(line)

    all_actions = executed + noted
    # capture.py expects focus() to be in full-screen virtual coordinates.
    # We always drive cropping via *focus_capture* to keep focus relative-to-current-view consistent.
    capture_actions = [a for a in all_actions if not a.lstrip().startswith("focus(")]
    if focus_capture is not None:
        x1, y1, x2, y2 = focus_capture
        capture_actions.insert(0, f"focus({x1},{y1},{x2},{y2})")
    screenshot_b64 = _run_capture(capture_actions, width, height, marks)

    output = json.dumps({
        "executed": executed,
        "noted": noted,
        "wants_screenshot": wants_screenshot,
        "screenshot_b64": screenshot_b64,
    })
    sys.stdout.write(output)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
