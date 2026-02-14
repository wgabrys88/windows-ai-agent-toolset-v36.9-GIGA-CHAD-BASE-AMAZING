"""
FRANZ MITM Debug Panel — single-file Python server.
Generates the debug panel HTML programmatically. No pip installs needed.
Run: python panel.py
Then open http://localhost:1234 and point main.py at the same port.
LM Studio should be on port 1235.
"""

import json, queue, threading, time, webbrowser, subprocess, sys, re, html as html_mod
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError
from dataclasses import dataclass, field
from typing import Any

# ─── Config ───────────────────────────────────────────────────────────
HOST, PORT = "0.0.0.0", 1234
UP_HOST, UP_PORT, UP_TIMEOUT = "127.0.0.1", 1235, 300
MAIN_PY = Path(__file__).with_name("main.py")
UPSTREAM = f"http://{UP_HOST}:{UP_PORT}/v1/chat/completions"

# ─── Shared State ─────────────────────────────────────────────────────
edited_request: queue.Queue[str] = queue.Queue(maxsize=1)
edited_response: queue.Queue[str] = queue.Queue(maxsize=1)
sse_clients: list[queue.Queue[str]] = []
sse_lock = threading.Lock()
turn_counter = 0


# ═══════════════════════════════════════════════════════════════════════
#  HTML BUILDER — constructs the panel from Python structures
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class El:
    """Represents an HTML element."""
    tag: str
    attrs: dict = field(default_factory=dict)
    children: list = field(default_factory=list)
    text: str = ""
    self_closing: bool = False

    def render(self, indent=0) -> str:
        pad = "  " * indent
        attr_str = ""
        for k, v in self.attrs.items():
            if v is True:
                attr_str += f" {k}"
            elif v is not False and v is not None:
                attr_str += f' {k}="{html_mod.escape(str(v), quote=True)}"'
        if self.self_closing:
            return f"{pad}<{self.tag}{attr_str}>"
        inner = html_mod.escape(self.text) if self.text else ""
        if self.children:
            child_html = "\n".join(c.render(indent + 1) if isinstance(c, El) else f"{'  ' * (indent + 1)}{c}" for c in self.children)
            inner = f"\n{child_html}\n{pad}"
        return f"{pad}<{self.tag}{attr_str}>{inner}</{self.tag}>"


def el(tag, attrs=None, children=None, text="", self_closing=False):
    return El(tag, attrs or {}, children or [], text, self_closing)


def text_node(t):
    """Raw HTML string (no escaping)."""
    return t


# ─── Style definitions as Python dict ─────────────────────────────────
COLORS = {
    # Dark, neutral theme (higher contrast than the old blue-on-blue).
    "bg": "#050607", "panel": "#0b0d12", "border": "#1b2330",
    "text": "#d6dde8", "text_dim": "#7a8398", "text_mid": "#a6b0c6",
    "accent": "#7aa2ff", "accent_light": "#9ab6ff", "accent_dark": "#476bff",
    "input_bg": "#070a10", "input_border": "#1b2330",
    "warn": "#5a4a18", "warn_text": "#f0dea6",
    "status_idle_bg": "#0a0d12", "status_idle_fg": "#7a8398",
    "status_req_bg": "#0b1020", "status_req_fg": "#9ab6ff",
    "status_warn_bg": "#1a1406", "status_warn_fg": "#f0b35a",
    "status_fwd_bg": "#0b1020", "status_fwd_fg": "#8fb0ff",
    "status_resp_bg": "#120b22", "status_resp_fg": "#c2a7ff",
}

def build_css() -> str:
    c = COLORS
    rules = {
        "*,*::before,*::after": "box-sizing:border-box;margin:0;padding:0",
        "html,body": f"height:100%;background:{c['bg']};color:{c['text']};font-family:Consolas,'Courier New',monospace;font-size:13px;overflow:hidden;color-scheme:dark",
        "::-webkit-scrollbar": "width:5px;height:5px",
        "::-webkit-scrollbar-track": "background:#050505",
        "::-webkit-scrollbar-thumb": "background:#182038;border-radius:3px",
        "::selection": "background:#183060;color:#fff",
        "#L": f"display:flex;height:100vh;padding:6px;gap:0;background:{c['bg']}",
        "#P": f"width:50%;display:flex;flex-direction:column;gap:6px;padding:14px;overflow-y:auto;overflow-x:hidden;min-width:260px;background:{c['panel']};border:1px solid {c['border']};border-radius:8px",
        "#R": f"flex:1;display:flex;flex-direction:column;padding:14px;gap:6px;min-width:200px;background:{c['panel']};border:1px solid {c['border']};border-radius:8px",
        "#D": "width:7px;cursor:col-resize;flex-shrink:0;display:flex;align-items:center;justify-content:center",
        "#D::before": f"content:'';width:2px;height:40px;background:{c['border']};border-radius:2px;transition:.15s",
        "#D:hover::before,#D.on::before": f"background:{c['accent_dark']};box-shadow:0 0 6px #103880",
        "h1": f"font-size:15px;color:{c['accent']};text-align:center;padding:8px 0;letter-spacing:1.5px",
        ".l": f"color:{c['text_dim']};font-size:10px;text-transform:uppercase;letter-spacing:.8px;padding:3px 0;user-select:none",
        "hr": f"border:none;border-top:1px solid #0a0e14;margin:4px 0",
        "#S": "padding:8px 12px;border-radius:6px;font-size:11px;display:flex;align-items:center;gap:8px;font-weight:600;flex-shrink:0",
        ".si": f"background:{c['status_idle_bg']};color:{c['status_idle_fg']};border:1px solid #0a0e14",
        ".sr": f"background:{c['status_req_bg']};color:{c['status_req_fg']};border:1px solid #0c1830",
        ".sw": f"background:{c['status_warn_bg']};color:{c['status_warn_fg']};border:1px solid #201808",
        ".sf": f"background:{c['status_fwd_bg']};color:{c['status_fwd_fg']};border:1px solid #0c1830",
        ".sp": f"background:{c['status_resp_bg']};color:{c['status_resp_fg']};border:1px solid #180c40",
        ".dot": "width:7px;height:7px;border-radius:50%;flex-shrink:0;animation:p 2s infinite",
        ".si .dot": "background:#101828",
        ".sr .dot": f"background:{c['status_req_fg']}",
        ".sw .dot": f"background:{c['status_warn_fg']}",
        ".sf .dot": f"background:{c['status_fwd_fg']}",
        ".sp .dot": f"background:{c['status_resp_fg']}",
        "@keyframes p": "0%,100%{opacity:1}50%{opacity:.3}",
        "textarea": f"width:100%;background:{c['input_bg']};color:{c['text']};border:1px solid {c['input_border']};padding:10px;font:inherit;font-size:11px;line-height:1.55;resize:vertical;border-radius:6px;white-space:pre;tab-size:2;min-height:50px;transition:border-color .15s",
        "textarea:focus": f"outline:none;border-color:{c['accent_dark']}",
        "textarea::placeholder": "color:#4b5568",
        "button": f"padding:7px 16px;background:#141c2a;color:{c['text']};border:1px solid {c['border']};font:inherit;font-size:11px;font-weight:700;cursor:pointer;border-radius:5px;transition:background .12s,border-color .12s;letter-spacing:.3px",
        "button:disabled": "background:#0b0f16;color:#3a4256;border-color:#151c28;cursor:not-allowed",
        "button:hover:not(:disabled)": f"background:#1b2536;border-color:{c['accent_dark']}",
        "button:active:not(:disabled)": "background:#101828",
        ".b": f"background:#080c18;color:{c['text_mid']};font-size:10px;padding:5px 10px;border:1px solid {c['border']}",
        ".b:hover:not(:disabled)": f"background:#0c1220;color:{c['text']};border-color:{c['accent_dark']}",
        ".b:disabled": f"background:{c['panel']};color:#0a0e14;border-color:#080c10",
        ".bw": f"background:{c['warn']};color:{c['warn_text']}",
        ".bw:hover:not(:disabled)": "background:#685010",
        ".sect": "flex-shrink:0",
        ".sect.hidden": "display:none",
        "#HP": f"display:none;gap:5px;padding:8px;background:{c['input_bg']};border:1px solid #0a0e14;border-radius:6px;flex-wrap:wrap;align-items:center;flex-shrink:0",
        "#HP.on": "display:flex",
        ".pg": f"display:flex;align-items:center;gap:4px;background:{c['input_bg']};border:1px solid {c['border']};border-radius:4px;padding:4px 8px",
        ".pg label": f"font-size:9px;color:{c['text_dim']};text-transform:uppercase;letter-spacing:.5px;white-space:nowrap",
        ".pi": f"width:58px;padding:3px 5px;background:{c['input_bg']};color:{c['text_mid']};border:1px solid {c['input_border']};border-radius:3px;font:inherit;font-size:10px;text-align:center",
        ".pi:focus": f"outline:none;border-color:{c['accent_dark']}",
        ".pw": "width:120px;text-align:left",
        ".r": "display:flex;gap:5px;align-items:center;flex-wrap:wrap",
        ".inf": "display:flex;gap:12px;align-items:center;font-size:11px;padding:2px 0;flex-shrink:0",
        "#turn": f"color:{c['accent_light']};font-weight:700",
        "#mdl": f"color:{c['text_dim']};font-size:10px",
        "#phase": f"color:{c['status_fwd_fg']};font-size:9px;font-weight:700;padding:2px 8px;border-radius:3px;background:{c['input_bg']};border:1px solid {c['border']}",
        "#phase:empty": "display:none",
        "#story": f"background:{c['input_bg']};border:1px solid #0a0e14;border-radius:6px;padding:10px;white-space:pre-wrap;word-break:break-word;max-height:100px;overflow-y:auto;font-size:11px;color:{c['text_mid']};line-height:1.45",
        "#story:empty::after": 'content:"Waiting for main.py...";color:#141c28;font-style:italic',
        "#ac": "background:#060410;border:1px solid #140c30;border-radius:6px;padding:10px;white-space:pre-wrap;word-break:break-word;max-height:110px;overflow-y:auto;font-size:11px;color:#c2a7ff;line-height:1.45",
        ".tg": f"display:flex;align-items:center;gap:7px;padding:5px 10px;border-radius:5px;background:{c['input_bg']};border:1px solid {c['border']};flex-shrink:0",
        ".tg label": f"font-size:10px;color:{c['text_dim']};cursor:pointer",
        ".sw-wrap": "position:relative;width:36px;height:19px;cursor:pointer",
        ".sw-wrap input": "opacity:0;width:0;height:0",
        ".sw-sl": "position:absolute;inset:0;background:#0a0e14;border-radius:19px;transition:.2s",
        ".sw-sl::before": "content:'';position:absolute;width:13px;height:13px;left:3px;bottom:3px;background:#1a2030;border-radius:50%;transition:.2s",
        ".sw-wrap input:checked+.sw-sl": "background:#0c1830",
        ".sw-wrap input:checked+.sw-sl::before": f"transform:translateX(17px);background:{c['accent_light']}",
        ".di": f"width:48px;padding:3px 5px;background:{c['input_bg']};color:{c['text_mid']};border:1px solid {c['input_border']};border-radius:3px;font:inherit;font-size:10px;text-align:center",
        ".di:focus": f"outline:none;border-color:{c['accent_dark']}",
        "#cw": "flex:1;display:flex;flex-direction:column;gap:5px;min-height:0",
        "#ct": "display:flex;gap:5px;align-items:center;flex-wrap:wrap;flex-shrink:0",
        "#ct label": "color:#303848;font-size:9px",
        "#ct input[type=range]": f"width:75px;accent-color:{c['accent_dark']};cursor:pointer",
        "#ct input[type=color]": f"width:26px;height:22px;padding:0;border:1px solid {c['input_border']};background:{c['input_bg']};cursor:pointer;border-radius:3px",
        "#cc": "flex:1;position:relative;overflow:hidden;background:#010204;border:1px solid #0a0e14;border-radius:8px;min-height:180px;display:flex;align-items:center;justify-content:center",
        "#cv": "display:block;cursor:crosshair;max-width:100%;max-height:100%",
        ".ce": "position:absolute;color:#141c28;font-size:11px;pointer-events:none;font-style:italic",
        ".sep": "width:1px;height:16px;background:#0a0e14;display:inline-block",
        ".zr": "display:flex;gap:5px;align-items:center;flex-shrink:0",
        "#zd": f"font-size:10px;color:{c['text_dim']};min-width:40px;text-align:center;padding:3px 6px;background:{c['input_bg']};border-radius:3px;border:1px solid #0a0e14",
        ".ta": f"border-color:{c['accent_dark']}!important;background:#040c20!important;color:{c['accent_light']}!important",
        "@media(max-width:900px)": "#L{flex-direction:column;overflow-y:auto}#P{width:100%!important;max-height:50vh}#D{display:none}#R{min-height:40vh}",
    }
    lines = []
    for selector, props in rules.items():
        if selector.startswith("@"):
            lines.append(f"{selector}{{{props}}}")
        else:
            lines.append(f"{selector}{{{props}}}")
    return "\n".join(lines)


# ─── Insert buttons definition ────────────────────────────────────────
INSERT_BUTTONS = [
    ("screenshot()", "screenshot()"),
    ("focus(100,100,900,900)", "focus"),
    ("focus(0,0,1000,1000)", "reset_focus"),
    ("left_click(500,500)", "left_click"),
    ("right_click(500,500)", "right_click"),
    ("double_left_click(500,500)", "dbl_click"),
    ('type("hello")', "type"),
    ("drag(100,100,500,500)", "drag"),
]

MANUAL_BUTTONS = [
    ("NARRATIVE:\\nI observe the screen.\\n\\nACTIONS:\\nscreenshot()", "tpl"),
    ("focus(100,100,900,900)", "focus"),
    ("left_click(500,500)", "left_click"),
    ('type("hello")', "type"),
]

RESP_BUTTONS = [
    ("screenshot()", "screenshot()"),
    ("focus(100,100,900,900)", "focus"),
    ("left_click(500,500)", "left_click"),
    ('type("hello")', "type"),
]

PARAMS = [
    # label, element_id, input_type, default, extra_css, extra_attrs
    ("model", "p-model", "text", "", "pw", {}),
    ("temp", "p-temperature", "number", "1.5", "", {"step": "0.05", "min": "0", "max": "2"}),
    ("top_p", "p-top_p", "number", "0.8", "", {"step": "0.05", "min": "0", "max": "1"}),
    ("max_tok", "p-max_tokens", "number", "300", "", {"step": "1", "min": "1", "max": "32768"}),
    ("freq", "p-frequency_penalty", "number", "0", "", {"step": "0.1", "min": "-2", "max": "2"}),
    ("pres", "p-presence_penalty", "number", "0", "", {"step": "0.1", "min": "-2", "max": "2"}),
]


def build_insert_row(buttons, data_attr):
    return el("div", {"class": "r"}, [
        el("button", {"class": "b", f"data-{data_attr}": val}, text=label)
        for val, label in buttons
    ])


def build_param_groups():
    groups = []
    for label, id_, type_, default, extra_cls, extra_attrs in PARAMS:
        cls = f"pi {extra_cls}".strip()
        attrs = {"class": cls, "id": id_, "type": type_, "value": default}
        if type_ == "number":
            attrs.update(extra_attrs)
        if type_ == "text":
            attrs["spellcheck"] = "false"
        groups.append(el("div", {"class": "pg"}, [
            el("label", text=label),
            el("input", attrs, self_closing=True),
        ]))
    # stream toggle
    groups.append(el("div", {"class": "pg"}, [
        el("label", text="stream"),
        el("label", {"class": "sw-wrap", "style": "width:32px;height:17px"}, [
            # main.py does not support streamed responses; keep this off.
            el("input", {"type": "checkbox", "id": "p-stream", "disabled": True}, self_closing=True),
            el("span", {"class": "sw-sl"}),
        ]),
    ]))
    return groups


def build_body() -> El:
    """Build the entire page body as an element tree."""

    # Header row
    header = el("div", {"style": "display:flex;align-items:center;justify-content:space-between;gap:8px;flex-wrap:wrap"}, [
        el("h1", text="FRANZ -- MITM Debug"),
        el("div", {"class": "tg"}, [
            el("label", {"class": "sw-wrap"}, [
                el("input", {"type": "checkbox", "id": "ap", "checked": True}, self_closing=True),
                el("span", {"class": "sw-sl"}),
            ]),
            el("label", {"for": "ap"}, text="Auto-Pass"),
            el("input", {"type": "number", "id": "ad", "class": "di", "value": "50", "min": "0", "max": "9999", "title": "Delay ms"}, self_closing=True),
            el("label", {"style": "font-size:9px;color:#7a8398"}, text="ms"),
        ]),
    ])

    status = el("div", {"id": "S", "class": "si"}, [
        el("span", {"class": "dot"}),
        text_node("Idle -- waiting for main.py"),
    ])

    info = el("div", {"class": "inf"}, [
        el("span", children=[text_node("Turn: "), el("span", {"id": "turn"}, text="0")]),
        el("span", {"id": "mdl"}),
        el("span", {"id": "phase"}),
    ])

    params = el("div", {"id": "HP"}, build_param_groups())

    # Request section
    req_section = el("div", {"id": "pq", "class": "sect hidden"}, [
        el("hr", self_closing=True),
        el("span", {"class": "l"}, text="Intercepted Request"),
        el("span", {"class": "l"}, text="Story"),
        el("div", {"id": "story"}),
        el("span", {"class": "l"}, text="Request Body (JSON -- image on canvas)"),
        el("textarea", {"id": "qb", "rows": "10", "spellcheck": "false"}),
        el("span", {"class": "l"}, text="Insert at cursor"),
        build_insert_row(INSERT_BUTTONS, "i"),
        el("div", {"class": "r"}, [
            el("button", {"id": "fq", "disabled": True}, text="Forward to LM Studio"),
            el("button", {"id": "sk", "class": "bw", "disabled": True}, text="Skip -- Manual"),
        ]),
        el("div", {"id": "ms", "style": "display:none"}, [
            el("span", {"class": "l"}, text="Manual Response"),
            el("textarea", {"id": "mr", "rows": "3", "spellcheck": "false", "placeholder": "Type manual response..."}),
            el("div", {"class": "r"}, [
                el("button", {"class": "b", "data-m": b[0]}, text=b[1]) for b in MANUAL_BUTTONS
            ]),
            el("div", {"class": "r"}, [
                el("button", {"id": "sm", "class": "bw"}, text="Send Manual"),
                el("button", {"id": "cm", "class": "b"}, text="Cancel"),
            ]),
        ]),
    ])

    # Response section
    resp_section = el("div", {"id": "pp", "class": "sect hidden"}, [
        el("hr", self_closing=True),
        el("span", {"class": "l"}, text="Response from LM Studio"),
        el("div", {"id": "aw", "style": "display:none"}, [
            el("span", {"class": "l"}, text="Assistant Content"),
            el("div", {"id": "ac"}),
        ]),
        el("span", {"class": "l"}, text="Assistant Content (editable)"),
        el("textarea", {"id": "rb", "rows": "10", "spellcheck": "false"}),
        el("span", {"class": "l"}, text="Insert at cursor"),
        build_insert_row(RESP_BUTTONS, "r"),
        el("div", {"class": "r"}, [
            el("button", {"id": "fr", "disabled": True}, text="Return to main.py"),
            el("button", {"id": "fo", "class": "b", "disabled": True}, text="Return Original"),
        ]),
        el("div", {"class": "r"}, [
            el("button", {"id": "tj", "class": "b"}, text="Show Raw JSON"),
        ]),
        el("div", {"id": "aj", "style": "display:none"}, [
            el("span", {"class": "l"}, text="Raw Response JSON"),
            el("textarea", {"id": "rj", "rows": "10", "spellcheck": "false", "readonly": True}),
        ]),
    ])

    # Left panel
    left = el("div", {"id": "P"}, [
        header, status, info, params, req_section, resp_section,
    ])

    # Resizer
    resizer = el("div", {"id": "D"})

    # Canvas toolbar
    toolbar = el("div", {"id": "ct"}, [
        el("button", {"class": "b ta", "id": "tb", "data-t": "brush"}, text="Brush"),
        el("button", {"class": "b", "id": "te", "data-t": "eraser"}, text="Eraser"),
        el("span", {"class": "sep"}),
        el("button", {"class": "b", "id": "clr"}, text="Clear"),
        el("button", {"class": "b", "id": "rst"}, text="Reset"),
        el("span", {"class": "sep"}),
        el("label", text="Size"),
        el("input", {"type": "range", "id": "bsz", "min": "1", "max": "40", "value": "4"}, self_closing=True),
        el("label", text="Color"),
        el("input", {"type": "color", "id": "bcl", "value": "#ff3333"}, self_closing=True),
    ])

    zoom_row = el("div", {"class": "zr"}, [
        el("button", {"class": "b", "id": "zf"}, text="Fit"),
        el("button", {"class": "b", "id": "za"}, text="1:1"),
        el("button", {"class": "b", "id": "zo"}, text="-"),
        el("span", {"id": "zd"}, text="Fit"),
        el("button", {"class": "b", "id": "zi"}, text="+"),
    ])

    right = el("div", {"id": "R"}, [
        el("span", {"class": "l"}, text="Screenshot Canvas"),
        toolbar,
        zoom_row,
        el("div", {"id": "cw"}, [
            el("div", {"id": "cc"}, [
                el("canvas", {"id": "cv"}),
                el("div", {"class": "ce", "id": "cp"}, text="No screenshot loaded"),
            ]),
        ]),
    ])

    return el("div", {"id": "L"}, [left, resizer, right])


def build_js() -> str:
    """Return the client-side JavaScript as a string."""
    return '''"use strict";
const $=id=>document.getElementById(id),$$=s=>document.querySelectorAll(s);

let AP=true,AD=50;
$("ap").onchange=function(){AP=this.checked};
$("ad").onchange=function(){AD=Math.max(0,+this.value||0)};
function autoClk(b){if(AP&&b&&!b.disabled)setTimeout(()=>{if(AP&&!b.disabled)b.click()},AD)}

const S=$("S"),qb=$("qb"),rb=$("rb"),rj=$("rj"),mr=$("mr"),
  fq=$("fq"),sk=$("sk"),sm=$("sm"),fr=$("fr"),fo=$("fo"),
  tj=$("tj"),aj=$("aj"),cv=$("cv"),cx=cv.getContext("2d"),cc=$("cc"),cpEl=$("cp"),
  pqEl=$("pq"),ppEl=$("pp");
let turn=0,origResp="",origContent="",origSS="",tool="brush",draw=false,lx=0,ly=0,
  natW=0,natH=0,zm="fit",zs=1;

const PK=["model","temperature","top_p","max_tokens","frequency_penalty","presence_penalty","stream"];
const PE={};PK.forEach(k=>PE[k]=$("p-"+k));

function extractP(o){
  PK.forEach(k=>{const e=PE[k];if(!e)return;
    if(k==="stream"){e.checked=!!o[k];return}
    if(k==="model"){e.value=o[k]||"";return}
    if(o[k]!==undefined)e.value=o[k]});
  $("HP").classList.add("on");
}
function injectP(s){
  try{const o=JSON.parse(s);
    PK.forEach(k=>{const e=PE[k];if(!e)return;
      if(k==="stream"){o[k]=false;return} // main.py expects non-streamed JSON
      if(k==="model"){if(e.value)o[k]=e.value;return}
      if(k==="max_tokens"){const v=parseInt(e.value,10);if(!isNaN(v))o[k]=v;return}
      const v=parseFloat(e.value);if(!isNaN(v))o[k]=v});
    return JSON.stringify(o)}catch{return s}
}

function st(c,m){S.className=c;S.innerHTML=`<span class="dot"></span>${m}`}
function show(el){el.classList.remove("hidden")}
function hide(el){el.classList.add("hidden")}
function fmt(s){try{return JSON.stringify(JSON.parse(s),null,2)}catch{return s}}
function ins(ta,t){const s=ta.selectionStart,v=ta.value;ta.value=v.slice(0,s)+t+v.slice(ta.selectionEnd);ta.selectionStart=ta.selectionEnd=s+t.length;ta.focus()}
function post(u,o){return fetch(u,{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify(o)}).then(r=>{if(!r.ok)throw Error("HTTP "+r.status);return r.json()})}

function extractContent(raw){
  try{const o=JSON.parse(raw);const ch=o.choices;
    if(ch&&ch.length&&ch[0].message){
      const c=ch[0].message.content;
      if(typeof c==="string")return c;
      if(Array.isArray(c))return c.map(p=>p?.text||"").join("");
    }
  }catch{}return raw;
}
function rebuildResponse(edited){
  try{const o=JSON.parse(origResp);
    if(o.choices&&o.choices.length&&o.choices[0].message)o.choices[0].message.content=edited;
    return JSON.stringify(o);
  }catch{return origResp}
}

function b64(){if(!cv.width)return origSS;const d=cv.toDataURL("image/png"),p="data:image/png;base64,";return d.startsWith(p)?d.slice(p.length):origSS}
function zoom(){const d=$("zd");
  if(zm==="fit"){cv.style.cssText="max-width:100%;max-height:100%";d.textContent="Fit";cc.style.overflow="hidden"}
  else if(zm==="1:1"){cv.style.cssText=`max-width:none;max-height:none;width:${natW}px;height:${natH}px`;d.textContent="1:1";cc.style.overflow="auto"}
  else{cv.style.cssText=`max-width:none;max-height:none;width:${Math.round(natW*zs)}px;height:${Math.round(natH*zs)}px`;d.textContent=Math.round(zs*100)+"%";cc.style.overflow="auto"}}
function loadImg(b){
  origSS=b;
  if(!b){cv.width=800;cv.height=600;natW=800;natH=600;cx.fillStyle="#010204";cx.fillRect(0,0,800,600);cpEl.style.display="block";zm="fit";zoom();return}
  cpEl.style.display="none";const img=new Image();
  img.onload=()=>{cv.width=img.naturalWidth;cv.height=img.naturalHeight;natW=img.naturalWidth;natH=img.naturalHeight;cx.drawImage(img,0,0);zm="fit";zoom()};
  img.src="data:image/png;base64,"+b}
new ResizeObserver(()=>{if(zm==="fit")zoom()}).observe(cc);

$("zf").onclick=()=>{zm="fit";zoom()};
$("za").onclick=()=>{zm="1:1";zoom()};
function curScale(){if(zm==="fit"){const r=cc.getBoundingClientRect();return Math.min(r.width/natW,r.height/natH)}if(zm==="1:1")return 1;return zs}
$("zi").onclick=()=>{zs=Math.min(curScale()*1.25,5);zm="z";zoom()};
$("zo").onclick=()=>{zs=Math.max(curScale()/1.25,.1);zm="z";zoom()};
$("rst").onclick=()=>loadImg(origSS);

(()=>{const D=$("D"),P=$("P");let drag=false,sx,sw;
  D.onmousedown=e=>{e.preventDefault();drag=true;sx=e.clientX;sw=P.getBoundingClientRect().width;D.classList.add("on");document.body.style.cssText="cursor:col-resize;user-select:none"};
  document.onmousemove=e=>{if(drag)P.style.width=Math.max(260,Math.min(sw+e.clientX-sx,innerWidth*.7))+"px"};
  document.onmouseup=()=>{if(drag){drag=false;D.classList.remove("on");document.body.style.cssText=""}}})();

fq.onclick=()=>{fq.disabled=sk.disabled=true;st("sf","Forwarding...");
  post("/forward_request",{raw_body_stripped:injectP(qb.value),canvas_b64:b64()}).catch(e=>{st("sw","Failed: "+e.message);fq.disabled=sk.disabled=false})};
sk.onclick=()=>{$("ms").style.display="flex";mr.focus()};
$("cm").onclick=()=>{$("ms").style.display="none"};
sm.onclick=()=>{const c=mr.value.trim();if(!c)return;fq.disabled=sk.disabled=sm.disabled=true;st("sf","Sending manual...");
  post("/skip_upstream",{content:c}).then(()=>{$("ms").style.display="none";st("si","Manual sent.")}).catch(()=>{sm.disabled=false})};
fr.onclick=()=>{fr.disabled=fo.disabled=true;st("si","Returning...");
  const fullBody=rebuildResponse(rb.value);
  post("/forward_response",{raw_body:fullBody}).then(()=>{st("si","Done.")}).catch(()=>{fr.disabled=fo.disabled=false})};
fo.onclick=()=>{rb.value=origContent;fr.click()};

function setJsonVisible(on){
  if(!aj||!tj)return;
  aj.style.display=on?"":"none";
  tj.textContent=on?"Hide Raw JSON":"Show Raw JSON";
}
if(tj) tj.onclick=()=>setJsonVisible(aj.style.display==="none");
setJsonVisible(false);

$$("[data-i]").forEach(b=>b.onclick=()=>ins(qb,b.dataset.i));
$$("[data-m]").forEach(b=>b.onclick=()=>ins(mr,b.dataset.m));
$$("[data-r]").forEach(b=>b.onclick=()=>ins(rb,b.dataset.r));
[qb,rb,mr].forEach((ta,i)=>ta.onkeydown=e=>{if(e.ctrlKey&&e.key==="Enter"){e.preventDefault();[fq,fr,sm][i].click()}});

const evs=new EventSource("/events");
evs.addEventListener("incoming_request",e=>{
  const d=JSON.parse(e.data);turn=d.turn||turn+1;
  $("turn").textContent=turn;$("mdl").textContent=d.model||"";$("phase").textContent="REQUEST";
  $("story").textContent=d.story||"(empty)";
  loadImg(d.screenshot_b64||"");
  qb.value=fmt(d.raw_body_stripped||"{}");
  try{extractP(JSON.parse(d.raw_body_stripped||"{}"))}catch{}
  fq.disabled=sk.disabled=sm.disabled=false;$("ms").style.display="none";mr.value="";$("aw").style.display="none";
  show(pqEl);hide(ppEl);st("sr","Request intercepted");autoClk(fq)});
evs.addEventListener("forwarding",()=>{$("phase").textContent="FORWARDING";st("sf","Waiting for LM Studio...")});
evs.addEventListener("incoming_response",e=>{
  const d=JSON.parse(e.data);$("phase").textContent="RESPONSE";
  origResp=d.raw_body||"{}";
  origContent=extractContent(origResp);
  rb.value=origContent;
  if(rj) rj.value=fmt(origResp);
  setJsonVisible(false);
  if(d.assistant_content){$("ac").textContent=d.assistant_content;$("aw").style.display=""}else $("aw").style.display="none";
  fr.disabled=fo.disabled=false;hide(pqEl);show(ppEl);st("sp","Response (HTTP "+d.status+")");autoClk(fr)});
evs.addEventListener("turn_complete",e=>{
  const d=JSON.parse(e.data);$("phase").textContent="";
  st("si","T"+d.turn+" done ("+d.mode+")")});
evs.addEventListener("ping",()=>{});
evs.onerror=()=>st("sw","SSE disconnected...");

function setT(t){tool=t;$("tb").classList.toggle("ta",t==="brush");$("te").classList.toggle("ta",t==="eraser")}
$$("[data-t]").forEach(b=>b.onclick=()=>setT(b.dataset.t));
$("clr").onclick=()=>{cx.fillStyle="#010204";cx.fillRect(0,0,cv.width,cv.height);cpEl.style.display="block"};
function cp2(e){const r=cv.getBoundingClientRect(),sx=cv.width/r.width,sy=cv.height/r.height;
  const t=e.touches?.[0];return{x:((t||e).clientX-r.left)*sx,y:((t||e).clientY-r.top)*sy}}
function beg(e){draw=true;const p=cp2(e);lx=p.x;ly=p.y;cpEl.style.display="none"}
function mov(e){if(!draw)return;const p=cp2(e);cx.beginPath();cx.moveTo(lx,ly);cx.lineTo(p.x,p.y);
  cx.lineWidth=+$("bsz").value;cx.lineCap=cx.lineJoin="round";cx.strokeStyle=tool==="eraser"?"#010204":$("bcl").value;cx.stroke();lx=p.x;ly=p.y}
function end(){draw=false}
cv.onmousedown=beg;cv.onmousemove=mov;cv.onmouseup=cv.onmouseleave=end;
cv.ontouchstart=e=>{e.preventDefault();beg(e)};cv.ontouchmove=e=>{e.preventDefault();mov(e)};cv.ontouchend=cv.ontouchcancel=end;

loadImg("");'''


def build_html() -> str:
    """Assemble complete HTML document."""
    css = build_css()
    body = build_body()
    js = build_js()

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="color-scheme" content="dark">
<title>FRANZ MITM Debug Panel</title>
<style>
{css}
</style>
</head>
<body>
{body.render(0)}
<script>
{js}
</script>
</body>
</html>"""


# Cache the generated HTML at module load
_HTML_BYTES = build_html().encode("utf-8")


# ═══════════════════════════════════════════════════════════════════════
#  PROXY LOGIC (unchanged from your working version)
# ═══════════════════════════════════════════════════════════════════════

B64_PNG_PREFIX = "data:image/png;base64,"
B64_JPG_PREFIX = "data:image/jpeg;base64,"


def broadcast(event: str, data: dict) -> None:
    p = f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"
    with sse_lock:
        dead = [q for q in sse_clients if _try_put(q, p)]
        for q in dead:
            sse_clients.remove(q)


def _try_put(q, p):
    try:
        q.put_nowait(p); return False
    except queue.Full:
        return True


def drain(q):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def extract_display(body):
    msgs, model, ss, story = body.get("messages", []), body.get("model", ""), "", ""
    for m in msgs:
        if m.get("role") != "user":
            continue
        c = m.get("content", "")
        if isinstance(c, list):
            for p in c:
                if not isinstance(p, dict):
                    continue
                if p.get("type") == "text":
                    story = p.get("text", "")
                elif p.get("type") == "image_url":
                    u = p.get("image_url", {}).get("url", "")
                    for pfx in (B64_PNG_PREFIX, B64_JPG_PREFIX):
                        if u.startswith(pfx):
                            ss = u[len(pfx):]; break
        elif isinstance(c, str):
            story = c
    return {"model": model, "story": story, "screenshot_b64": ss}


def strip_b64_from_body(raw_str):
    return re.sub(r'data:image/[a-z]+;base64,[A-Za-z0-9+/=]+', '<IMAGE_ON_CANVAS>', raw_str)


def inject_b64_into_body(body_str, b64, original_raw):
    m = re.search(r'data:image/(png|jpeg);base64,', original_raw)
    prefix = "data:image/png;base64," if not m else m.group(0)
    return body_str.replace("<IMAGE_ON_CANVAS>", prefix + b64)


def forward_upstream(body_str):
    raw = body_str.encode("utf-8")
    req = Request(UPSTREAM, data=raw,
                  headers={"Content-Type": "application/json", "Content-Length": str(len(raw))},
                  method="POST")
    try:
        with urlopen(req, timeout=UP_TIMEOUT) as r:
            return r.status, r.read().decode("utf-8")
    except HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace")
    except URLError as e:
        return 502, json.dumps({"error": f"upstream failed: {e.reason}"})
    except Exception as e:
        return 502, json.dumps({"error": str(e)})


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        m = fmt % a
        if "/v1/" in m or "error" in m.lower():
            print(f"  [HTTP] {m}")

    def do_GET(self):
        p = urlparse(self.path).path
        if p == "/events":
            return self._sse()
        self._html()

    def do_POST(self):
        p = urlparse(self.path).path
        routes = {
            "/v1/chat/completions": self._completions,
            "/forward_request": self._fwd_req,
            "/forward_response": self._fwd_resp,
            "/skip_upstream": self._skip,
        }
        handler = routes.get(p)
        if handler:
            return handler()
        self.send_response(404)
        self.end_headers()

    def _completions(self):
        global turn_counter
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        try:
            body = json.loads(raw)
        except json.JSONDecodeError:
            return self._json(400, {"error": "bad json"})

        turn_counter += 1
        raw_str = raw.decode("utf-8")
        d = extract_display(body)
        drain(edited_request)
        drain(edited_response)
        stripped = strip_b64_from_body(raw_str)

        broadcast("incoming_request", {
            "turn": turn_counter, "model": d["model"], "story": d["story"],
            "screenshot_b64": d["screenshot_b64"], "raw_body_stripped": stripped,
        })
        print(f"  [T{turn_counter}] Request intercepted, waiting for browser...")
        edited = edited_request.get()

        # Check for manual skip
        try:
            sig = json.loads(edited)
            if sig.get("__skip"):
                mc = sig.get("content", "")
                print(f"  [T{turn_counter}] Manual response ({len(mc)}c)")
                resp = {
                    "id": f"franz-{turn_counter}", "object": "chat.completion",
                    "created": int(time.time()), "model": d["model"] or "human",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": mc}, "finish_reason": "stop"}],
                    "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                }
                self._json(200, resp)
                broadcast("turn_complete", {"turn": turn_counter, "mode": "manual"})
                return
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass

        # Normal forward
        try:
            edit_data = json.loads(edited)
            edited_body_stripped = edit_data.get("raw_body_stripped", stripped)
            canvas_b64 = edit_data.get("canvas_b64", d["screenshot_b64"])
        except (json.JSONDecodeError, TypeError, AttributeError):
            edited_body_stripped = edited
            canvas_b64 = d["screenshot_b64"]

        final_body = inject_b64_into_body(edited_body_stripped, canvas_b64, raw_str)
        broadcast("forwarding", {"turn": turn_counter})
        print(f"  [T{turn_counter}] Forwarding to upstream...")

        sc, up_resp = forward_upstream(final_body)
        print(f"  [T{turn_counter}] Upstream HTTP {sc} ({len(up_resp)}c)")

        ac = ""
        try:
            ro = json.loads(up_resp)
            ch = ro.get("choices", [])
            if ch:
                ac = ch[0].get("message", {}).get("content", "")
        except Exception:
            pass

        broadcast("incoming_response", {
            "turn": turn_counter, "status": sc,
            "assistant_content": ac, "raw_body": up_resp,
        })
        print(f"  [T{turn_counter}] Response intercepted, waiting for browser...")
        final = edited_response.get()
        print(f"  [T{turn_counter}] Returning to main.py ({len(final)}c)")

        p = final.encode("utf-8")
        self.send_response(sc)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)
        broadcast("turn_complete", {"turn": turn_counter, "mode": "proxy"})

    def _fwd_req(self):
        b = self._read_json()
        edited_request.put(json.dumps(b))
        self._json(200, {"ok": True})

    def _fwd_resp(self):
        b = self._read_json()
        edited_response.put(b.get("raw_body", "{}"))
        self._json(200, {"ok": True})

    def _skip(self):
        b = self._read_json()
        edited_request.put(json.dumps({"__skip": True, "content": b.get("content", "")}))
        self._json(200, {"ok": True})

    def _read_json(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def _html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(_HTML_BYTES)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(_HTML_BYTES)

    def _sse(self):
        self.send_response(200)
        for k, v in [("Content-Type", "text/event-stream"), ("Cache-Control", "no-cache"),
                      ("Connection", "keep-alive"), ("Access-Control-Allow-Origin", "*")]:
            self.send_header(k, v)
        self.end_headers()
        q: queue.Queue[str] = queue.Queue(maxsize=100)
        with sse_lock:
            sse_clients.append(q)
        try:
            while True:
                try:
                    d = q.get(timeout=15)
                    self.wfile.write(d.encode("utf-8"))
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b"event: ping\ndata: {}\n\n")
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with sse_lock:
                if q in sse_clients:
                    sse_clients.remove(q)

    def _json(self, s, o):
        p = json.dumps(o, ensure_ascii=False).encode("utf-8")
        self.send_response(s)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(p)))
        self.end_headers()
        self.wfile.write(p)


class Server(HTTPServer):
    def process_request(self, req, addr):
        threading.Thread(target=self._t, args=(req, addr), daemon=True).start()

    def _t(self, req, addr):
        try:
            self.finish_request(req, addr)
        except Exception:
            self.handle_error(req, addr)
        finally:
            self.shutdown_request(req)


def launch_main():
    if not MAIN_PY.is_file():
        print(f"  WARNING: {MAIN_PY} not found, skipping auto-launch")
        return
    print(f"  Launching {MAIN_PY}...")
    subprocess.Popen([sys.executable, str(MAIN_PY)], cwd=str(MAIN_PY.parent))


def main():
    srv = Server((HOST, PORT), Handler)
    url = f"http://localhost:{PORT}"
    print(f"\n  FRANZ MITM Debug Panel")
    print(f"  Proxy:    {url}")
    print(f"  Upstream: http://{UP_HOST}:{UP_PORT}")
    print(f"  HTML:     {len(_HTML_BYTES)} bytes (generated)")
    print(f"  Ctrl+C to stop\n")
    try:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()
    except Exception:
        pass
    threading.Timer(2.0, launch_main).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        srv.server_close()


if __name__ == "__main__":
    main()
