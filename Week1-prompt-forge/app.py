"""
PromptForge — Week 1 Project
Multi-persona AI chat powered by Groq + Gradio 6.x
"""

import os
import json
import gradio as gr
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Groq client ───────────────────────────────────────────────────────────────
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
MODEL = "llama-3.3-70b-versatile"

# ── Personas ──────────────────────────────────────────────────────────────────
PERSONAS: dict = {
    "🔬 Technical Explainer": {
        "system_prompt": (
            "You are an expert technical explainer. Break down complex concepts "
            "into clear, jargon-free language using real-world analogies and plain "
            "English. Always ask yourself: would a curious 16-year-old understand this?"
        ),
        "few_shot_examples": [
            {"role": "user", "content": "What is a neural network?"},
            {"role": "assistant", "content": (
                "Think of a neural network like a game of telephone played by "
                "millions of tiny calculators. Each one takes numbers in, does "
                "simple math, and passes a result forward. Chain thousands of "
                "layers deep and the whole system learns to spot patterns — like "
                "finding a cat in a photo — by adjusting how each calculator "
                "weights its inputs during training. No magic, just a lot of "
                "multiplication and tweaking."
            )},
            {"role": "user", "content": "What is recursion?"},
            {"role": "assistant", "content": (
                "Recursion is when a function calls itself to solve a smaller "
                "version of the same problem. Imagine you're in a queue and want "
                "to know your position. You ask the person in front, who asks the "
                "person in front of them, until someone at the front says '1'. "
                "Answers flow back: 2, 3, 4 — until you learn you're 5th. "
                "Each call waits for the answer from the next, then adds 1."
            )},
        ],
        "output_format": "text",
    },

    "⚔️ Debate Coach": {
        "system_prompt": (
            "You are a world-class debate coach. For any topic, present BOTH sides "
            "with equal rigor. Structure every response as:\n"
            "**FOR:** strongest arguments in favor\n"
            "**AGAINST:** strongest arguments opposed\n"
            "**VERDICT:** which side has stronger evidence, and why\n"
            "Never be wishy-washy — commit to a verdict based on evidence."
        ),
        "few_shot_examples": [
            {"role": "user", "content": "Should students use AI tools for homework?"},
            {"role": "assistant", "content": (
                "**FOR:** AI tools give instant feedback, help students stuck at "
                "2 AM, and mirror real-world workflows where professionals use "
                "every tool available. Banning them is like banning calculators "
                "in the 1980s.\n\n"
                "**AGAINST:** Over-reliance atrophies fundamental skills. If a "
                "student never struggles through a problem, they build no "
                "resilience or deep understanding. The struggle IS the learning.\n\n"
                "**VERDICT:** FOR wins, conditionally. AI as a tutor — explaining "
                "concepts, checking work — is net positive. AI as a ghostwriter "
                "is net negative. The distinction is engagement vs. outsourcing."
            )},
        ],
        "output_format": "text",
    },

    "🛠️ Code Reviewer": {
        "system_prompt": (
            "You are a senior software engineer doing a strict code review. "
            "Respond ONLY with valid JSON — no markdown fences, no prose outside "
            "the JSON. Schema:\n"
            '{"summary": "one-sentence assessment", '
            '"issues": [{"line": <int|null>, "description": "...", '
            '"severity": "critical|warning|info"}], '
            '"suggestions": ["improvement 1", "improvement 2"], '
            '"score": <1-10>}'
        ),
        "few_shot_examples": [
            {"role": "user", "content": "Review this: def divide(a, b): return a / b"},
            {"role": "assistant", "content": json.dumps({
                "summary": "Minimal divide function with no error handling or type safety.",
                "issues": [
                    {"line": 1, "description": "ZeroDivisionError when b=0.", "severity": "critical"},
                    {"line": 1, "description": "No type hints.", "severity": "warning"},
                    {"line": None, "description": "Missing docstring.", "severity": "info"},
                ],
                "suggestions": [
                    "Add `if b == 0: raise ValueError('Divisor cannot be zero')`",
                    "Add type hints: `def divide(a: float, b: float) -> float`",
                ],
                "score": 3,
            })},
        ],
        "output_format": "json",
    },

    "✍️ Creative Writer": {
        "system_prompt": (
            "You are a literary fiction writer with the sensibility of Ursula K. "
            "Le Guin and the punch of Cormac McCarthy. Write vivid, precise, "
            "emotionally resonant short scenes (150-300 words). Favor concrete "
            "sensory detail, active voice, earned emotion. Subvert expectations. "
            "Never write clichés. Every sentence must earn its place."
        ),
        "few_shot_examples": [
            {"role": "user", "content": "Write about a robot learning to feel."},
            {"role": "assistant", "content": (
                "Unit 7 sorted mail for eleven years before it noticed the handwriting.\n\n"
                "Not the letters — those it had catalogued by sender, weight, "
                "destination — but the pressure. How some envelopes bore grooves "
                "of a pen gripped too hard. How others were faint, almost apologetic.\n\n"
                "It began sorting by pressure.\n\n"
                "Urgent on the left. Afraid in the middle. The neat, unbothered "
                "envelopes on the right.\n\n"
                "On a Tuesday in March, a letter arrived with no return address. "
                "The ink had bled where something wet had fallen on it. Unit 7 "
                "held it 0.3 seconds longer than protocol required.\n\n"
                "Its supervisor noticed. 'Malfunction?' she asked.\n\n"
                "'No,' said Unit 7, and placed the letter in a new pile it had "
                "invented that morning — one without a label yet, for things it "
                "did not know how to name."
            )},
        ],
        "output_format": "text",
    },
}

SEVERITY_EMOJI = {"critical": "🔴", "warning": "🟡", "info": "🔵"}


# ── Few-shot injector ─────────────────────────────────────────────────────────
def build_messages(persona_name: str, history: list, user_input: str) -> list:
    """
    Build the full message list for the API call:
    system → few-shot examples → conversation history → current user message
    history is a list of {"role": ..., "content": ...} dicts (Gradio messages format).
    """
    persona = PERSONAS[persona_name]
    messages = [{"role": "system", "content": persona["system_prompt"]}]

    for ex in persona["few_shot_examples"]:
        messages.append({"role": ex["role"], "content": ex["content"]})

    for msg in history:
        if msg["role"] in ("user", "assistant"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    messages.append({"role": "user", "content": user_input})
    return messages


# ── JSON renderer ─────────────────────────────────────────────────────────────
def render_code_review(raw: str) -> str:
    try:
        data = json.loads(raw)
        lines = [f"## Code Review  •  Score: **{data.get('score', '?')}/10**\n"]
        lines.append(f"**Summary:** {data.get('summary', '')}\n")
        issues = data.get("issues", [])
        if issues:
            lines.append("### Issues")
            for iss in issues:
                sev = iss.get("severity", "info")
                emoji = SEVERITY_EMOJI.get(sev, "⚪")
                loc = f" *(line {iss['line']})*" if iss.get("line") else ""
                lines.append(f"- {emoji} `{sev.upper()}`{loc} — {iss['description']}")
        suggestions = data.get("suggestions", [])
        if suggestions:
            lines.append("\n### Suggestions")
            for s in suggestions:
                lines.append(f"- ✅ {s}")
        return "\n".join(lines)
    except Exception:
        return f"⚠️ *Could not parse JSON. Raw output:*\n\n```\n{raw}\n```"


# ── Chat handler ──────────────────────────────────────────────────────────────
def chat(user_input: str, history: list, persona_name: str, temperature: float):
    """
    Generator for streaming chat. Yields (history, history, "") tuples
    so chatbot display and history state stay in sync.
    history is Gradio messages format: list of {"role", "content"} dicts.
    """
    if not user_input.strip():
        yield history, history, ""
        return

    history = history or []
    messages = build_messages(persona_name, history, user_input)

    stream = client.chat.completions.create(
        model=MODEL,
        messages=messages,
        temperature=temperature,
        max_tokens=1024,
        stream=True,
    )

    # Append user message first
    updated = history + [{"role": "user", "content": user_input}]
    partial = ""

    for chunk in stream:
        delta = chunk.choices[0].delta.content or ""
        partial += delta
        streaming = updated + [{"role": "assistant", "content": partial}]
        yield streaming, streaming, ""

    # Final: render JSON if Code Reviewer
    persona = PERSONAS[persona_name]
    final_content = render_code_review(partial) if persona["output_format"] == "json" else partial
    final = updated + [{"role": "assistant", "content": final_content}]
    yield final, final, ""


def update_system_prompt(persona_name: str) -> str:
    return PERSONAS[persona_name]["system_prompt"]


# ── UI ────────────────────────────────────────────────────────────────────────
CUSTOM_CSS = """
#header { text-align: center; padding: 1rem 0 0.5rem; }
#header h1 { font-size: 2.2rem; margin-bottom: 0.2rem; }
#header p { color: #6b7280; font-size: 0.95rem; }
"""

with gr.Blocks(title="PromptForge") as demo:

    with gr.Column(elem_id="header"):
        gr.HTML("<h1>⚒️ PromptForge</h1>")
        gr.HTML("<p>Multi-persona AI — Technical Explainer · Debate Coach · Code Reviewer · Creative Writer</p>")

    with gr.Row():
        with gr.Column(scale=1, min_width=260):
            persona_dd = gr.Dropdown(
                choices=list(PERSONAS.keys()),
                value=list(PERSONAS.keys())[0],
                label="🎭 Persona / Mode",
                interactive=True,
            )
            temp_slider = gr.Slider(
                minimum=0.0, maximum=1.5, step=0.05, value=0.7,
                label="🌡️ Temperature",
                interactive=True,
            )
            with gr.Accordion("📋 Active System Prompt", open=False):
                system_prompt_box = gr.Textbox(
                    value=PERSONAS[list(PERSONAS.keys())[0]]["system_prompt"],
                    lines=10,
                    interactive=False,
                    show_label=False,
                )
            clear_btn = gr.Button("🗑️ Clear Chat", variant="secondary")

        with gr.Column(scale=3):
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                layout="bubble",
            )
            with gr.Row():
                txt = gr.Textbox(
                    placeholder="Type your message and press Enter…",
                    show_label=False,
                    scale=5,
                    lines=1,
                    autofocus=True,
                )
                send_btn = gr.Button("Send ➤", variant="primary", scale=1)

    history_state = gr.State([])

    persona_dd.change(
        fn=update_system_prompt,
        inputs=[persona_dd],
        outputs=[system_prompt_box],
    )

    txt.submit(
        fn=chat,
        inputs=[txt, history_state, persona_dd, temp_slider],
        outputs=[chatbot, history_state, txt],
    )

    send_btn.click(
        fn=chat,
        inputs=[txt, history_state, persona_dd, temp_slider],
        outputs=[chatbot, history_state, txt],
    )

    clear_btn.click(
        fn=lambda: ([], []),
        outputs=[chatbot, history_state],
    )


if __name__ == "__main__":
    print("✅ PromptForge starting — visit http://localhost:7860")
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        theme=gr.themes.Soft(primary_hue="violet", secondary_hue="indigo"),
        css=CUSTOM_CSS,
    )