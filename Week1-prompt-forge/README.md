# Week 1 — PromptForge

This folder contains the Week 1 PromptForge app for the MSTC Hackathon.

## Contents

- `app.py` — Gradio web app with multiple personas:
  - Technical Explainer
  - Debate Coach
  - Code Reviewer
  - Creative Writer

- `requirements.txt` — Python dependencies required to run the app.

## Setup

1. Create or activate a Python virtual environment.

```bash
python -m venv venv
.\venv\Scripts\activate
```

2. Install the required packages.

```bash
pip install -r requirements.txt
```

3. Add your Groq API key to a `.env` file.

```ini
GROQ_API_KEY=your_api_key_here
```

## Run the Gradio app

```bash
python app.py
```

Then open the local URL shown in the terminal (default: `http://localhost:7860`).

## Run the Week 1 task example

```bash
python app.py
```

## Notes

- `app.py` is the main interactive interface.

