# Socratic Agent – Ask First, Test on Demand

## What is it?
A small, auditable Socratic tutor that externalizes the learner’s reasoning. It uses a deference gate (τ, γ) to delay answers and restricts interaction to audited speech acts.

## 10-second preview
![demo](assets/preview.gif)

## Try it in 2 minutes (local)
```bash
git clone https://github.com/CarlosUrteaga/socratic-agent
cd socratic-agent
python -m venv .venv && source .venv/bin/activate   # or conda env create -f environment.yml
pip install -r requirements.txt
hf auth login # log with hf account
cd source-code-demo
# run example
python server.py &  (sleep 10  && (clear  & python demo.py))
# run chat
python server.py &  (sleep 10  && (clear  & python chat_app.py))