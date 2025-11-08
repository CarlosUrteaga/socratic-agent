# Socratic Agent – Ask First, Test on Demand

## What is it?
A small, auditable Socratic tutor that externalizes the learner’s reasoning. It uses a deference gate (τ, γ) to delay answers and restricts interaction to audited speech acts.


## Try it in 2 minutes (local)
```bash
git clone https://github.com/CarlosUrteaga/socratic-agent
cd socratic-agent
python -m venv .socratic && source .socratic/bin/activate   # or conda env create -f environment.yml
pip install -r requirements.txt
hf auth login # log with hf account
````

```bash
cd source-code-demo
# run example
python server.py &  (sleep 10  && (clear  & python demo.py))
# run chat
# python server.py &  (sleep 10  && (clear  & python chat_app.py))
```

chat flow example

```bash
# 1) set source of truth
RAG[https://en.wikipedia.org/wiki/Retrieval-augmented_generation,https://fastapi.tiangolo.com/]: Explain Retrieval-Augmented Generation and why it reduces hallucinations.

# 2) Learner prior knowledge/preferences → controller RETRIEVES and asks for a draft
I know it retrieves external docs before answering; I want the intuition for fewer hallucinations.

# 3) Learner draft → controller provides feedback + a quick quiz
My current understanding is RAG pulls relevant text into the prompt so the model grounds its answer.  
Evidence: the snippets say RAG retrieves from databases/web before generation (see Wikipedia). 
Reasoning: the model can cite facts instead of guessing. Limits: bad or off-topic retrieval hurts.

# 4) Quiz answers → controller finalizes
It reduces hallucinations because retrieved context constrains the model to verifiable facts. RAG can fail if retrieval is irrelevant/incorrect or the model ignores context.
```
img

![01](img/01.png)
