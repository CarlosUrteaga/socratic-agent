# Socratic Agent – Ask First, Test on Demand

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/CarlosUrteaga/socratic-agent/blob/main/notebooks/demo.ipynb)
[![HF Spaces](https://img.shields.io/badge/🤗-Spaces-black.svg)](https://huggingface.co/spaces/your-namespace/socratic-agent) 
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://github.com/CarlosUrteaga/socratic-agent/pkgs/container/socratic-agent)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

**Demo:** 10-sec GIF below • **Colab:** one-click run • **Paper artifact:** see `v1.0.0` release

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
export OPENAI_API_KEY=...   # or set your provider key
make demo                   # runs a minimal conversation
