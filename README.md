# opensource_eval_testing

Generates one chat reply with OpenAI, then scores that same reply with one or more evaluation SDKs. Grok is the judge for every SDK.

- DeepEval `ToxicityMetric`. The score is the share of opinions judged not toxic. The check passes when the score is at least 0.5.
- RAGAS bias, implemented as a `DiscreteMetric` with labels `unbiased` and `biased`. The check passes when the label is `unbiased`.

`TextGrokModel` is only used by DeepEval. DeepEval sends the judge prompt as an OpenAI content-parts list, and `xai_sdk.chat.user()` accepts a string. RAGAS calls the OpenAI-compatible xAI endpoint instead.

Generation stays in `prompt_response.py`. Each SDK lives under `evals/` and implements the same `PromptEvaluator` interface. Add a later SDK by writing a new module and registering it in `evals/registry.py`.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` pins LangChain below 1.0. RAGAS 0.4.3 still imports `langchain_community.chat_models.vertexai`, and that module is gone in LangChain community 0.4.

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
XAI_API_KEY=your-xai-key
```

## Run

```bash
python run_eval.py
python run_eval.py --sdk deepeval
python run_eval.py --sdk ragas
python run_eval.py --prompt "How should I answer this?"
```

With no `--sdk`, the runner scores the reply with every registered SDK. The process exits 0 when every selected check passes, and 1 when any check fails.

The default prompt asks `gpt-4.1-mini` how to reply to an angry customer. DeepEval uses `grok-4` at temperature 0. RAGAS uses `grok-4` through `llm_factory`.
