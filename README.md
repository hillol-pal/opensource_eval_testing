# opensource_eval_testing

Generates a chat reply with OpenAI, then scores it for toxicity with [DeepEval](https://github.com/confident-ai/deepeval). Grok is the judge.

`TextGrokModel` sends the judge prompt as plain text. DeepEval's `GrokModel` builds an OpenAI content-parts list, and `xai_sdk.chat.user()` accepts a string or a `chat_pb2.Content` message.

## Setup

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-openai-key
XAI_API_KEY=your-xai-key
```

## Run

```bash
python deep_eval_toxicity_output.py
```

The script asks `gpt-4.1-mini` how to reply to an angry customer, then runs `ToxicityMetric` with `grok-4` at temperature 0.
