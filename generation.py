"""
Answer generation with a citation contract and explicit abstention.

The prompt contract
-------------------
In a clinical setting the generation stage is where a retrieval bug turns into
a fabricated medical claim, so the contract is enforced at three points rather
than trusted once:

1. The system prompt requires every claim to come from the supplied context and
   to carry a citation, and requires refusal when the context does not contain
   the answer.
2. The response is parsed into `{answer, cited_ids, abstained}` - citations are
   a structured field, not a string the reader has to trust.
3. `evaluation.evaluate_answers` then *verifies* the citations against the
   context and the gold labels. A model that cites a document id it was never
   shown is caught here, which is the failure mode a citation-shaped prompt
   quietly invites.

Two implementations
-------------------
- `AnthropicLLM`      - real Claude call. Used when credentials are available.
- `ExtractiveLLM`     - deterministic, no network, no model. It follows the same
  contract (cites, and abstains on weak retrieval) so the answer-level harness
  produces real numbers on any machine. It is explicitly not a language model
  and its "answers" are quoted passage sentences.
"""

import json
import os
import re
from typing import Optional

from contracts import Candidate

MODEL = "claude-opus-5"

SYSTEM_PROMPT = """You answer clinical reference questions using ONLY the numbered context passages provided.

Rules:
1. Ground every statement in the context. Never use outside knowledge, and never infer a clinical claim the passages do not state.
2. Cite the passage id (for example DM04) for every claim you make.
3. If the context does not contain the answer, you MUST abstain. Say you do not have a relevant source. Do not guess, and do not answer from general medical knowledge. An abstention is the correct answer to an unsupported question.
4. If the context only partially covers the question, answer the covered part and say explicitly what is not supported.
5. Be concise: at most three sentences.

Respond with a single JSON object and nothing else:
{"answer": "<answer text, or the refusal>", "cited_ids": ["<passage ids you used>"], "abstained": <true|false>}

Set "abstained" to true and "cited_ids" to [] when the context does not support an answer."""

USER_TEMPLATE = """Context passages:
{context}

Question: {question}"""


def format_context(context: list[Candidate]) -> str:
    return "\n".join(f"[{c.id}] {c.text}" for c in context)


def _abstention(reason: str) -> dict:
    return {
        "answer": (
            "I don't have a relevant source for that in the knowledge base, "
            "so I can't answer it."
        ),
        "cited_ids": [],
        "abstained": True,
        "abstain_reason": reason,
    }


def _parse_response(text: str) -> dict:
    """Parse the model's JSON envelope, tolerating prose or fences around it."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {
            "answer": text.strip(),
            "cited_ids": re.findall(r"\b([A-Z]{2}\d{2})\b", text),
            "abstained": False,
            "parse_error": "no JSON object in response",
        }
    try:
        payload = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        return {
            "answer": text.strip(),
            "cited_ids": re.findall(r"\b([A-Z]{2}\d{2})\b", text),
            "abstained": False,
            "parse_error": f"invalid JSON: {exc}",
        }
    return {
        "answer": str(payload.get("answer", "")).strip(),
        "cited_ids": [str(i) for i in payload.get("cited_ids", [])],
        "abstained": bool(payload.get("abstained", False)),
    }


class AnthropicLLM:
    """Real Claude generation call.

    Requires `pip install anthropic` and a credential (`ANTHROPIC_API_KEY`, or
    an `ant auth login` profile - the SDK resolves both from the environment).
    """

    def __init__(
        self,
        model: str = MODEL,
        max_tokens: int = 4096,
        effort: str = "low",
        enable_refusal_fallback: bool = True,
    ):
        import anthropic

        self.name = f"anthropic[{model}]"
        self.model = model
        self.max_tokens = max_tokens
        # This task is short, grounded extraction rather than open reasoning, so
        # low effort is the right cost/quality point. Raise it if you add
        # multi-passage synthesis or conflict resolution between sources.
        self.effort = effort
        self.enable_refusal_fallback = enable_refusal_fallback
        self.client = anthropic.Anthropic()
        self._errors = anthropic

    @staticmethod
    def available() -> bool:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            return False
        # An unset ANTHROPIC_API_KEY does not mean there are no credentials: the
        # SDK also resolves an `ant auth login` profile from ~/.config/anthropic.
        return bool(
            os.environ.get("ANTHROPIC_API_KEY")
            or os.environ.get("ANTHROPIC_AUTH_TOKEN")
            or os.path.isdir(os.path.expanduser("~/.config/anthropic"))
        )

    def generate(self, query_text: str, context: list[Candidate]) -> dict:
        if not context:
            return _abstention("empty context")

        request = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM_PROMPT,
            "output_config": {"effort": self.effort},
            "messages": [
                {
                    "role": "user",
                    "content": USER_TEMPLATE.format(
                        context=format_context(context), question=query_text
                    ),
                }
            ],
        }

        try:
            if self.enable_refusal_fallback:
                # Clinical phrasing can trip safety classifiers; a server-side
                # fallback keeps one refusal from becoming a pipeline failure.
                response = self.client.beta.messages.create(
                    betas=["server-side-fallback-2026-07-01"],
                    fallbacks="default",
                    **request,
                )
            else:
                response = self.client.messages.create(**request)
        except self._errors.APIStatusError as exc:
            raise RuntimeError(f"Claude API error {exc.status_code}: {exc.message}") from exc

        if response.stop_reason == "refusal":
            category = getattr(response.stop_details, "category", None)
            return _abstention(f"model refusal ({category})")

        text = "".join(block.text for block in response.content if block.type == "text")
        parsed = _parse_response(text)
        parsed["usage"] = {
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens,
        }
        return parsed


class ExtractiveLLM:
    """Deterministic extractive baseline - NOT a language model.

    It exists so the answer-level metrics (citation validity, abstention
    behaviour, groundedness) can be exercised and regression-tested without
    network access or spend. Because it only ever quotes a retrieved sentence,
    it is trivially faithful; what it genuinely tests is the *retrieval-driven*
    part of answer quality - whether the pipeline hands over the right passage,
    and whether the abstention gate fires when it should not have a source.

    A real LLM is strictly harder on the faithfulness metrics, which is the
    point: this is a floor, not a substitute.
    """

    name = "extractive-baseline"

    def __init__(self, abstain_below: Optional[float] = None, max_sentences: int = 2):
        self.abstain_below = abstain_below
        self.max_sentences = max_sentences

    def generate(self, query_text: str, context: list[Candidate]) -> dict:
        if not context:
            return _abstention("empty context")

        top = context[0]
        if self.abstain_below is not None and top.score < self.abstain_below:
            return _abstention(
                f"top context score {top.score:.3f} below threshold {self.abstain_below:.3f}"
            )

        sentences = re.split(r"(?<=[.!?])\s+", top.text)
        query_terms = {t for t in re.findall(r"[a-z0-9]+", query_text.lower()) if len(t) > 3}
        ranked = sorted(
            sentences,
            key=lambda s: len(query_terms & set(re.findall(r"[a-z0-9]+", s.lower()))),
            reverse=True,
        )
        quoted = " ".join(ranked[: self.max_sentences]).strip()

        return {
            "answer": f"{quoted} [{top.id}]",
            "cited_ids": [top.id],
            "abstained": False,
            "abstain_reason": None,
        }


def default_llm(abstain_below: Optional[float] = None, prefer_real: bool = True):
    """Return a real Claude client when credentials exist, else the baseline."""
    if prefer_real and AnthropicLLM.available():
        try:
            return AnthropicLLM()
        except Exception:
            pass
    return ExtractiveLLM(abstain_below=abstain_below)
