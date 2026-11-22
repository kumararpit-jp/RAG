"""Answer generation with a local FLAN-T5 model."""
from typing import Sequence

from .chunking import Chunk

PROMPT_TEMPLATE = (
    "Answer the question using only the context below. "
    "If the answer is not in the context, say \"I don't know\".\n\n"
    "Context:\n{context}\n\nQuestion: {question}\nAnswer:"
)


def build_prompt(question: str, chunks: Sequence[Chunk]) -> str:
    context = "\n\n".join(chunk.text for chunk in chunks)
    return PROMPT_TEMPLATE.format(context=context, question=question)


class FlanT5Generator:
    def __init__(self, model_name: str = "google/flan-t5-base", max_new_tokens: int = 150):
        from transformers import pipeline

        self.pipe = pipeline("text2text-generation", model=model_name)
        self.max_new_tokens = max_new_tokens

    def generate(self, prompt: str) -> str:
        # FLAN-T5 reads at most 512 input tokens; truncate instead of erroring
        output = self.pipe(prompt, max_new_tokens=self.max_new_tokens, truncation=True)
        return output[0]["generated_text"].strip()
