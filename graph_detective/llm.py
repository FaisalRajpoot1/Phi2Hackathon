"""The language models, behind two small functions:

- local_generate(): Phi-2 or Phi-4-mini, running on this computer (or on a GPU).
- gemini_json(): Google's Gemini, answering in JSON that follows a schema.

Nothing is loaded, and no key is read, until one of them is called. PyTorch and
transformers are imported only when a local model is first used, so the app
starts fast when it only needs Gemini.
"""
import functools
import os
import time

from google import genai
from pydantic import ValidationError

PHI2 = "microsoft/phi-2"
PHI4_MINI = "microsoft/Phi-4-mini-instruct"
MAX_NEW_TOKENS = 256
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash-lite"
# None: a GPU if this computer has one, else the CPU. The live demo sets "cuda":
# on ZeroGPU the models must go to the GPU when the app starts, before one is attached.
DEVICE = None


class PromptTooLong(ValueError):
    """The prompt and the answer together don't fit in the model's context."""


class MissingKey(RuntimeError):
    """No Gemini API key was given or set."""


class ModelUnavailable(RuntimeError):
    """Google has no model with this name, for example because it was retired."""


class GeminiBusy(RuntimeError):
    """Gemini kept answering "too many requests" (HTTP 429)."""


class BadAnswer(ValueError):
    """The answer was not the JSON that was asked for. `raw` keeps the answer."""

    def __init__(self, message, raw):
        super().__init__(message)
        self.raw = raw


# ---- Local models (Phi) ---------------------------------------------------

@functools.lru_cache(maxsize=2)
def load_local_model(model_id):
    """Load a model and its tokenizer once. At most two models are kept (the live demo
    keeps Phi-2 and Phi-4-mini); the laptop app only ever loads one.

    The model loads in bfloat16, also on a CPU: measured on a 16 GB laptop, Phi-2
    peaks at 5.7 GB this way, against 11.0 GB in float32 (the original's setting).
    """
    import torch
    import transformers

    device = DEVICE or ("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
    return tokenizer, model.to(device).eval()


def format_prompt(tokenizer, instruction):
    """A chat model gets its own chat template. Phi-2, a base model with no template,
    gets the "Instruct: ... Output:" format from its model card."""
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template([{"role": "user", "content": instruction}],
                                             tokenize=False, add_generation_prompt=True)
    return f"Instruct: {instruction}\nOutput:"


def local_generate(instruction, model_id=PHI2, max_new_tokens=MAX_NEW_TOKENS):
    """Run a local model with greedy decoding, and return only the new text.

    A prompt that doesn't fit in the model's context is refused, never cut.
    """
    tokenizer, model = load_local_model(model_id)
    inputs = tokenizer(format_prompt(tokenizer, instruction), return_tensors="pt")
    prompt_tokens = inputs["input_ids"].shape[1]
    limit = model.config.max_position_embeddings
    if prompt_tokens + max_new_tokens > limit:
        raise PromptTooLong(f"The prompt has {prompt_tokens} tokens. With {max_new_tokens} more for the "
                            f"answer, that is more than {model_id} can read ({limit} tokens).")
    inputs = {name: tensor.to(model.device) for name, tensor in inputs.items()}
    output = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    return tokenizer.decode(output[0, prompt_tokens:], skip_special_tokens=True).strip()


# ---- Gemini -----------------------------------------------------------------

def gemini_key():
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def http_status(error):
    """The HTTP status of a google-genai error. The Interactions API's errors carry
    `status_code`; the older API's errors carry `code`."""
    status = getattr(error, "status_code", None) or getattr(error, "code", None)
    return status if isinstance(status, int) else None


def gemini_json(prompt, schema, model=DEFAULT_GEMINI_MODEL, api_key=None, client=None,
                retries=4, sleep=time.sleep):
    """Ask Gemini, and return its answer as an instance of the pydantic `schema`.

    The key is read now, not at import. When Gemini is busy (HTTP 429), this waits
    5, 10, 20, then 40 seconds before trying again. Nothing is stored on Google's side.
    """
    if client is None:
        key = api_key or gemini_key()
        if not key:
            raise MissingKey("No Gemini API key. Paste one in the sidebar, or set GEMINI_API_KEY.")
        client = genai.Client(api_key=key)
    for attempt in range(retries + 1):
        try:
            interaction = client.interactions.create(
                model=model,
                input=prompt,
                store=False,
                response_format={"type": "text", "mime_type": "application/json",
                                 "schema": schema.model_json_schema()},
            )
            break
        except Exception as error:
            status = http_status(error)
            if status == 404:
                raise ModelUnavailable(f"The Gemini model '{model}' is not available. It may have been "
                                       "retired; please pick another model.") from error
            if status != 429:
                raise
            if attempt == retries:
                raise GeminiBusy("Gemini's free limit was reached. Please wait a minute and try again.") from error
            sleep(5 * 2 ** attempt)
    try:
        return schema.model_validate_json(interaction.output_text)
    except ValidationError as error:
        raise BadAnswer("Gemini's answer was not the JSON that was asked for.", raw=interaction.output_text) from error
