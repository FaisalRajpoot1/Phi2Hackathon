"""Fakes for the slow or outside parts: loading a language model, and Google's API.

There are two kinds:
- Fake *modules* for the legacy code, so the original files load without torch,
  the 5 GB Phi-2 model, the `spaces` package or the retired Google SDK.
- A fake tokenizer and model for the new code. They use real torch tensors,
  so the new code runs exactly as it does with a real model.
"""
import queue
import types
from types import SimpleNamespace

import torch

REPLY = "RING: Person 1, Person 2"
REPLY_IDS = [7, 8, 9]


# ---- For the legacy code ------------------------------------------------

class LegacyStreamer:
    """Stands in for transformers.TextIteratorStreamer: text goes in with put(), out by iterating."""

    def __init__(self, tokenizer, **kwargs):
        self._queue = queue.Queue()

    def put(self, text):
        self._queue.put(text)

    def end(self):
        self._queue.put(None)

    def __iter__(self):
        while (text := self._queue.get(timeout=10)) is not None:
            yield text


class LegacyBatch(dict):
    def to(self, device):
        return self


class LegacyTokenizer:
    """One token per word."""

    def __call__(self, texts, return_tensors=None):
        return LegacyBatch(input_ids=[[1] * len(text.split()) for text in texts])


class LegacyModel:
    """Answers REPLY through the streamer, like model.generate does, and records its settings."""

    def __init__(self):
        self.generate_kwargs = None

    def to(self, device):
        return self

    def generate(self, **kwargs):
        self.generate_kwargs = kwargs
        kwargs["streamer"].put(REPLY)
        kwargs["streamer"].end()


def legacy_modules(tool_decorator):
    """Fake modules for loading legacy/crewai/tools/*.py.

    Returns the modules to put in sys.modules, and a record of what the code did with them.
    """
    record = SimpleNamespace(loads=0, model=LegacyModel())

    def counted(result):
        def from_pretrained(*args, **kwargs):
            record.loads += 1
            return result
        return from_pretrained

    torch_module = types.ModuleType("torch")
    torch_module.cuda = SimpleNamespace(is_available=lambda: False)
    torch_module.backends = SimpleNamespace(mps=SimpleNamespace(is_available=lambda: False))
    torch_module.float16, torch_module.float32 = "float16", "float32"

    transformers = types.ModuleType("transformers")
    transformers.AutoTokenizer = SimpleNamespace(from_pretrained=counted(LegacyTokenizer()))
    transformers.AutoModelForCausalLM = SimpleNamespace(from_pretrained=counted(record.model))
    transformers.TextIteratorStreamer = LegacyStreamer

    langchain = types.ModuleType("langchain")
    langchain.tools = types.ModuleType("langchain.tools")
    langchain.tools.tool = tool_decorator

    genai = types.ModuleType("google.generativeai")
    genai.configure = lambda api_key: None
    genai.GenerativeModel = lambda name: SimpleNamespace(model_name=name)
    api_core = types.ModuleType("google.api_core")
    api_core.exceptions = SimpleNamespace(DeadlineExceeded=TimeoutError)

    modules = {
        "torch": torch_module,
        "transformers": transformers,
        "spaces": types.ModuleType("spaces"),
        "langchain": langchain,
        "langchain.tools": langchain.tools,
        "google.generativeai": genai,
        "google.api_core": api_core,
    }
    return modules, record


# ---- For the new code ---------------------------------------------------

class FakeTokenizer:
    """Stands in for a Hugging Face tokenizer: one token per word, and REPLY_IDS decode to REPLY.
    It records every text it was given. Like Phi-2's tokenizer, it has no chat template
    unless a test gives it one (Phi-4-mini's tokenizer has one)."""

    eos_token_id = 0

    def __init__(self):
        self.texts = []
        self.chat_template = None

    def __call__(self, text, return_tensors=None):
        self.texts.append(text)
        ids = torch.ones((1, len(text.split())), dtype=torch.long)
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        text = "".join(f"<|{m['role']}|>{m['content']}<|end|>" for m in messages)
        return text + ("<|assistant|>" if add_generation_prompt else "")

    def decode(self, ids, skip_special_tokens=False):
        return REPLY if ids.tolist() == REPLY_IDS else "<unexpected ids>"


class FakeCausalLM:
    """Stands in for a causal language model with Phi-2's 2,048-token context: it writes
    REPLY_IDS after the prompt, and records the settings of every generate() call."""

    device = torch.device("cpu")
    config = SimpleNamespace(max_position_embeddings=2048)

    def __init__(self):
        self.calls = []

    def to(self, device):
        return self

    def eval(self):
        return self

    def generate(self, input_ids, attention_mask=None, **kwargs):
        self.calls.append(dict(kwargs, prompt_tokens=input_ids.shape[1]))
        return torch.cat([input_ids, torch.tensor([REPLY_IDS])], dim=1)
