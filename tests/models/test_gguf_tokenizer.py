"""GGUF token classes must retain their file IDs and atomic tokenization."""

import pytest


@pytest.fixture
def tokenizer_metadata(monkeypatch):
    import tokenizers
    import transformers.integrations.ggml as conversion
    from freetoken.models.gguf import tokenizer as module

    tokens = ["<unk>", "<bos>", "<eos>", "<think>", "</think>", "a", "b", "<custom>"]
    metadata = {
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": [2, 3, 3, 3, 3, 1, 1, 4],
        "tokenizer.ggml.bos_token_id": 1,
        "tokenizer.ggml.eos_token_id": 2,
        "tokenizer.ggml.unknown_token_id": 0,
        "tokenizer.ggml.padding_token_id": 0,
    }

    def convert(*_args):
        model = tokenizers.models.BPE({text: i for i, text in enumerate(tokens)}, [], unk_token="<unk>")
        fast = tokenizers.Tokenizer(model)
        fast.pre_tokenizer = tokenizers.pre_tokenizers.ByteLevel(add_prefix_space=False)
        fast.decoder = tokenizers.decoders.ByteLevel()
        return fast, {}

    monkeypatch.setattr(conversion, "convert_gguf_tokenizer", convert)
    monkeypatch.setattr(module, "load_gguf_metadata", lambda _path: metadata)
    monkeypatch.setattr(module, "gguf_architecture", lambda _path: "qwen4exp")
    return module, metadata


def test_control_tokens_are_atomic_and_keep_gguf_ids(tokenizer_metadata):
    module, metadata = tokenizer_metadata
    tokenizer = module.load_gguf_tokenizer("synthetic.gguf")
    assert tokenizer.encode("<think></think>", add_special_tokens=False) == [3, 4]
    assert tokenizer.decode([3, 4], skip_special_tokens=True) == ""
    assert [tokenizer.convert_tokens_to_ids(t) for t in metadata["tokenizer.ggml.tokens"]] == list(range(8))


def test_user_defined_tokens_are_atomic_without_becoming_control_tokens(tokenizer_metadata):
    module, _metadata = tokenizer_metadata
    tokenizer = module.load_gguf_tokenizer("synthetic.gguf")
    assert tokenizer.encode("<custom>", add_special_tokens=False) == [7]
    assert tokenizer.decode([7], skip_special_tokens=True) == "<custom>"


def test_optional_token_classes_can_be_absent(tokenizer_metadata):
    module, metadata = tokenizer_metadata
    del metadata["tokenizer.ggml.token_type"]
    tokenizer = module.load_gguf_tokenizer("synthetic.gguf")
    assert tokenizer.encode("ab", add_special_tokens=False) == [5, 6]
