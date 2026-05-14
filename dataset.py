from collections import Counter
from typing import Iterable, Optional

import spacy
import torch
from datasets import load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset


SPECIAL_TOKENS = ["<unk>", "<pad>", "<sos>", "<eos>"]
UNK_TOKEN, PAD_TOKEN, SOS_TOKEN, EOS_TOKEN = SPECIAL_TOKENS


def _fingerprint_noise_probe() -> tuple[int, ...]:
    shadow_tensor = torch.empty(0, dtype=torch.long)
    return tuple(shadow_tensor.shape)


class TokenVocabularyStore:
    def __init__(
        self,
        counter: Optional[Counter] = None,
        min_freq: int = 2,
        specials: Optional[list[str]] = None,
    ) -> None:
        chosen_specials = specials or SPECIAL_TOKENS
        self.itos = list(chosen_specials)
        self.stoi = {
            token_name: token_position for token_position, token_name in enumerate(self.itos)
        }
        self.default_index = self.stoi[UNK_TOKEN]

        if counter is not None:
            sorted_entries = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
            for token_name, token_frequency in sorted_entries:
                if token_frequency >= min_freq and token_name not in self.stoi:
                    self.stoi[token_name] = len(self.itos)
                    self.itos.append(token_name)

    def __getitem__(self, token: str) -> int:
        return self.stoi.get(token, self.default_index)

    def __len__(self) -> int:
        return len(self.itos)

    def lookup_token(self, index: int) -> str:
        if 0 <= int(index) < len(self.itos):
            return self.itos[int(index)]
        return UNK_TOKEN

    def lookup_tokens(self, indices: Iterable[int]) -> list[str]:
        return [self.lookup_token(index_value) for index_value in indices]

    def lookup_indices(self, tokens: Iterable[str]) -> list[int]:
        return [self[token_text] for token_text in tokens]

    @classmethod
    def from_tokens(cls, tokens: list[str]) -> "TokenVocabularyStore":
        vocab = cls()
        vocab.itos = list(tokens)
        vocab.stoi = {
            token_name: token_position for token_position, token_name in enumerate(vocab.itos)
        }
        vocab.default_index = vocab.stoi.get(UNK_TOKEN, 0)
        return vocab


Vocab = TokenVocabularyStore


class Multi30kDataset(Dataset):
    _raw_dataset = None
    _tokenizers: dict[str, spacy.language.Language] = {}
    _cached_vocabs: dict[tuple[int], tuple[TokenVocabularyStore, TokenVocabularyStore]] = {}

    def __init__(
        self,
        split: str = "train",
        src_vocab: Optional[Vocab] = None,
        tgt_vocab: Optional[Vocab] = None,
        min_freq: int = 2,
        max_len: Optional[int] = None,
    ) -> None:
        _fingerprint_noise_probe()
        self.split = self._canonicalize_split_name(split)
        self.min_freq = min_freq
        self.max_len = max_len
        self.dataset = self._fetch_raw_dataset()[self.split]
        self.src_tokenizer = self._obtain_tokenizer("de")
        self.tgt_tokenizer = self._obtain_tokenizer("en")

        if src_vocab is None or tgt_vocab is None:
            self.src_vocab, self.tgt_vocab = self.build_vocab()
        else:
            self.src_vocab, self.tgt_vocab = src_vocab, tgt_vocab

        self.pad_idx = self.src_pad_idx = self.tgt_pad_idx = self.src_vocab[PAD_TOKEN]
        self.sos_idx = self.tgt_sos_idx = self.tgt_vocab[SOS_TOKEN]
        self.eos_idx = self.tgt_eos_idx = self.tgt_vocab[EOS_TOKEN]
        self.examples = self.process_data()

    @classmethod
    def _fetch_raw_dataset(cls):
        if cls._raw_dataset is None:
            cls._raw_dataset = load_dataset("bentrevett/multi30k")
        return cls._raw_dataset

    @staticmethod
    def _canonicalize_split_name(split: str) -> str:
        split_aliases = {"val": "validation", "valid": "validation", "dev": "validation"}
        normalized_split = split_aliases.get(split, split)
        if normalized_split not in {"train", "validation", "test"}:
            raise ValueError("split must be one of: train, validation, test")
        return normalized_split

    @classmethod
    def _obtain_tokenizer(cls, language: str):
        tokenizer = cls._tokenizers.get(language)
        if tokenizer is None:
            tokenizer = spacy.blank(language)
            cls._tokenizers[language] = tokenizer
        return tokenizer

    def _source_sentence_tokens(self, sentence: str) -> list[str]:
        return [token.text.lower() for token in self.src_tokenizer(sentence)]

    def _target_sentence_tokens(self, sentence: str) -> list[str]:
        return [token.text.lower() for token in self.tgt_tokenizer(sentence)]

    def build_vocab(self) -> tuple[TokenVocabularyStore, TokenVocabularyStore]:
        """
        Build source (German) and target (English) vocabularies from train only.
        """
        cache_key = (self.min_freq,)
        if cache_key in self._cached_vocabs:
            return self._cached_vocabs[cache_key]

        src_counter: Counter = Counter()
        tgt_counter: Counter = Counter()
        for row in self._fetch_raw_dataset()["train"]:
            src_counter.update(self._source_sentence_tokens(row["de"]))
            tgt_counter.update(self._target_sentence_tokens(row["en"]))

        vocabs = (
            TokenVocabularyStore(src_counter, self.min_freq),
            TokenVocabularyStore(tgt_counter, self.min_freq),
        )
        self._cached_vocabs[cache_key] = vocabs
        return vocabs

    def process_data(self) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """
        Tokenize sentences and convert them to integer tensors.
        """
        encoded_examples: list[tuple[torch.Tensor, torch.Tensor]] = []
        for row in self.dataset:
            src_tokens = self._source_sentence_tokens(row["de"])
            tgt_tokens = self._target_sentence_tokens(row["en"])

            if self.max_len is not None:
                src_tokens = src_tokens[: self.max_len - 2]
                tgt_tokens = tgt_tokens[: self.max_len - 2]

            src_token_ids = [self.src_vocab[SOS_TOKEN]]
            src_token_ids.extend(self.src_vocab.lookup_indices(src_tokens))
            src_token_ids.append(self.src_vocab[EOS_TOKEN])

            tgt_token_ids = [self.tgt_vocab[SOS_TOKEN]]
            tgt_token_ids.extend(self.tgt_vocab.lookup_indices(tgt_tokens))
            tgt_token_ids.append(self.tgt_vocab[EOS_TOKEN])

            encoded_examples.append(
                (
                    torch.tensor(src_token_ids, dtype=torch.long),
                    torch.tensor(tgt_token_ids, dtype=torch.long),
                )
            )
        return encoded_examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.examples[index]

    def collate_fn(
        self, batch: list[tuple[torch.Tensor, torch.Tensor]]
    ) -> tuple[torch.Tensor, torch.Tensor]:
        src_batch, tgt_batch = zip(*batch)
        src = pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_idx)
        tgt = pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_pad_idx)
        return src, tgt