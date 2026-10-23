"""Keep speaker groups separate and never truncate a paired training sentence."""
import random
import torch
from fish_speech.datasets.semantic import AutoTextSemanticInstructionIterableDataset
import fish_speech.datasets.semantic as semantic
from fish_speech.text.clean import clean_text
from balance import probabilities


class BalancedDataModule(semantic.SemanticDataModule):
    """Support the single-process loader without changing the pinned upstream."""

    def _loader(self, dataset):
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            # Resolve at call time: the shared GPU pilot replaces this collator
            # with a full-context padding collator to exercise memory use.
            collate_fn=semantic.TextDataCollator(self.tokenizer, self.max_length),
            num_workers=self.num_workers,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset)

    def val_dataloader(self):
        return self._loader(self.val_dataset)


class BalancedDataset(AutoTextSemanticInstructionIterableDataset):
    def __init__(self, family_weights, **kwargs):
        super().__init__(**kwargs)
        self.family_weights = dict(family_weights)

    def init_mock_data_server(self):
        if self.groups is not None:
            return
        super().init_mock_data_server()
        # group sentence counts otherwise let Talha overwhelm the other sources.
        seconds = [sum(len(s.semantics[0].values)/21.5 for s in g.sentences) for g in self.groups]
        self.group_weights = probabilities([g.name for g in self.groups], seconds, self.family_weights)
        print("Balanced family weights:", self.family_weights, "speaker groups:", len(self.groups), flush=True)

    def augment(self):
        if self.groups is None:
            self.init_mock_data_server()
        group = random.choices(self.groups, weights=self.group_weights, k=1)[0]
        # Bound CPU work as well as GPU sequence size. Whole clips from one speaker.
        tokens, labels = [], []
        length = 0
        for _ in range(8):
            sentence = random.choice(group.sentences)
            t, l = self.pack_sentences([clean_text(random.choice(sentence.texts))],
                                       [sentence.semantics], skip_text=False)
            if t.shape[1] > self.max_length:
                raise ValueError(f"A clip from {group.name} exceeds context; reduce max clip length")
            if length + t.shape[1] > self.max_length:
                break
            tokens.append(t)
            labels.append(l)
            length += t.shape[1]
        return {"tokens": torch.cat(tokens, dim=1), "labels": torch.cat(labels, dim=1)}
