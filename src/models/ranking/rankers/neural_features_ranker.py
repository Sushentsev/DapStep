from typing import Optional, List

import torch
from torch.nn.utils.rnn import pad_sequence

from src.models.base import NeuralModel
from src.models.convert import get_stack_values, get_stack_features
from src.models.embeddings.embeddings import StackNeuralEmbedding, AnnotationNeuralEmbedding
from src.models.ranking.similarities.base import SimModule
from src.preprocess.token import Token
from src.utils import device


class NeuralFeaturesRanker(NeuralModel):
    def __init__(self, stack_emb_module: StackNeuralEmbedding, fixer_emb_module: StackNeuralEmbedding,
                 annotation_emb_module: AnnotationNeuralEmbedding, sim_module: SimModule):
        super(NeuralFeaturesRanker, self).__init__()
        self._stack_emb_module = stack_emb_module
        self._fixer_emb_module = fixer_emb_module
        self._annotation_emb_module = annotation_emb_module
        self._sim_module = sim_module

    def raw_stack_emb(self, stack: List[Token[int]]) -> torch.Tensor:
        stack_len = torch.tensor([len(stack)], dtype=torch.long).to(device)
        stack = torch.tensor([get_stack_values(stack)], dtype=torch.long).to(device)
        return self._stack_emb_module.emb(stack, stack_len)[0]

    def raw_fixers_emb(self, fixers_stacks: List[List[Token[int]]]) -> torch.Tensor:
        stacks_lens = torch.tensor([len(stack) for stack in fixers_stacks], dtype=torch.long).to(device)
        stacks = pad_sequence([torch.tensor(get_stack_values(stack), dtype=torch.long)
                               for stack in fixers_stacks], batch_first=True, padding_value=0).to(device)

        embs = []
        for stack in fixers_stacks:
            annotations_lens = torch.tensor([len(token.data.annotations) for token in stack],
                                            dtype=torch.long).to(device)
            annotations = pad_sequence([torch.tensor(token.data.annotations, dtype=torch.float)
                                        for token in stack], batch_first=True, padding_value=0).to(device)

            annotations_emb = self._annotation_emb_module.emb(annotations,
                                                              annotations_lens)  # -> (num_annotations, emb_dim)

            features = torch.tensor(get_stack_features(stack), dtype=torch.float).to(device)
            embs.append(torch.cat((annotations_emb, features), dim=-1))

        return self._fixer_emb_module.emb(stacks, stacks_lens, pad_sequence(embs, batch_first=True, padding_value=0))

    def raw_sim(self, stack_emb: torch.Tensor, fixers_emb: torch.Tensor,
                overall_features: Optional[List[List[float]]] = None) -> torch.Tensor:
        if overall_features:
            overall_features = torch.tensor(overall_features, dtype=torch.float).to(device)
        return self._sim_module.sim(stack_emb, fixers_emb, overall_features)

    def raw_predict(self, stack: List[Token[int]], fixers_stacks: List[List[Token[int]]],
                    overall_features: Optional[List[List[float]]] = None) -> torch.Tensor:
        stack_emb = self.raw_stack_emb(stack)
        fixers_emb = self.raw_fixers_emb(fixers_stacks)
        return self.raw_sim(stack_emb, fixers_emb, overall_features)
