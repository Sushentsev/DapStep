from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier

from src.models.base import NeuralModel
from src.models.classification.neural_classifiers import NeuralClassifier, CNNNeuralClassifier, \
    RNNNeuralClassifier
from src.models.embeddings.embeddings import StackNeuralEmbedding, RNNStackNeuralEmbedding, \
    CNNStackNeuralEmbedding, AnnotationNeuralEmbedding
from src.models.ranking.rankers.manual_features_ranker import ManualFeaturesRanker
from src.models.ranking.rankers.neural_features_ranker import NeuralFeaturesRanker
from src.models.ranking.similarities.mlp_similarity import MLPSimilarity
from src.preprocess.entry_coders import Stack2FrameNames, Stack2FrameFileNames, Stack2FrameSubsystems
from src.preprocess.seq_coder import SeqCoder
from src.preprocess.tokenizers import FrameTokenizer


def seq_coder_ctor(entry_coder_type: str, cased: bool, trim_len: int, rem_equals: bool) -> SeqCoder:
    if entry_coder_type == "name":
        entry_coder = Stack2FrameNames(cased, trim_len, rem_equals)
    elif entry_coder_type == "file_name":
        entry_coder = Stack2FrameFileNames(cased, trim_len, rem_equals)
    elif entry_coder_type == "subsystem":
        entry_coder = Stack2FrameSubsystems(cased, trim_len, rem_equals)
    else:
        raise ValueError("Wrong entry coder type!")

    return SeqCoder(entry_coder, FrameTokenizer())


def embeddings_modules_ctor(vocab_size: int, frame_features_dim: int,
                            emb_type: str) -> Tuple[StackNeuralEmbedding, StackNeuralEmbedding]:
    if emb_type == "rnn":
        stack_emb_module = RNNStackNeuralEmbedding(vocab_size, embedding_dim=70, hidden_size=100)
        fixer_emb_module = RNNStackNeuralEmbedding(vocab_size, frame_features_dim, embedding_dim=70, hidden_size=100)
    elif emb_type == "cnn":
        stack_emb_module = CNNStackNeuralEmbedding(64, [3, 4, 5], vocab_size, embedding_dim=70)
        fixer_emb_module = CNNStackNeuralEmbedding(64, [3, 4, 5], vocab_size, frame_features_dim, embedding_dim=70)
    else:
        raise ValueError("Wrong stack embedding type.")

    return stack_emb_module, fixer_emb_module


def ranking_model_ctor(vocab_size: int, frame_features_dim: int, overall_features_dim: int,
                       emb_type: str, manual_features: bool = True) -> NeuralModel:
    if not manual_features:
        annotation_emb_module = AnnotationNeuralEmbedding(2, 15)
        frame_features_dim += annotation_emb_module.dim

    stack_emb_module, fixer_emb_module = embeddings_modules_ctor(vocab_size, frame_features_dim, emb_type)
    sim_module = MLPSimilarity(stack_emb_module.dim, fixer_emb_module.dim, overall_features_dim)

    if manual_features:
        return ManualFeaturesRanker(stack_emb_module, fixer_emb_module, sim_module)
    else:
        return NeuralFeaturesRanker(stack_emb_module, fixer_emb_module, annotation_emb_module, sim_module)


def classification_model_ctor(vocab_size: int, num_classes: int, model_type: str) -> NeuralClassifier:
    if model_type == "cnn":
        return CNNNeuralClassifier(num_classes, vocab_size, 64, [3, 4, 5], 70)
        # return CNNNeuralClassifier(num_classes, vocab_size, 32, [3, 4, 5], 50)
    elif model_type == "rnn":
        return RNNNeuralClassifier(num_classes, vocab_size, 70, 100)
        # return RNNNeuralClassifier(num_classes, vocab_size, 50, 70)
    else:
        raise ValueError("Wrong model type.")


def sklearn_model_ctor(model_type: str):
    if model_type == "SGDClassifier":
        return SGDClassifier(loss="log", alpha=1e-5)
    elif model_type == "RandomForest":
        return RandomForestClassifier(max_depth=None, min_samples_leaf=1, n_estimators=100)
    else:
        raise ValueError("Wrong model type.")