from typing import List, Set, Dict

from tqdm import tqdm

from src.data.objects.stack import Stack
from src.features.frame_features_builder import FrameFeaturesBuilder
from src.features.overall_features_builder import OverallFeaturesBuilder
from src.preprocess.seq_coder import SeqCoder
from src.preprocess.token import Token
from src.stack_builders.builders import UserStackBuilder

CodedStack = List[int]
CodedStackTokens = List[Token[int]]


def get_non_empty_users(user2stack: Dict[int, List[Token[int]]]) -> Set[int]:
    return {user_id for user_id in user2stack if len(user2stack[user_id]) > 0}


def fixers_stacks_ctor(stacks: List[Stack], y: List[int], user_ids: Set[int],
                       user_stack_builder: UserStackBuilder, frame_features_builder: FrameFeaturesBuilder,
                       seq_coder: SeqCoder, overall_features_builder: OverallFeaturesBuilder):
    filtered_stacks, assignees_stacks, overall_features, y_filtered = [], [], [], []

    for stack, label in tqdm(list(zip(stacks, y))):
        user2stack = user_stack_builder(stack, user_ids)
        frame_features_builder.build(stack, user2stack)

        stack_coded = seq_coder.transform(stack)
        user2stack_coded = {user_id: seq_coder.transform(user2stack[user_id]) for user_id in user_ids}

        non_empty_users = get_non_empty_users(user2stack_coded)

        if len(non_empty_users) > 1 and label in non_empty_users:
            non_empty_users_list = list(non_empty_users)
            filtered_stacks.append(stack_coded)
            assignees_stacks.append([user2stack_coded[user_id] for user_id in non_empty_users_list])
            user2overall_features = overall_features_builder(stack, user_ids)
            overall_features.append([user2overall_features[user_id] for user_id in non_empty_users_list])
            y_filtered.append(non_empty_users_list.index(label))

    return filtered_stacks, assignees_stacks, overall_features, y_filtered
