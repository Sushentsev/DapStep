from typing import Tuple, List, Any, Set


def remove_unseen(seen: Set[Any], X: List[Any], y: List[Any]) -> Tuple[List[Any], List[Any]]:
    X_new, y_new = [], []

    for x_, y_ in zip(X, y):
        if y_ in seen:
            X_new.append(x_)
            y_new.append(y_)

    return X_new, y_new


def train_test_split(X: List[Any], y: List[Any], test_size: int, include_unseen: int = True) \
        -> Tuple[List[Any], List[Any], List[Any], List[Any]]:
    if test_size == 0:
        return X, [], y, []

    X_train, X_test = X[:-test_size], X[-test_size:]
    y_train, y_test = y[:-test_size], y[-test_size:]

    if not include_unseen:
        X_test, y_test = remove_unseen(set(y_train), X_test, y_test)

    return X_train, X_test, y_train, y_test
