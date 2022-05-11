import argparse
import mlflow
from .model import nested_cv, kfold_cv
from typing import List, Optional


def train(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset",
        default="././data/train.csv",
        type=str,
        help="path to dataset",
    )

    parser.add_argument(
        "-s",
        "--save-model-path",
        default="././data/model.joblib",
        type=str,
        help="path to save the model",
    )

    parser.add_argument("--target", type=str, help="Target column name")

    parser.add_argument(
        "-rs", "--random-state", default=13, type=int, help="random state"
    )

    parser.add_argument(
        "-m",
        "--model",
        default="rf",
        type=str,
        choices=["rf", "knn"],
        help="model to fit",
    )

    parser.add_argument(
        "-ncv",
        "--nested-cv",
        default=False,
        type=bool,
        help="If True: use automatic hyperparameter optimization via nested cv + grid search",
    )

    parser.add_argument(
        "-cvo",
        "--cross-validation-outer",
        default=5,
        type=int,
        help="Number of CV folds in outer loop",
    )

    parser.add_argument(
        "-cvi",
        "--cross-validation-inner",
        default=None,
        type=int,
        help="Number of CV folds in inner loop",
    )

    # параметры для подготовки фичей
    parser.add_argument(
        "-sc",
        "--scaler",
        default=None,
        type=str,
        choices=["minmax", "std"],
        help="scaler",
    )

    parser.add_argument(
        "-dr", "--dim-reduced", default=None, type=int, help="Reduced dimensions"
    )

    parser.add_argument(
        "-fs",
        "--feature-selector",
        default=None,
        type=str,
        choices=["rf", "kbest"],
        help="feature selector",
    )

    parser.add_argument("--kbest", default=None, type=int, help="k for KBest selector")

    # параметры для спецификации модели
    parser.add_argument(
        "-n",
        default=None,
        type=int,
        help="Number of estimators for Random Forest / Number of neighbours for KNN",
    )

    parser.add_argument(
        "-cr",
        "--criterion",
        default=None,
        type=str,
        choices=["gini", "entropy"],
        help="decision criterion for the tree",
    )

    parser.add_argument(
        "-md",
        "--max-depth",
        default=None,
        type=int,
        help="The maximum depth of the tree.",
    )

    parser.add_argument(
        "-mf",
        "--max-features",
        default=None,
        type=float,
        help="Fraction of features to consider when looking for the best split",
    )

    parser.add_argument(
        "-bt",
        "--bootstrap",
        default=None,
        type=bool,
        help="""Whether bootstrap samples are used when building trees.
        If False, the whole dataset is used to build each tree.""",
    )

    parser.add_argument(
        "-w",
        "--weights",
        default=None,
        type=str,
        choices=["uniform", "distance"],
        help="Weight function used in KNN prediction",
    )

    args = parser.parse_args()

    with mlflow.start_run():
        if args.nested_cv:
            nested_cv(args)
        else:
            kfold_cv(args)
    return
