import argparse
from joblib import dump, load

import mlflow
import mlflow.sklearn

import pandas as pd
from .pipeline import create_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()

parser.add_argument(
    '-d', '--dataset',
    default='././data/train.csv',
    type=str,
    help='path to dataset'
)

parser.add_argument(
    '-s', '--save-model-path',
    default='././data/model.joblib',
    type=str,
    help='path to save the model'
)

parser.add_argument(
    '-t', '--test-split-ratio',
    default=0.2,
    type=float,
    help='test split ratio'
)

parser.add_argument(
    '-rs', '--random-state',
    default=13,
    type=int,
    help='random state'
)

parser.add_argument(
    '-sc', '--scaler',
    type=str,
    choices=['minmax', 'std'],
    help='scaler'
)

# parser.add_argument(
#     '-fs', '--selector',
#     type=str,
#     choices=['rf'],
#     help='feature selector'
# )

parser.add_argument(
    '-cr', '--criterion',
    default='gini',
    type=str,
    choices=['gini', 'entropy'],
    help='decision criterion for the tree'
)

parser.add_argument(
    '-n', '--n-estimators',
    default=100,
    type=int,
    help='The number of trees in the forest.'
)

parser.add_argument(
    '-md', '--max-depth',
    default=None,
    type=int,
    help='The maximum depth of the tree.'
)

parser.add_argument(
    '-mf', '--max-features',
    default='auto',
    help='''
    The number of features to consider when looking for the best split:
    If int, then consider max_features features at each split.
    If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
    If “auto”, then max_features=sqrt(n_features).
    If “sqrt”, then max_features=sqrt(n_features) (same as “auto”).
    If “log2”, then max_features=log2(n_features).
    If None, then max_features=n_features.
    Note: the search for a split does not stop until at least one valid partition of the node samples is found, even if it requires to effectively inspect more than max_features features.
    '''
)

parser.add_argument(
    '-bt', '--bootstrap',
    default=True,
    type=bool,
    help='Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.'
)

args = parser.parse_args()

df = pd.read_csv(args.dataset)
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='Cover_Type'), df['Cover_Type'],
                                                    test_size=args.test_split_ratio, random_state=args.random_state)


def train():
    with mlflow.start_run():
        model = RandomForestClassifier(criterion=args.criterion, n_estimators=args.n_estimators,
                                       max_depth=args.max_depth,
                                       max_features=args.max_features, bootstrap=args.bootstrap,
                                       random_state=args.random_state)
        pipeline = create_pipeline(model, args.scaler)
        pipeline.fit(X_train, y_train)
        accuracy = accuracy_score(y_test, pipeline.predict(X_test))
        # mlflow.log_param("use_scaler", use_scaler)
        # mlflow.log_param("max_iter", max_iter)
        # mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        print(f"Accuracy: {accuracy}.")
        dump(pipeline, args.save_model_path)
        print(f"Model is saved to {args.save_model_path}.")
    return
