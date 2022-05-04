import argparse
from joblib import dump, load

import mlflow

import pandas as pd
from .pipeline import create_pipeline
from sklearn.model_selection import train_test_split, cross_validate, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

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
    '--target',
    type=str,
    help='Target column name'
)

parser.add_argument(
    '-rs', '--random-state',
    default=13,
    type=int,
    help='random state'
)

parser.add_argument(
    '-sc', '--scaler',
    default=None,
    type=str,
    choices=['minmax', 'std'],
    help='scaler'
)

parser.add_argument(
    '-dr', '--dim-reduced',
    default=None,
    type=int,
    help='Reduced dimensions'
)

parser.add_argument(
    '-cv', '--cross-validation',
    default=5,
    type=int,
    help='Number of CV folds'
)

parser.add_argument(
    '-fs', '--feature-selector',
    default=None,
    type=str,
    choices=['rf', 'kbest'],
    help='feature selector'
)

parser.add_argument(
    '--kbest',
    default=None,
    type=int,
    help='k for KBest selector'
)

parser.add_argument(
    '-m', '--model',
    default='rf',
    type=str,
    choices=['rf', 'knn'],
    help='model to fit'
)

parser.add_argument(
    '-n',
    default=100,
    type=int,
    help='Number of estimators for Random Forest / Number of neighbours for KNN'
)

parser.add_argument(
    '-cr', '--criterion',
    default='gini',
    type=str,
    choices=['gini', 'entropy'],
    help='decision criterion for the tree'
)

parser.add_argument(
    '-md', '--max-depth',
    default=None,
    type=int,
    help='The maximum depth of the tree.'
)

parser.add_argument(
    '-mf', '--max-features',
    type=int,
    help='The number of features to consider when looking for the best split'
)

parser.add_argument(
    '-bt', '--bootstrap',
    default=True,
    type=bool,
    help='Whether bootstrap samples are used when building trees. If False, the whole dataset is used to build each tree.'
)

parser.add_argument(
    '-w', '--weights',
    default='distance',
    type=str,
    choices=['uniform', 'distance'],
    help='Weight function used in KNN prediction'
)

args = parser.parse_args()

df = pd.read_csv(args.dataset)
X, y = df.drop(columns=args.target), df[args.target]

def train():
    with mlflow.start_run():
        if args.model == 'knn':
            model = KNeighborsClassifier(n_neighbors=args.n, weights=args.weights)
        else:
            max_features = args.max_features if args.max_features else 'auto'
            model = RandomForestClassifier(n_estimators=args.n, criterion=args.criterion,
                                           max_depth=args.max_depth,
                                           max_features=max_features, bootstrap=args.bootstrap,
                                           random_state=args.random_state)
        pipeline = create_pipeline(model, args.scaler, args.dim_reduced, args.feature_selector, args.kbest)
        pipeline.fit(X, y)
        dump(pipeline, args.save_model_path)
        print(f"Model is saved to {args.save_model_path}.")

        mlflow.log_param("model", args.model)
        mlflow.log_param("scaler", args.scaler)
        mlflow.log_param("random state", args.random_state)
        mlflow.log_param("dim-reduced", args.dim_reduced)
        mlflow.log_param("cross-validation", args.cross_validation)
        mlflow.log_param("feature-selector", args.feature_selector)
        mlflow.log_param("kbest", args.kbest)

        if args.model == 'knn':
            mlflow.log_param('n_neighbors', args.n)
            mlflow.log_param('weights', args.weights)
        else:
            mlflow.log_param('n_estimators', args.n)
            mlflow.log_param('criterion', args.criterion)
            mlflow.log_param('max_depth', args.max_depth)
            mlflow.log_param('max_features', max_features)
            mlflow.log_param('bootstrap', args.bootstrap)

        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
        scores = cross_validate(pipeline, X, y, cv=args.cross_validation, scoring=scoring)
        accuracy = scores['test_accuracy'].mean()
        precision = scores['test_precision_weighted'].mean()
        recall = scores['test_recall_weighted'].mean()
        f1 = scores['test_f1_weighted'].mean()

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

        print(f"Accuracy: {accuracy}.")
        print(f"Precision: {precision}.")
        print(f"Recall: {recall}.")
        print(f"F1-score: {f1}.")
    return
