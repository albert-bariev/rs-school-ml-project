import argparse
from joblib import dump, load

import mlflow
import mlflow.sklearn

import pandas as pd
from .pipeline import create_pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

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
X_train, X_test, y_train, y_test = train_test_split(df.drop(columns='Cover_Type'), df['Cover_Type'],
                                                    test_size=args.test_split_ratio, random_state=args.random_state)


def train():
    with mlflow.start_run():
        if args.model == 'knn':
            model = KNeighborsClassifier(n_neighbors=args.n, weights=args.weights)
        else:
            max_features = args.max_features if args.max_features else 'auto'
            model = RandomForestClassifier(criterion=args.criterion, n_estimators=args.n_estimators,
                                           max_depth=args.max_depth,
                                           max_features=max_features, bootstrap=args.bootstrap,
                                           random_state=args.random_state)
        pipeline = create_pipeline(model, args.scaler)
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)  # TODO добавить CV
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        # mlflow.log_param("use_scaler", use_scaler)
        # mlflow.log_param("max_iter", max_iter)
        # mlflow.log_param("logreg_c", logreg_c)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"Accuracy: {accuracy}.")
        print(f"Precision: {precision}.")
        print(f"Recall: {recall}.")
        print(f"F1-score: {f1}.")
        dump(pipeline, args.save_model_path)
        print(f"Model is saved to {args.save_model_path}.")
    return
