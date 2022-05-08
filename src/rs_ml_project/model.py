import pandas as pd
import numpy as np
import mlflow
from joblib import dump
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from .pipeline import create_pipeline


def get_data(args):
    df = pd.read_csv(args.dataset)
    return df.drop(columns=args.target).to_numpy(), df[args.target]


def kfold_cv(args):
    X, y = get_data(args)
    params, metrics = {}, {}
    if args.model == 'knn':
        params['n_neighbors'] = args.n if args.n else 5
        params['weights'] = args.weights if args.weights else 'distance'
        model = KNeighborsClassifier(**params)
    else:
        params['n_estimators'] = args.n if args.n else 100
        params['criterion'] = args.criterion if args.criterion else 'gini'
        params['max_depth'] = args.max_depth
        params['max_features'] = args.max_features if args.max_features else 'auto'
        params['bootstrap'] = args.bootstrap or True
        params['random_state'] = args.random_state
        model = RandomForestClassifier(**params)
    params['0_model'] = args.model
    params['random_state'] = args.random_state
    params['1_nested_cv'] = args.nested_cv
    params['1_cv_outer'] = args.cross_validation_outer
    params['1_cv_inner'] = args.cross_validation_inner
    params['2_scaler'] = args.scaler
    params['2_dim_reduced'] = args.dim_reduced
    params['2_feature_selector'] = args.feature_selector
    params['2_kbest'] = args.kbest

    mlflow.log_params(params)

    pipeline = create_pipeline(model, args.scaler, args.dim_reduced, args.feature_selector, args.kbest)
    pipeline.fit(X, y)
    dump(pipeline, args.save_model_path)
    print(f"Model is saved to {args.save_model_path}.")

    scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    scores = cross_validate(pipeline, X, y, cv=KFold(n_splits=args.cross_validation_outer, shuffle=True,
                                                     random_state=args.random_state),
                            scoring=scoring)
    metrics['accuracy'] = scores['test_accuracy'].mean()
    metrics['precision'] = scores['test_precision_weighted'].mean()
    metrics['recall'] = scores['test_recall_weighted'].mean()
    metrics['f1_score'] = scores['test_f1_weighted'].mean()
    mlflow.log_metrics(metrics)
    print(metrics)
    return


def nested_cv(args):
    X, y = get_data(args)
    space = {
        'knn': {
            'model__n_neighbors': [1, 2, 5, 10, 20, 50],
            'model__weights': ['uniform', 'distance']
        },
        'rf': {
            'model__n_estimators': [10, 20, 50, 100, 300],
            'model__criterion': ['gini', 'entropy'],
            'model__max_depth': [3, 5, 7, 10, 15],
            'model__max_features': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            'model__bootstrap': [True, False]
        }
    }
    params, metrics, best_params = {}, {}, {}
    params['0_model'] = args.model
    params['random_state'] = args.random_state
    params['1_nested_cv'] = args.nested_cv
    params['1_cv_outer'] = args.cross_validation_outer
    params['1_cv_inner'] = args.cross_validation_inner if args.cross_validation_inner else 3
    params['2_scaler'] = args.scaler
    params['2_dim_reduced'] = args.dim_reduced
    params['2_feature_selector'] = args.feature_selector
    params['2_kbest'] = args.kbest

    cv_outer = KFold(n_splits=params['1_cv_outer'], shuffle=True, random_state=params['random_state'])
    accuracy_lst, precision_lst, recall_lst, f1_lst = [], [], [], []
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]
        cv_inner = KFold(n_splits=params['1_cv_inner'], shuffle=True, random_state=params['random_state'])

        if params['0_model'] == 'knn':
            model = KNeighborsClassifier()
        else:
            model = RandomForestClassifier(random_state=params['random_state'])
        pipeline = create_pipeline(model, params['2_scaler'], params['2_dim_reduced'], params['2_feature_selector'],
                                   params['2_kbest'])
        search = GridSearchCV(pipeline, space[params['0_model']], scoring='accuracy', cv=cv_inner, refit=True,
                              n_jobs=-1)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        y_pred = best_model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)

        if not accuracy_lst or acc > max(accuracy_lst):
            best_params = result.best_params_

        accuracy_lst.append(acc)
        precision_lst.append(precision_score(y_test, y_pred, average='weighted'))
        recall_lst.append(recall_score(y_test, y_pred, average='weighted'))
        f1_lst.append(f1_score(y_test, y_pred, average='weighted'))

    for key, value in best_params.items():
        p = key[7:]
        params[p] = value

    mlflow.log_params(params)

    metrics['accuracy'] = np.mean(accuracy_lst)
    metrics['precision'] = np.mean(precision_lst)
    metrics['recall'] = np.mean(recall_lst)
    metrics['f1_score'] = np.mean(f1_lst)
    mlflow.log_metrics(metrics)
    print(metrics)
    return
