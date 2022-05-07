import pandas as pd
import mlflow
from joblib import dump
from sklearn.model_selection import cross_validate, KFold, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from .pipeline import create_pipeline


def get_data(args):
    df = pd.read_csv(args.dataset)
    return df.drop(columns=args.target), df[args.target]


def kfold_cv(args):
    X, y = get_data(args)
    params = {}
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


def nested_cv(args):
    X, y = get_data(args)
    space = {
        'knn': {
            'n_neighbors': [1, 2, 5, 10, 20, 50],
            'weights': ['uniform', 'distance']
        },
        'rf': {
            'n_estimators': [10, 20, 50, 100, 300],
            'criterion': ['gini', 'entropy'],
            'max_depth': [3, 5, 7, 10, 15],
            'max_features': [0.1, 0.2, 0.3, 0.5, 0.7, 1.0],
            'bootstrap': [True, False]
        }
    }
    cv_outer = KFold(n_splits=args.cross_validation_outer, shuffle=True, random_state=args.random_state)
    outer_results = []
    for train_ix, test_ix in cv_outer.split(X):
        X_train, X_test = X[train_ix, :], X[test_ix, :]
        y_train, y_test = y[train_ix], y[test_ix]

        cv_inner = KFold(n_splits=args.cross_validation_inner, shuffle=True, random_state=args.random_state)

        if args.model == 'knn':
            model = KNeighborsClassifier()
        else:
            model = RandomForestClassifier(random_state=args.random_state)

        search = GridSearchCV(model, space[args.model], scoring='accuracy', cv=cv_inner, refit=True)
        result = search.fit(X_train, y_train)
        best_model = result.best_estimator_
        yhat = best_model.predict(X_test)
        acc = accuracy_score(y_test, yhat)
        outer_results.append(acc)
        print('>acc=%.3f, est=%.3f, cfg=%s' % (acc, result.best_score_, result.best_params_))
    print('Accuracy: %.3f (%.3f)' % (mean(outer_results), std(outer_results)))
