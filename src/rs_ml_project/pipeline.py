from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier


def create_pipeline(classifier, scaler=None, feature_selector=None):
    steps = []
    if scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'std':
        steps.append(('scaler', StandardScaler()))

    if feature_selector == 'rf':    #TODO добавить разные селекторы
        steps.append(('feature selector', SelectFromModel(RandomForestClassifier())))

    steps.append(('model', classifier))

    pipe = Pipeline(steps)
    return pipe
