from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectFromModel, SelectKBest
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from boruta import BorutaPy


def create_pipeline(classifier, scaler, dim_reduced, feature_selector, kbest):
    steps = []
    if scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'std':
        steps.append(('scaler', StandardScaler()))

    if dim_reduced:
        steps.append(('dimensionality reduction', PCA(n_components=dim_reduced)))

    if feature_selector == 'rf':
        steps.append(('feature selector', SelectFromModel(RandomForestClassifier())))
    elif feature_selector == 'kbest':
        steps.append(('feature selector', SelectKBest(k=kbest)))

    steps.append(('model', classifier))

    pipe = Pipeline(steps)
    return pipe
