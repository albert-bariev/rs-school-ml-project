from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectFromModel, RFECV, SequentialFeatureSelector
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


def create_pipeline(classifier, scaler, dim_reduced, feature_selector):
    steps = []
    if scaler == 'minmax':
        steps.append(('scaler', MinMaxScaler()))
    elif scaler == 'std':
        steps.append(('scaler', StandardScaler()))

    if dim_reduced:
        steps.append(('dimensionality reduction', PCA(n_components=dim_reduced)))

    if feature_selector == 'rf':    #TODO добавить разные селекторы
        steps.append(('feature selector', SelectFromModel(RandomForestClassifier())))

    steps.append(('model', classifier))

    pipe = Pipeline(steps)
    return pipe
