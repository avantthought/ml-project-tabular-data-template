"""Place modeling pipeline function(s)."""

from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, FunctionTransformer

from src.modeling.process import drop_fieds, FeaturesToDict


def create_pipeline(model, fields_to_drop):
    """
    Builds a sklearn modeling pipeline.
    
    :param sklearn.base.BaseEstimator model: instantiated model to be placed at the end of the pipeline
    :param List[str] fields_to_drop: list of column names to drop from input data
    :return: modeling pipeline
    :rtype: sklearn.pipeline.Pipeline
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ('dict_creator', FeaturesToDict()),
        ('dict_vectorizer', DictVectorizer(sparse=False))
    ])
    preprocessor = ColumnTransformer(transformers=[
        ('numeric_tranformer', numeric_transformer, selector(dtype_exclude=['category', 'object'])),
        ('categorical_transformer', categorical_transformer, selector(dtype_include=['category', 'object']))
    ])
    pipeline = Pipeline(steps=[
        ('column_dropper', FunctionTransformer(drop_fieds, validate=False, kw_args={'fields': fields_to_drop})),
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    return pipeline
