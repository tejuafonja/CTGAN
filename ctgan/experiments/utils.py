import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def histogram_intersection(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=True,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.
    Calculates the amount of overlap between two histograms.
    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "histogram_intersection"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    hist_real, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    hist_fake, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )

    result = np.minimum(hist_real, hist_fake).sum()

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result

def histogram_binning(data, bins=50, categorial=False):
    """Compute the histogram of `data`.
        This function first calculates a
        monotonically increasing array of bin edges,
        if `type(bins)==int`, then computes the histogram of `data`.
    Args:
        data (array_like):
            Data to evaluate
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        categorial (bool, optional):
            Whether or not `data` is categorial. Defaults to `False`.
            If `categorial=False`, and `type(bins) == int`, `data` range should be between 0 and 1.
    Returns:
        hist (array):
            The normalized count of samples in each bin.
        bin_edges (array):
            Bin edges (length(hist)+1)
    """

    if type(bins) == int:

        if categorial:
            bin_edges = np.arange(0, bins + 1)
        else:
            bin_edges = np.linspace(0, 1, bins)
    else:
        bin_edges = bins

    hist, bin_edges = np.histogram(data, bins=bin_edges)

    if sum(hist) == 0:
        hist = np.zeros_like(hist)

    else:
        hist = hist / sum(hist)

    return hist, bin_edges

def column_transformer(real_col, fake_col, column_name=None, kind="minmax", fit_data=None):
    """Column Transformer
    Args:
        real_col (array-like):
            Real data column to evaluate
        fake_col (array-like):
            Fake data column to evaluate
        kind (str, optional):
            Kind of transformer. Defaults to "minmax".
            Must be one of (`minmax`, `label`, `onehot`)
        fit_data (array-like, optional):
            Data to fit the column transformer on. Defaults to `None`. 
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
    Returns:
        real_col (array-like):
            Transformed real data column
        fake_col (array-like):
            Transformed fake data column
        transformer (array-like):
            Transformer object
    """
    
    if kind == "label":
        transformer = LabelEncoder()
        
        real_col = real_col.astype("str")
        fake_col = fake_col.astype("str")
        
        if fit_data is not None:
            fit_data = fit_data.astype("str")
            transformer.fit(fit_data)
        else:
            # fit encoder on concatenated real and fake data
            real_fake = np.concatenate([real_col, fake_col])
            transformer.fit(real_fake)
        
        # encode real and fake data
        real_col = transformer.transform(real_col)
        fake_col = transformer.transform(fake_col)
        
        if column_name is not None:
            real_col = pd.DataFrame(real_col, columns=[column_name])
            fake_col = pd.DataFrame(fake_col, columns=[column_name])

    elif kind == "onehot":
        transformer = OneHotEncoder(sparse=False)
        
        if fit_data is not None:
            transformer.fit(fit_data[:, None])
        else:
            # fit encoder on concatenated real and fake data
            real_fake = np.concatenate([real_col, fake_col])
            transformer.fit(real_fake[:, None])

        # encode real and fake data
        real_col = transformer.transform(real_col[:, None])
        fake_col = transformer.transform(fake_col[:, None])
        
        if column_name is not None:
            columns = [f"{column_name}_{i.strip()}" for i in transformer.categories_[0]]
            real_col = pd.DataFrame(real_col, columns=columns)
            fake_col = pd.DataFrame(fake_col, columns=columns)

    elif kind == "minmax":
        transformer = MinMaxScaler()
        
        if fit_data is not None:
            transformer.fit(fit_data[:, None])
        else:
            # compute statistics on real data
            transformer.fit(real_col[:, None])

        # rescale real and fake data
        real_col = transformer.transform(real_col[:, None]).squeeze()
        fake_col = transformer.transform(fake_col[:, None]).squeeze()
        
        if column_name is not None:
            real_col = pd.DataFrame(real_col, columns=[column_name])
            fake_col = pd.DataFrame(fake_col, columns=[column_name])
            
    else:
        transformer = None
        if column_name is not None:
            real_col = pd.DataFrame(real_col, columns=[column_name])
            fake_col = pd.DataFrame(fake_col, columns=[column_name])
        
    return real_col, fake_col, transformer


def data_transformer(realdata, fakedata):
    column_type = {}
    
    realdata = realdata.reset_index(drop=True)
    fakedata = fakedata.reset_index(drop=True)
    
    for col in realdata:
        field_type = realdata[col].dropna().infer_objects().dtype.kind
        if field_type == "O":
            n_categories = set(realdata[col])
            if len(n_categories) == 2:
                field_type = "b"
        column_type[col] = field_type
     
    sorted_real_columns = sorted(realdata.columns.tolist())
    sorted_fake_columns = sorted(fakedata.columns.tolist())

    assert sorted_real_columns == sorted_fake_columns

    realdata = realdata[sorted_real_columns]
    fakedata = fakedata[sorted_fake_columns]

    real_iter = realdata.iteritems()
    fake_iter = fakedata.iteritems()
    
    def _kind(field_type):
        if field_type == "O":
            return "onehot"
        elif field_type == "b":
            return "label"
        else: 
            return None
        
    results = joblib.Parallel(prefer="threads", n_jobs=-1)(
        joblib.delayed(column_transformer)(
            real_col=real_col,
            fake_col=fake_col,
            column_name=column_name,
            kind=_kind(column_type[column_name]),
            )
        for (column_name, real_col), (_, fake_col) in zip(real_iter, fake_iter)
        )
    
    transformed_realdata = []
    transformed_fakedata = []
    transformers = []
    
    for tupl in results:
        transformed_realdata.append(tupl[0])
        transformed_fakedata.append(tupl[1])
        transformers.append(tupl[2])
        
   
    transformed_realdata = pd.concat(transformed_realdata, axis=1)
    transformed_fakedata = pd.concat(transformed_fakedata, axis=1)

    return transformed_realdata, transformed_fakedata, transformers
            

def column_metric_wrapper(
    realdata, fakedata, column_metric, cat_cols=None, random_state=1000
):
    """Column Metric Wrapper
    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        column_metric (func):
            Column metric to apply
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
            If `None`, infers categorical columns from `realdata`
        random_state (int, optional):
            Random state number for reproducibility.
            Defaults to `1000`.
    Returns:
        result_df (pd.DataFrame):
            Result of `metric` applied on each column.
    """

    sorted_real_columns = sorted(realdata.columns.tolist())
    sorted_fake_columns = sorted(fakedata.columns.tolist())

    assert sorted_real_columns == sorted_fake_columns

    realdata = realdata[sorted_real_columns]
    fakedata = fakedata[sorted_fake_columns]

    real_iter = realdata.iteritems()
    fake_iter = fakedata.iteritems()

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns

    results = joblib.Parallel(prefer="threads", n_jobs=-1)(
        joblib.delayed(column_metric)(
            realdata=real_col.to_frame(),
            fakedata=fake_col.to_frame(),
            column_name=column_name,
            categorial=True if column_name in cat_cols else False,
            random_state=random_state,
            return_dataframe=True,
        )
        for (column_name, real_col), (_, fake_col) in zip(real_iter, fake_iter)
    )
    result_df = pd.concat(results)

    return result_df.reset_index(drop=True)