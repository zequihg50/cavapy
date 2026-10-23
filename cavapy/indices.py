import numpy as np
import xarray

from scipy.stats import linregress

# General utilities

def count_events(arr, n):
    arr = np.asarray(arr, dtype=bool)
    padded = np.r_[False, arr, False]
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return np.sum((ends - starts) >= n)

def count_longest_event(arr):
    arr = np.asarray(arr, dtype=bool)
    if not arr.any():
        return 0

    padded = np.r_[False, arr, False]
    changes = np.diff(padded.astype(int))

    starts = np.where(changes == 1)[0]
    ends = np.where(changes == -1)[0]

    return np.max(ends - starts)

def calc_trend(y, years):
    valid = np.isfinite(y) & np.isfinite(years)

    if valid.sum() < 3:
        return np.nan, np.nan

    result = linregress(years[valid], y[valid])
    return result.slope, result.pvalue

def mean_and_trend(data, years):
    annual_mean = data.mean("time")
    trend, pvalue = xarray.apply_ufunc(
        calc_trend,
        data, years,
        input_core_dims=[["time"], ["time"]],
        output_core_dims=[[], []],
        vectorize=True,
        output_dtypes=[float, float],
    )

    return annual_mean, trend, pvalue

# Indices

def prcptot(da):
    return da.resample({"time": "YE"}).sum()

def dd(da):
    dd = (da <= 1).resample({"time": "YE"}).sum()
    dd.attrs["units"] = "N° days"

    return dd

def cdd(da):
    cdd = xarray.apply_ufunc(
        count_longest_event,
        (da < 1).groupby("time.year"),
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[int]).rename({"year": "time"})
    cdd.attrs["units"] = "N° days"

    return cdd

def cwd(da):
    cwd = xarray.apply_ufunc(
        count_longest_event,
        (da > 1).groupby("time.year"),
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[int]).rename({"year": "time"})
    cwd.attrs["units"] = "N° days"

    return cwd

def r20(da):
    r20 = (da > 20).resample({"time": "YE"}).sum()
    r20.attrs["units"] = "N° days"

    return r20

def su(da):
    su = (da > 25).resample({"time": "YE"}).sum()
    su.attrs["units"] = "N° days"

    return su

def tr(da):
    tr = (da > 20).resample({"time": "YE"}).sum()
    tr.attrs["units"] = "N° days"

    return tr

def hwvs(da):
    hwvs = xarray.apply_ufunc(
        lambda x: count_events(x, 3),
        (da > 35).groupby("time.year"),
        input_core_dims=[["time"]],
        output_core_dims=[[]],
        vectorize=True,
        output_dtypes=[int]).rename({"year": "time"})
    hwvs.attrs["units"] = "N° days"

    return hwvs
