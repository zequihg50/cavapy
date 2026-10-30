import pandas as pd
import xarray

def load_era5(ds, vs, time_lim, lat_lim, lon_lim, workers=4, cs=7500):
    min_lon, max_lon = lon_lim
    min_lat, max_lat = lat_lim
    min_time, max_time = time_lim

    if (min_lon < 0) and (max_lon < 0):
        era5 = ds[vs].sel(
            longitude=slice(min_lon+360, max_lon+360),
            latitude=slice(max_lat, min_lat),
            time=slice(min_time, max_time)).chunk({"time": cs}).load(scheduler="processes", num_workers=4)
        era5 = era5.assign_coords(longitude=((era5.longitude + 180) % 360) - 180).sortby("longitude")
    elif abs(min_lon) > max_lon:
        era5 = xarray.merge([
            ds[vs].sel(
                longitude=slice(None, max_lon),
                latitude=slice(max_lat, min_lat),
                time=slice(min_time, max_time)).chunk({"time": cs}).load(scheduler="processes", num_workers=4),
            ds[vs].sel(
                longitude=slice(ds["longitude"].max() - abs(min_lon), None),
                latitude=slice(max_lat, min_lat),
                time=slice(min_time, max_time)).chunk({"time": cs}).load(scheduler="processes", num_workers=4)
        ], join="outer", compat="no_conflicts")
        era5 = era5.assign_coords(longitude=((era5.longitude + 180) % 360) - 180).sortby("longitude")
    else:
        era5 = ds[vs].sel(
            longitude=slice(min_lon, max_lon),
            latitude=slice(max_lat, min_lat),
            time=slice(min_time, max_time)).chunk({"time": cs}).load(scheduler="processes", num_workers=4)

    return era5

def load_cordex_core(subset, vs, time_lim, lat_lim, lon_lim, cs=10_000):
    time0, time1 = time_lim
    lat0, lat1 = lat_lim
    lon0, lon1 = lon_lim

    dss = []
    model_coord = subset["model"] + ":" + subset["rcm"]
    for location in subset["location"]:
        ds = xarray.open_dataset(location)

        country_bbox = ds[vs].sel(
            longitude=slice(lon0, lon1),
            latitude=slice(lat0, lat1))

        time_subset = country_bbox.sel(time=slice(time0, time1))

        # need to load into memory now before any calendar handling
        ds = time_subset.chunk({"time": cs}).load(scheduler="processes", num_workers=4).convert_calendar("standard", align_on="date")
        normalized_time = pd.DatetimeIndex(ds.time.values).normalize()
        ds = ds.assign_coords(time=normalized_time)

        dss.append(ds)

    ds = xarray.concat(
        dss,
        dim=xarray.IndexVariable("model", model_coord),
        join="outer",
        coords="minimal")

    return ds
