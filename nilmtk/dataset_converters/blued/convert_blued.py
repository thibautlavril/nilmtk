# -*- coding: utf-8 -*-
"""
Created on Mon Oct 27 10:41:45 2014

@author: thibaut
"""

from __future__ import print_function, division
import pandas as pd
import numpy as np
import os
from os.path import join, isdir, isfile
from sys import stdout
import scipy.io
from dateutil.parser import parse
import dateutil.tz
import datetime
from nilmtk.datastore import Key
from nilmtk.measurement import LEVEL_NAMES
from nilm_metadata import convert_yaml_to_hdf5
from nilmtk.utils import get_module_directory


def convert_blued(blued_path, hdf_filename):
    """
    Parameters
    ----------
    redd_path : str
    The root path of the REDD low_freq dataset.
    hdf_filename : str
    The destination HDF5 filename (including path and suffix).
    """
    _convert(blued_path, hdf_filename, _blued_measurement_mapping_func,
             'US/Eastern')
    # Add metadata
    convert_yaml_to_hdf5(join(get_module_directory(), 
                              'dataset_converters', 
                              'blued', 
                              'metadata'),
                         hdf_filename)
    print("Done converting BLUED to HDF5!")


def _convert(input_path, hdf_filename, measurement_mapping_func, tz):
    """
    Parameters
    ----------
    input_path : str
    The root path of the REDD low_freq dataset.
    hdf_filename : str
    The destination HDF5 filename (including path and suffix).
    measurement_mapping_func : function
    Must take these parameters:
    - house_id
    - chan_id
    Function should return a list of tuples e.g. [('power', 'active')]
    tz : str
    Timezone e.g. 'US/Eastern'
    """
    assert isdir(input_path)
    # Open HDF5 file
    with pd.get_store(hdf_filename) as store:
        # Iterate though all houses and datasets
        locations = [1]
        for lc_id in locations:
            print("Loading location", lc_id, end="... ")
            stdout.flush()
            datasets = [1]
            for ds_id in datasets:
                print(ds_id, end=" ")
                stdout.flush()
                df_list = _load_ds(lc_id, ds_id, input_path, tz)
                for meter in [1, 2]:
                    key = Key(building=lc_id, meter=meter)
                    df = df_list[meter-1]
                    store.put(str(key), df, format='table')
                    store.flush()
                    print()
        store.close()


def _load_ds(location, dataset, input_path, tz):
    """
    Parameters
    ----------
    input_path : (str) the root path of the REDD low_freq dataset
    key_obj : (nilmtk.Key) the house and channel to load
    columns : list of tuples (for hierarchical column index)
    tz : str e.g. 'US/Eastern'
    Returns
    -------
    DataFrame of data.
    """
    # Get path
    location_path = 'location_00{:d}'.format(location)
    dataset_path = "_".join((location_path, 'dataset_00{:d}'.format(dataset)))
    path = join(input_path, dataset_path)
    assert isdir(path)
    # Load power for each matlab file in dataset
    power = None
    for sub_file in range(1, 5):
        # Get file
        location_path = 'location_00{:d}'.format(location)
        filename = "_".join((location_path,
                             'matlab_{:d}.mat'.format(sub_file)))
        filename = join(path, filename)
        assert isfile(filename)
        # Load matlab file
        mat = scipy.io.loadmat(filename)
        t = mat['data'][0][0][2]
        t = t.reshape(len(t))
        tt = mat['data'][0][0][3].reshape(len(t))
        Qa = mat['data'][0][0][4][0].reshape(len(t), 1)
        Qb = mat['data'][0][0][5][0].reshape(len(t), 1)
        Pa = mat['data'][0][0][6][0].reshape(len(t), 1)
        Pb = mat['data'][0][0][7][0].reshape(len(t), 1)
        startDate = mat['data'][0][0][9][0][0][0][0][0]
        startTime = mat['data'][0][0][10][0][0][0][0][0]
        del mat
        # Put in np.array
        p = np.concatenate((Pa, Pb, Qa, Qb), axis=1)
        if power is None:
            power = p
            tt_power = tt
        else:
            power = np.concatenate((power, p), axis=0)
            tt_power = np.concatenate((tt_power, tt), axis=0)
    # Calculation of offset
    start = parse(startDate+' '+startTime)
    start = start.replace(tzinfo=dateutil.tz.gettz(tz))
    zero = datetime.datetime(1970, 1, 1)
    zero = zero.replace(tzinfo=dateutil.tz.gettz('UTC'))
    offset = (start-zero).total_seconds()
    tt_power = tt_power+offset

    # Put the power in 2 panda DataFrames
    df_list = []
    for meter in [1, 2]:
        measurements = _blued_measurement_mapping_func(location, meter)
        m = (meter-1)
        idx = pd.MultiIndex.from_tuples(measurements, names=LEVEL_NAMES)
        df = pd.DataFrame(power[:, [m, m+2]], columns=idx,
                          index=tt_power, dtype='float32')
        # raw REDD data isn't always sorted
        df = df.sort_index()
        # Convert the integer index column to timezone-aware datetime
        df.index = pd.to_datetime(df.index.values, unit='s', utc=True)
        df = df.tz_convert(tz)
        df_list.append(df)
    return df_list


def _blued_measurement_mapping_func(house_id, meter):
        return [('power', 'active'), ('power', 'reactive')]
