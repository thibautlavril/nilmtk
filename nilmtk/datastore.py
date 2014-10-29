from __future__ import print_function, division
import pandas as pd
from itertools import repeat
from time import time
from copy import deepcopy
from .timeframe import TimeFrame, timeframes_from_periodindex
from .node import Node

# do not edit! added by PythonBreakpoints
from pdb import set_trace as _breakpoint


MAX_MEM_ALLOWANCE_IN_BYTES = 1E9

class DataStore(object):
    """
    Provides a common interface to all physical data stores.  
    Supports hierarchical stores.
    
    The DataStore class lives in the bottom layer of NILMTK.  It loads
    a single chunk at a time from physical location and returns a
    DataFrame.

    * Deals with: retrieving data from disk / network / direct from a meter
    * Optimised for: handling large amounts of data
    * Services it provides: delivering a generator of pd.DataFrames of data given a
      specific time span and columns
    * Totally agnostic about what the data 'means'. It could be voltage,
      current, temperature, PIR readings etc.
    * could have subclasses for NILMTK HDF5, NILMTK CSV, Xively, REDD, iAWE,
      UK-DALE, MetOffice XLS data, Current Cost meters etc.
    * One DataStore per HDF5 file or folder or CSV files or Xively
      feed etc.

    Attributes
    ----------
    window : nilmtk.TimeFrame
        Defines the timeframe we are interested in.
    """
    def __init__(self):
        self.window = TimeFrame()


class HDFDataStore(DataStore):
    def __init__(self, filename, mode='r'):
        """
        Parameters
        ----------
        filename : string
        mode : string
            File open mode.  e.g. 'r' or 'w'
        """
        self.store = pd.HDFStore(filename, mode=mode)
        super(HDFDataStore, self).__init__()

    def load(self, key, cols=None, sections=None, n_look_ahead_rows=0,
             chunksize=1000000):

        #_breakpoint()  # 9a2a1bb0

        """
        Parameters
        ----------
        key : string, the location of a table within the DataStore.
        cols : list of Measurements, optional
            e.g. [('power', 'active'), ('power', 'reactive'), ('voltage')]
            if not provided then will return all columns from the table.
        sections : list of nilmtk.TimeFrame objects or a pd.PeriodIndex, optional
            defines the time sections to load.  If `self.window` is enabled
            then each `section` will be intersected with `self.window`.
        n_look_ahead_rows : int, optional, defaults to 0
            If >0 then each returned DataFrame will have a `look_ahead`
            property which will be a DataFrame of length `n_look_ahead_rows`
            of the data immediately in front of the data in the main DataFrame.
        chunksize : int, optional

        Returns
        ------- 
        generator of DataFrame objects
            Each DataFrame is has extra attributes:
                - timeframe : TimeFrame of section intersected with self.window
                - look_ahead : pd.DataFrame:
                    with `n_look_ahead_rows` rows.  The first row will be for
                    `section.end`.  `look_ahead` stores data which appears on 
                    disk immediately after `section.end`; i.e. it ignores
                    the next `section.start`.

            Returns an empty DataFrame if no data is available for the
            specified section (or if the section.intersect(self.window)
            is empty).
        """
        # TODO: calculate chunksize default based on physical 
        # memory installed and number of columns

        # Make sure key has a slash at the front but not at the end.
        if key[0] != '/':
            key = '/' + key
        if len(key) > 1 and key[-1] == '/':
            key = key[:-1]


        sections = [TimeFrame()] if sections is None else sections
        if isinstance(sections, pd.PeriodIndex):
            sections = timeframes_from_periodindex(sections)

        self.all_sections_smaller_than_chunksize = True

        for section in sections:
            window_intersect = self.window.intersect(section)
            if window_intersect.empty:
                # Trick to make a generator with just a single, empty DataFrame
                generator = repeat(pd.DataFrame(), 1)
            else:
                terms = window_intersect.query_terms('window_intersect')
                generator = self.store.select(key=key, cols=cols, where=terms,
                                              chunksize=chunksize).__iter__()
                
            first_sub_chunk = True
            for data in generator:
                if len(data) <= 2:
                    continue

                if not first_sub_chunk:
                    self.all_sections_smaller_than_chunksize = False

                # Load look ahead if necessary
                if n_look_ahead_rows > 0:
                    if len(data.index) > 0:
                        look_ahead_coords = self.store.select_as_coordinates(
                            key=key, where="index>data.index[-1]")
                    else:
                        look_ahead_coords = []
                    if len(look_ahead_coords) > 0:
                        look_ahead_start = look_ahead_coords[0]
                        look_ahead_iterator = self.store.select(
                            key=key, chunksize=n_look_ahead_rows,
                            cols=cols, start=look_ahead_start).__iter__()
                        data.look_ahead = next(look_ahead_iterator)
                    else:
                        data.look_ahead = pd.DataFrame()

                # Set timeframe
                start = None
                end = None
                if len(data) == chunksize:
                    # If len(data) == chunksize then we can fairly safely
                    # assume that the full `section` is larger than the
                    # chunksize so we're going to go through several
                    # 'sub chunks'.
                    if first_sub_chunk:
                        start = window_intersect.start
                elif not first_sub_chunk:
                    # This is the last subchunk
                    end = window_intersect.end
                else:
                    # Just a single 'subchunk'
                    start = window_intersect.start
                    end = window_intersect.end

                if start is None:
                    start = data.index[0]
                if end is None:
                    end = data.index[-1]

                data.timeframe = TimeFrame(start, end)
                first_sub_chunk = False

                yield data

    def append(self, *args, **kwargs):
        self.store.append(*args, **kwargs)

    def load_metadata(self, key='/'):
        """
        Parameters
        ----------
        key : string, optional
            if '/' then load metadata for the whole dataset.

        Returns
        -------
        metadata : dict
        """
        if key == '/':
            node = self.store.root
        else:
            node = self.store.get_node(key)

        metadata = deepcopy(node._v_attrs.metadata)
        return metadata

    def save_metadata(self, key, metadata):
        """
        Parameters
        ----------
        key : string
        metadata : dict
        """

        if key == '/':
            node = self.store.root
        else:
            node = self.store.get_node(key)

        node._v_attrs.metadata = metadata
        self.store.flush()

    def elements_below_key(self, key='/'):
        """
        Returns
        -------
        list of strings
        """
        if key == '/' or not key:
            node = self.store.root
        else:
            node = self.store.get_node(key)
        return node._v_children.keys()

    def close(self):
        self.store.close()

    def open(self):
        self.store.close()
    
    def _check_columns(self, key, columns):
        if columns is None:
            return
        if not self._table_has_column_names(key, columns):
            raise KeyError('at least one of ' + str(columns) + 
                           ' is not a valid column')

    def _table_has_column_names(self, key, cols):
        """
        Parameters
        ----------
        cols : string or list of strings
        
        Returns
        -------
        boolean
        """
        assert cols is not None
        self._check_key(key)
        if isinstance(cols, str):
            cols = [cols]
        query_cols = set(cols)
        table_cols = set(self._column_names(key) + ['index'])
        return query_cols.issubset(table_cols)

    def _column_names(self, key):
        self._check_key(key)
        storer = self._get_storer(key)
        col_names = storer.non_index_axes[0][1:][0]
        return col_names

    def _check_data_will_fit_in_memory(self, key, nrows, cols=None):
        # Check we won't use too much memory
        mem_requirement = self._estimate_memory_requirement(key, nrows, cols)
        if mem_requirement > MAX_MEM_ALLOWANCE_IN_BYTES:
            raise MemoryError('Requested data would use {:.3f}MBytes:'
                              ' too much memory.'
                              .format(mem_requirement / 1E6))

    def _estimate_memory_requirement(self, key, nrows, cols=None, paranoid=False):
        """Returns estimated mem requirement in bytes."""
        BYTES_PER_ELEMENT = 4
        BYTES_PER_TIMESTAMP = 8
        if paranoid:
            self._check_key(key)
        if cols is None:
            cols = self._column_names(key)
        elif paranoid:
            self._check_columns(key, cols)
        ncols = len(cols)
        est_mem_usage_for_data = nrows * ncols * BYTES_PER_ELEMENT
        est_mem_usage_for_index = nrows * BYTES_PER_TIMESTAMP
        if cols == ['index']:
            return est_mem_usage_for_index
        else:
            return est_mem_usage_for_data + est_mem_usage_for_index
       
    def _nrows(self, key, timeframe=None):
        """
        Returns
        -------
        nrows : int
        """
        timeframe_intersect = self.window.intersect(timeframe)
        if timeframe_intersect.empty:
            nrows = 0
        elif timeframe_intersect:
            terms = timeframe_intersect.query_terms('timeframe_intersect')
            coords = self.store.select_as_coordinates(key, terms)
            nrows = len(coords)
        else:
            storer = self._get_storer(key)
            nrows = storer.nrows
        return nrows
        
    def get_timeframe(self, key):
        """
        Returns
        -------
        nilmtk.TimeFrame of entire table after intersecting with self.window.
        """
        data_start_date = self.store.select(key, [0]).index[0]
        data_end_date = self.store.select(key, start=-1).index[0]
        timeframe = TimeFrame(data_start_date, data_end_date)
        return self.window.intersect(timeframe)
    
    def _keys(self):
        return self.store.keys()

    def _get_storer(self, key):
        self._check_key(key)
        storer = self.store.get_storer(key)
        assert storer is not None, "cannot get storer for key = " + key
        return storer
    
    def _check_key(self, key):
        """
        Parameters
        ----------
        key : string
        """
        if key not in self._keys():
            raise KeyError(key + ' not in store')


def join_key(*args):
    """
    Examples
    --------
    >>> join_key('building1', 'elec', 'meter1')
    '/building1/elec/meter1'

    >>> join_key('/')
    '/'

    >>> join_key('')
    '/'
    """
    key = '/'
    for arg in args:
        arg_stripped = str(arg).strip('/')
        if arg_stripped:
            key += arg_stripped + '/'
    if len(key) > 1:
        key = key[:-1] # remove last trailing slash
    return key


class Key(object):
    """A location of data or metadata within NILMTK.
    
    Attributes
    ----------
    building : int
    meter : int
    utility : str
    """

    def __init__(self, string=None, building=None, meter=None):
        """
        Parameters
        ----------
        string : str, optional
            e.g. 'building1/elec/meter1'
        building : int, optional
        meter : int, optional
        """
        self.utility = None
        if string is None:
            self.building = building
            self.meter = meter
        else:
            split = string.strip('/').split('/')
            assert split[0].startswith('building'), "The first element must be 'building<I>', e.g. 'building1'; not '{}'.".format(split[0])
            try:
                self.building = int(split[0].replace("building", ""))
            except ValueError as e:
                raise ValueError("'building' must be followed by an integer.\n{}"
                                 .format(e))
            if len(split) > 1:
                self.utility = split[1]
            if len(split) == 3:
                assert split[2].startswith('meter')
                self.meter = int(split[-1].replace("meter", ""))
            else:
                self.meter = None
        self._check()

    def _check(self):
        assert isinstance(self.building, int)
        assert self.building >= 1
        if self.meter is not None:
            assert isinstance(self.meter, int)
            assert self.meter >= 1

    def __repr__(self):
        self._check()
        s = "/building{:d}".format(self.building)
        if self.meter is not None:
            s += "/elec/meter{:d}".format(self.meter)
        return s
