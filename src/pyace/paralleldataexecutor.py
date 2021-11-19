import logging

log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)

import warnings
from concurrent.futures import Future
from functools import partial
import multiprocessing
from threading import Lock

from concurrent.futures import Executor
from concurrent.futures import ProcessPoolExecutor

import pandas as pd

# very random name to avoid global namespace collision
LOCAL_DATAFRAME_VARIALBE_NAME = "dh3ah3k5sk3n9v62m3l2"

import __main__



# https://stackoverflow.com/questions/10434593/dummyexecutor-for-pythons-futures
class SerialExecutor(Executor):
    def __init__(self, initializer=None, initargs=None):
        self.initializer = initializer
        self.initargs = initargs
        self._shutdown = False
        self._shutdownLock = Lock()

        self._initialize()

    def _initialize(self):
        self.initializer(*self.initargs)

    def submit(self, fn, *args, **kwargs):
        with self._shutdownLock:
            if self._shutdown:
                raise RuntimeError('cannot schedule new futures after shutdown')
            f = Future()
            try:
                result = fn(*args, **kwargs)
            except BaseException as e:
                f.set_exception(e)
            else:
                f.set_result(result)
            return f

    def shutdown(self, wait=True):
        with self._shutdownLock:
            self._shutdown = True


def split_by_batches(l, batch_size):
    return [l[i:i + batch_size] for i in range(0, len(l), batch_size)]


def local_dataframe_initializer(df):
    setattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME, df)


def batch_function_wrapper(batch_indices, pure_row_func):
    # import __main__
    _local_df = getattr(__main__, LOCAL_DATAFRAME_VARIALBE_NAME)
    batch_df = _local_df.loc[batch_indices]
    if isinstance(batch_df, pd.Series):
        return batch_df.map(pure_row_func)
    elif isinstance(batch_df, pd.DataFrame):
        return batch_df.apply(pure_row_func, axis=1)


class ParallelDataExecutor:
    MODE_PROCESS = "process"
    MODE_MPI = "mpi"
    MODE_SERIAL = "serial"

    def __init__(self, distributed_data: pd.DataFrame, parallel_mode: str = MODE_SERIAL, index_col: str = None,
                 batch_size: int = None, n_workers: int = None):
        self._distributed_df = distributed_data
        if index_col is not None:
            self._distributed_df.set_index(index_col)
        self._data_index = self._distributed_df.index
        if len(set(self._data_index)) < len(self._data_index):
            raise ValueError("Non-unique indices found in dataset")
        self._data_indices_batches = None

        self.parallel_mode = parallel_mode
        self._n_workers = n_workers
        if self.parallel_mode != ParallelDataExecutor.MODE_PROCESS:
            warnings.warn("ParallelDataExecutor: n_workers ({}) would be ignored for parallel_mode={}".format(n_workers,
                                                                                                              parallel_mode))

        self._executor = None
        if batch_size is None:
            self.batch_size = len(distributed_data)
        else:
            self.batch_size = batch_size
        self._actual_batch_size = None

    def _split_data_indices_batches(self):
        self._data_indices_batches = split_by_batches(self._data_index, self._actual_batch_size)

    def start_executor(self):
        if self._executor is not None:
            self.stop_executor()
        if self.parallel_mode == ParallelDataExecutor.MODE_MPI:
            try:
                from mpi4py import MPI
                from mpi4py.futures import MPIPoolExecutor
            except ImportError:
                raise ImportError(
                    'Could not import mpi4py to run in "mpi" parallel mode. Please install mpi4py or use another mode')
            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            self._n_workers = max(1, size - 1)
            self._executor = MPIPoolExecutor(globals={LOCAL_DATAFRAME_VARIALBE_NAME: self._distributed_df})
        elif self.parallel_mode == ParallelDataExecutor.MODE_PROCESS:
            if self._n_workers is None:
                self._n_workers = multiprocessing.cpu_count()
            self._executor = ProcessPoolExecutor(max_workers=self._n_workers, initializer=local_dataframe_initializer,
                                                 initargs=(self._distributed_df,))
        elif self.parallel_mode == ParallelDataExecutor.MODE_SERIAL:
            self._n_workers = 1
            self._executor = SerialExecutor(initializer=local_dataframe_initializer, initargs=(self._distributed_df,))
        else:
            raise ValueError("Unrecognized parallel_mode='{}'".format(self.parallel_mode))
        self._actual_batch_size = max(1, min(self.batch_size, len(self._data_index) // self._n_workers))
        log.info("Start executor in '{}' mode with {} workers, actual batch size is {}" \
                 .format(self.parallel_mode, self._n_workers, self._actual_batch_size))
        self._split_data_indices_batches()

    def stop_executor(self):
        if self._executor is not None:
            self._executor.shutdown()
            self._executor = None

    def map(self, pure_row_func=None, wrapped_pure_func=None):
        if self._executor is None:
            self.start_executor()

        res_series = pd.Series(index=self._data_index, dtype=object)
        if wrapped_pure_func is None:
            wrapped_pure_func = partial(batch_function_wrapper, pure_row_func=pure_row_func)
        for ind_range, val in zip(self._data_indices_batches,
                                  self._executor.map(wrapped_pure_func, self._data_indices_batches)):
            res_series[ind_range] = val
        return res_series
