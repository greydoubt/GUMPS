# SPDX-FileCopyrightText: 2024 Amgen
#
# SPDX-License-Identifier: BSD-3-Clause

"HDF5 interface using addict and h5py"

import contextlib
import copy
import json
import logging
import pprint
import warnings
from pathlib import Path

import filelock
import numpy
from addict import Dict

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=FutureWarning)
    import h5py

logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())

class H5():
    pp = pprint.PrettyPrinter(indent=4)

    def __init__(self, filename:str):
        self.root = Dict()
        self.filename = filename

    def load(self, paths=None, update:bool=False, lock:bool=False):
        "load the data from an hdf5"
        if lock is True:
            lock_file = filelock.FileLock(self.filename + '.lock')
        else:
            lock_file = contextlib.nullcontext()

        with lock_file:
            with h5py.File(self.filename, 'r') as h5file:
                logger.debug("loading hdf5 from %s", self.filename)
                data = Dict(recursively_load(h5file, '/', paths))
                if update:
                    self.root.update(data)
                else:
                    self.root = data

    def save(self, lock:bool=False):
        "save the data to hdf5"
        if lock is True:
            lock_file = filelock.FileLock(self.filename + '.lock')
        else:
            lock_file = contextlib.nullcontext()

        with lock_file:
            with h5py.File(self.filename, 'w') as h5file:
                logger.debug("saving hdf5 to %s", self.filename)
                recursively_save(h5file, '/', self.root)

    def save_json(self):
        "save to a json file"
        with Path(self.filename).open("w", encoding='utf-8') as fp:
            data = convert_from_numpy(self.root)
            logger.debug("saving json to %s", self.filename)
            json.dump(data, fp, indent=4, sort_keys=True)

    def load_json(self, update:bool=False):
        "load data from json"
        with Path(self.filename).open("r", encoding='utf-8') as fp:
            logger.debug("loading json from %s", self.filename)
            data = json.load(fp)
            data = recursively_load_dict(data)
            if update:
                self.root.update(data)
            else:
                self.root = data

    def append(self, lock:bool=False):
        "This can only be used to write new keys to the system, this is faster than having to read the data before writing it"
        if lock is True:
            lock_file = filelock.FileLock(self.filename + '.lock')
        else:
            lock_file = contextlib.nullcontext()

        with lock_file:
            with h5py.File(self.filename, 'a') as h5file:
                logger.debug("appending hdf5 to %s", self.filename)
                recursively_save(h5file, '/', self.root)

    def update(self, merge):
        "update a H5 object using another one"
        self.root.update(copy.deepcopy(merge.root))

    def __getitem__(self, key:str):
        key = key.lower()
        obj = self.root
        for i in key.split('/'):
            if i:
                obj = obj[i]
        return obj

    def __setitem__(self, key:str, value):
        key = key.lower()
        obj = self.root
        parts = key.split('/')
        for i in parts[:-1]:
            if i:
                obj = obj[i]
        obj[parts[-1]] = value

def convert_from_numpy(data):
    "convert from a numpy array"
    ans = Dict()
    for key_original,item in data.items():
        if isinstance(item, numpy.ndarray):
            item = item.tolist()

        if isinstance(item, numpy.generic):
            item = item.item()

        if isinstance(item, bytes):
            item = item.decode('ascii')

        if isinstance(item, Dict):
            ans[key_original] = convert_from_numpy(item)
        else:
            ans[key_original] = item
    return ans

def recursively_load_dict(data):
    "recursively load data from dictionaries"
    ans = Dict()
    for key_original,item in data.items():
        if isinstance(item, dict):
            ans[key_original] = recursively_load_dict(item)
        else:
            ans[key_original] = item
    return ans

def set_path(obj, path, value):
    "paths need to be broken up so that subobjects are correctly made"
    path = path.split('/')
    path = [i for i in path if i]

    temp = obj
    for part in path[:-1]:
        temp = temp[part]

    temp[path[-1]] = value

def convert_type(value):
    "convert a numpy array of bytes to a python string"
    #takes care of strings
    try:
        return value.decode('utf8')
    except AttributeError:
        return value

def recursively_load( h5file, path, paths):
    "recursively load data"
    ans = Dict()
    if paths is not None:
        for path in paths:
            item = h5file.get(path, None)
            if item is not None:
                if isinstance(item, h5py.Dataset):
                    set_path(ans, path, convert_type(item[()]))
                elif isinstance(item, h5py.Group):
                    set_path(ans, path, recursively_load(h5file, path + '/', None))
    else:
        for key_original in h5file[path].keys():
            local_path = path + key_original
            item = h5file[path][key_original]
            if isinstance(item, h5py.Dataset):
                ans[key_original] = convert_type(item[()])
            elif isinstance(item, h5py.Group):
                ans[key_original] = recursively_load(h5file, local_path + '/', None)
    return ans

def recursively_save( h5file, path, dic):
    "recursively save"
    # argument type checking
    if not isinstance(dic, dict):
        raise ValueError("must provide a dictionary")

    if not isinstance(path, str):
        raise ValueError("path must be a string")

    if not isinstance(h5file, h5py.File):
        raise ValueError("must be an open h5py file")
    # save items to the hdf5 file
    for key, item in dic.items():
        key = str(key)
        if not isinstance(key, str):
            raise ValueError("dict keys must be strings to save to hdf5")
        #handle   int, float, string and ndarray of int32, int64, float64
        if isinstance(item, str):
            h5file[path + key] = numpy.array(item.encode('ascii'))
        elif isinstance(item, list) and all(isinstance(i, str) for i in item):
            h5file[path + key] = numpy.array([i.encode('ascii') for i in item])
        elif isinstance(item, dict):
            recursively_save(h5file, path + key + '/', item)
        else:
            try:
                h5file[path + key] = numpy.array(item)
            except TypeError:
                raise TypeError(f'Cannot save {path}/{key} key with {type(item)} type.')
