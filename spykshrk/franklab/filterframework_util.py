import numpy as np
from spykshrk.util import AttrDict


class DataFormatError(RuntimeError):
    pass


def parse_filterframework_cells(data, level=0):
    store = AttrDict({})
    if isinstance(data, np.ndarray) or isinstance(data, np.void):
        if data.dtype.kind == 'O':
            for ind, entry in enumerate(data):
                new_attr = parse_filterframework_cells(entry, level=level+1)
                if new_attr is not None:
                    if entry.ndim == 2:
                        store[ind] = new_attr
                    elif entry.ndim == 1:
                        store.update(new_attr)
                    else:
                        raise RuntimeError('WUT? ಠ_ಠ')
        elif data.dtype.kind == 'V':
            for ind, entry in enumerate(data):
                if entry.ndim == 0:
                    # endpoint
                    for name in entry.dtype.names:
                        store[name] = np.squeeze(entry[name])
                    return store

                new_attr = parse_filterframework_cells(entry, level=level+1)
                if new_attr is not None:
                    if entry.ndim == 2:
                        store[ind] = new_attr
                    elif entry.ndim == 1:
                        store.update(new_attr)
                    else:
                        raise RuntimeError('WUT? ಠ_ಠ')

        elif data.dtype in ['i2', 'i4', 'i8', 'f4', 'f8', 'c8', 'c16']:
            # one possible leaf
            if data.size == 0:
                # No data in leaf
                return None
        else:
            if data.dtype.names is not None:
                # If this is a list of fields
                for name in data.dtype.names:
                    store[name] = np.squeeze(data[name])

                return store
    return store


def merge_filterframework_cells(data_list):
    merged = AttrDict()
    if all([isinstance(data, AttrDict) for data in data_list]):
        for data in data_list:
            key_collision = set(merged.keys()).intersection(data.keys())
            # resolve by looking at next level
            for key in key_collision:
                resolved = merge_filterframework_cells([merged[key], data[key]])
                merged[key] = resolved

            to_update = set(data.keys()).difference(merged.keys())
            for key in to_update:
                merged[key] = data[key]

        return merged

    elif all([isinstance(data, np.ndarray) for data in data_list]):
        # leaf
        if all([np.all(data_list[0] == data) for data in data_list]):
            # Data is all the same
            return data_list[0]
        else:
            raise DataFormatError('Collision detected, cannot merge datasets with overlapping keys and different data.')

    else:
        raise DataFormatError('Merge failed, either unsupported data or level with different datatypes provided.')




