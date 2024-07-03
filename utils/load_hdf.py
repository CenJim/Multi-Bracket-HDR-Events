import pandas as pd
import h5py
import numpy as np
import hdf5plugin
import os


def chunk_2d_array(array, chunk_size):
    chunked_array = []
    for i in range(0, len(array), chunk_size):
        chunked_array.append(array[i:i + chunk_size, :])
    return chunked_array


def get_dataset(file_path, chunk_flag=False, width=None, height=None, num_events_per_pixel=None):
    # open HDF5 file
    with h5py.File(os.path.join(file_path, 'events.h5'), 'r') as f:
        # obtain the t x y p
        dataset = np.zeros(shape=(len(f['events/p']), 4))
        dataset[:, 0] = f['events/t'][()]
        dataset[:, 1] = f['events/x'][()]
        dataset[:, 2] = f['events/y'][()]
        dataset[:, 3] = f['events/p'][()]

        with h5py.File(os.path.join(file_path, 'rectify_map.h5'), 'r') as rec:
            rectify_map = rec['rectify_map'][()]
            xy_rect = rectify_map[dataset[:, 2].astype(int), dataset[:, 1].astype(int)]
            dataset[:, 1] = xy_rect[:, 0]  # rectify the x of event sequence
            dataset[:, 2] = xy_rect[:, 1]  # rectify the y of event sequence

        if chunk_flag:
            return chunk_2d_array(dataset, int(width * height * num_events_per_pixel))
        else:
            return dataset


def get_event_offset(file_path):
    with h5py.File(os.path.join(file_path, 'events.h5'), 'r') as f:
        # obtain the offset
        offset = f['t_offset'][()]
        return offset


if __name__ == '__main__':
    file_path = '/Volumes/CenJim/train data/dataset/DSEC/train/interlaken_00_c/Interlaken events left/'
    print(f'offset: {get_event_offset(file_path)}')
    dataset = get_dataset(file_path, 640, 480, 0.5)
    # for event in dataset:
    #     print(event)
    #     break
    print(len(dataset))
    print(dataset[:3])
