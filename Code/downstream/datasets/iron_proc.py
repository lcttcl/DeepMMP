"""
@time    : 2022/8/17 21:19
@author  : x1aolata
@file    : iron_proc.py
@script  : 纯铁晶粒预处理代码
           将 pure_iron_grain_dataset.hdf5 文件 处理成 两个 npy矩阵
           便于后期读取
"""
import h5py
import numpy as np

root_path = '/jiangruohui/_datasets/iron/'
file_name = 'pure_iron_grain_dataset.hdf5'

with h5py.File(root_path + file_name, 'r') as hdf:
    real_image_value = hdf['real'].get('image')[()]
    real_boundary_value = hdf['real'].get('boundary')[()]
    real_image_np = np.array(real_image_value)
    real_boundary_np = np.array(real_boundary_value)

np.save(root_path + 'image.npy', real_image_np)
np.save(root_path + 'boundary.npy', real_boundary_np)
print(real_image_np.shape, real_boundary_np.shape)
