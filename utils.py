import numpy as np
import open3d as o3d
from random import randint, sample
import torch


def readPLY(path: str) -> np.ndarray:
    cloud = o3d.io.read_point_cloud(path)
    xyz = np.asarray(cloud.points)
    colors = np.asarray(cloud.colors)

    xyz[:,0] = np.interp(xyz[:,0], (xyz[:,0].min(), xyz[:,0].max()), (0, (np.max(xyz[:,0])) - np.min(xyz[:,0])))
    xyz[:,1] = np.interp(xyz[:,1], (xyz[:,1].min(), xyz[:,1].max()), (0, (np.max(xyz[:,1])) - np.min(xyz[:,1])))
    xyz[:,2] = np.interp(xyz[:,2], (xyz[:,2].min(), xyz[:,2].max()), (0, (np.max(xyz[:,2])) - np.min(xyz[:,2])))
    xyz = xyz.astype(int)
    
    arr = np.zeros((xyz[:,0].max()+1, xyz[:,1].max()+1, xyz[:,2].max()+1))
    arr = arr - 1
    arr = np.stack([arr,arr,arr])
    for coords,col in zip(xyz, colors):
        arr[:,coords[0], coords[1], coords[2]] = col[:]
    arr = np.einsum('abcd -> bcda', arr)

    masked = np.ma.masked_where((arr[...,0]+arr[...,1]+arr[...,2])>0,arr[...,0])
    arr=np.append(arr,masked.mask.astype(int)[...,None], axis=3)
    arr[arr==-1]=0
    return arr.astype(np.float32)


def getCentroid(listCoords, perfectCenter):
    minDist = 100000
    newCenter = perfectCenter
    for c in listCoords:
        mse = np.square(perfectCenter - c).mean()
        if mse <= minDist:
            minDist = mse
            newCenter = c
    return newCenter

def take_cube(inp):
    x_s, y_s, z_s = inp.shape[1:4]
    min_side_x, min_side_y, min_side_z = max(3,x_s//5), max(3,y_s//5), max(3,z_s//5)
    max_side_x, max_side_y, max_side_z = x_s//3, y_s//3, z_s//3
    if min_side_x > max_side_x or min_side_y > max_side_y or min_side_z > max_side_z:
        return inp
    side_x = randint(min_side_x,max_side_x)
    side_y = randint(min_side_y,max_side_y)
    side_z = randint(min_side_z,max_side_z)
    x,y,z = torch.where(inp[:, side_x:x_s-side_x, side_y:y_s-side_y, side_z:z_s-side_z, 3:4] > 0.1)[1:4]
    if len(x) == 0:
        return inp
    index = randint(0,len(x)-1)
    x,y,z = x[index],y[index],z[index]
    block = inp[:, max(0,x - side_x) : x + side_x, max(0,y - side_y) : y + side_y, max(0, z - side_z) : z + side_z, :].clone()
    idx = torch.where((block[:,:,:,:,3:4]>0.1).sum(dim=[1,2,3,4]) >= 64)[0]
    if len(idx) == 0:
        return inp
    inp[idx, :, :, :, :] = 0
    inp[idx, max(0,x - side_x) : x + side_x, max(0,y - side_y) : y + side_y, max(0, z - side_z) : z + side_z, :] = block[idx].clone()
    return inp