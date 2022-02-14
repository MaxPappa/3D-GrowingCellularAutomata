import numpy as np
import open3d as o3d


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