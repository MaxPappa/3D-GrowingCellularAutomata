
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb , 0, 0.9999)

def to_rgbOld(x):
    # assume rgb premultiplied by alpha
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb, 0, 0.9999)

def visualize(growedObj):
    grow = growedObj.detach().cpu().numpy()
    x,y,z = np.indices(np.array(grow.shape[:-1])+1)
    maskTmp = grow[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0,maskTmp)
    ax = plt.figure().add_subplot(projection='3d')
    colors = to_rgb(grow)
    ax.voxels(x,y,z, masked.mask, facecolors=colors)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.show()

def visualizeNew(growedObj):
    grow = growedObj.detach().cpu().numpy()
    x,y,z = np.indices(np.array(grow.shape[:-1])+1)
    maskTmp = grow[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    ax = plt.figure().add_subplot(projection='3d')
    colors = to_rgb(grow)
    ax.voxels(x,y,z, masked.mask, facecolors=colors)
    ax.set(xlabel='x', ylabel='y', zlabel='z')
    plt.show()


def plot_loss(loss_log):
    plt.figure(figsize=(10, 4))
    plt.title('Loss history (log10)')
    plt.plot(np.log10(loss_log), '.', alpha=0.1)
    plt.show()


def visualizeGO(growedObject: torch.tensor, toSave: bool = False):
    grow = growedObject.detach().cpu().numpy()
    x,y,z = np.indices(grow.shape[:-1])
    maskTmp = grow[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(grow)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], 
                mode='markers',
                marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
            )
        ],
        layout=dict(
            scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)), template='plotly_dark'
        )
    )
    fig.update_layout(autosize=False, width=250, height=250, margin=dict(l=0,r=0,t=0,b=0))#, template='plotly')
    if toSave:
        return fig
    else:
        fig.show()

def visualizeGONew(growedObject):
    grow = growedObject.detach().cpu().numpy()
    x,y,z = np.indices(grow.shape[:-1])
    maskTmp = grow[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(grow)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], 
                mode='markers',
                marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
            )
        ],
        layout=dict(
            scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)), template='plotly_dark'
        )
    )
    fig.update_layout(autosize=False, width=250, height=250, margin=dict(l=0,r=0,t=0,b=0))#, template='plotly')
    return fig