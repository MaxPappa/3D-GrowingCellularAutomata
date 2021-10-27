
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
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


def visualizeValidation(input: torch.Tensor, output: torch.tensor, groundTruth: torch.tensor, batch_size: int):
    inp = input.detach().cpu().numpy()
    out = output.detach().cpu().numpy()
    gt = groundTruth.detach().cpu().numpy()
    x,y,z = np.indices(inp[0].shape[:-1])
    fig = make_subplots(rows=1, cols=3, width=600, height=400)
    
    maskTmp = inp[:,:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(inp)
    fig.add_trace(
        go.Scatter3d(
            x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
            marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
        ), row=1, col=1
    )
    
    maskTmp = out[:,:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(out)
    fig.add_trace(
        go.Scatter3d(
            x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
            marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
        ), row=1, col=2
    )
    
    maskTmp = gt[:,:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(gt)
    fig.add_trace(
        go.Scatter3d(
            x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
            marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
        ), row=1, col=3
    )

    fig.show()

def visualizeBatch(inpBatch: torch.tensor, outBatch: torch.tensor, batch_size: int):
    inp = inpBatch.detach().cpu().numpy()
    out = outBatch.detach().cpu().numpy()
    x,y,z = np.indices(inp[0].shape[:-1])
    fig = make_subplots(rows=2, cols=batch_size)
    for i in [0,1]:
        for j in range(0, batch_size):
            maskTmp = inp[j:j+1,:,:,:,3:4].squeeze() if i == 0 else out[j:j+1,:,:,:,3:4].squeeze()
            masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
            colors = to_rgb(inp[j:j+1]) if i == 0 else to_rgb(out[j:j+1])
            
            fig.add_trace(
                go.Scatter3d(
                    x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
                    marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
                ), row=i, col=j
            )
            # fig.add_scatter3d(
            #     x=x[masked.mask], y=y[masked.mask], z=z[masked.mask],
            #     mode="markers", marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area'),
            #     layout=dict(
            #         scene=dict(xaxis=dict(visible=False),yaxis=dict(visible=False),zaxis=dict(visible=False)),
            #         template='plotly_dark', autosize=False, width=250, height=250, margin=dict(l=0,r=0,t=0,b=0)
            #     ), row=i, col=j
            # )
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




def visualizeImprovements(output: torch.tensor, groundTruth: torch.tensor):
    out = output.detach().cpu().numpy()
    gt = groundTruth.detach().cpu().numpy()
    x,y,z = np.indices(out.shape[:-1])
    specs = [{"type":"scene"} for i in range(0,2)]
    fig = make_subplots(rows=1, cols=2, specs=[specs])
    
    maskTmp = out[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(out)
    fig.add_trace(
        go.Scatter3d(
            x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
            marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
        ), row=1, col=1
    )
    maskTmp = gt[:,:,:,3:4].squeeze()
    masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
    colors = to_rgb(gt)
    fig.add_trace(
        go.Scatter3d(
            x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
            marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
        ), row=1, col=2
    )
    return fig