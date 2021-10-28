
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import torch

def to_alpha(x):
    return np.clip(x[..., 3:4], 0, 0.9999)

def to_rgb(x):
    rgb, a = x[..., :3], to_alpha(x)
    return np.clip(1.0-a+rgb , 0, 0.9999)

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

def visualizePatterns(inpBatch: torch.tensor, num: int):
    inp = inpBatch.detach().cpu().numpy()
    x,y,z = np.indices(inp[0].shape[:-1])
    rows = num//4
    cols = num//rows
    specs = [{"type":"scene"} for i in range(0,cols)]
    fig = make_subplots(rows=rows, cols=cols, specs=[specs for i in range(0,rows)])
    idx = 0
    for i in range(1,rows+1):
        for j in range(1,cols+1):
            maskTmp = inp[idx:idx+1,:,:,:,3:4].squeeze()
            masked = np.ma.masked_where((maskTmp)>0.1,maskTmp)
            colors = to_rgb(inp[idx:idx+1])
            
            fig.add_trace(
                go.Scatter3d(
                    x=x[masked.mask], y=y[masked.mask], z=z[masked.mask], mode='markers',
                    marker=dict(color=colors[masked.mask],size=2,symbol='square',sizemode='area')
                ), row=i, col=j
            )
            idx += 1
    return fig