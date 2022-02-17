import streamlit as st
from visualizationUtils import visualizeGO
# from streamlit_plotly_events import plotly_events
from utils import readPLY, getCentroid, take_cube
import numpy as np
import torch
import time
from CAModel import CAModel
from Config import ModelConfig
import random

st.title("3D Growing Cellular Automata")

animal = st.selectbox(
    "Select an animal",
    ('largepuffin', 'kangaroo', 'oryx', 'wildebeest', 'ostrich')
)

mode = st.selectbox(
    "Select a strategy",
    ("distillWay", "modifiedWay")
)

damageType = st.selectbox(
    "Select a type of damage",
    ("simple damage", "crop damage", "start from random seed", "replace with random cells")
)

rndm = st.number_input('Insert a number to set torch and random seeds. -1 is avoid manual seeds.', min_value=-1, step=1)
if rndm == -1:
    rndm = random.randint(0,1500)
torch.manual_seed(rndm)
random.seed(rndm)

perc = None
dmgTimes = None

if damageType == "replace with random cells":
    perc = st.slider("percentage of cells (RGBA and 12 hidden cell states) to replace with random values", 0.0, 1.0, 0.01)
elif damageType == "simple damage":
    dmgTimes = st.slider("number of times the object will be damaged in different positions", 1, 10, 1)

listCoords = []
target = readPLY(f"./plyFiles/{animal}.ply")
coords = np.where(target[:, :, :, 3:4] == 1)
for i in range(0, len(coords[0])):
    listCoords += [
        np.array([coords[0][i : i + 1][0], coords[1][i : i + 1][0], coords[2][i : i + 1][0]])
    ]

centerCoords = getCentroid(listCoords, np.array(target.shape[:3])//2)
seed = np.zeros(list(target.shape)[:-1] + [16], np.float32)

c = centerCoords if damageType != "start from random seed" else random.choice(listCoords)
seed[c[0], c[1], c[2], 3:] = 1.0
seed = torch.from_numpy(seed)
target = torch.from_numpy(target)

def make_cube_damage(inp):
    lungh = len(torch.where(inp[:,:,:,:,3:4]>0.1)[0])
    if lungh <=1:
        return inp
    num = random.sample(range(0,lungh-1), k=1)[0]
    coords = torch.where(inp[:,:,:,:,3:4]>0.1)
    x,y,z = coords[1][num],coords[2][num],coords[3][num]
    r = random.randint(5,8)
    inp[:,max(0,x-r):x+r, max(0,y-r):y+r, max(0,z-r):z+r,:] = 0
    return inp

def percentageNoisyCellsChange(inp, perc, equal=True):
    indices = (inp[:, :, :, :, 3:4] > 0.1)[0,:, :, :, 0].squeeze().nonzero(as_tuple=False)
    mask = torch.rand(indices.shape[0]) <= perc
    indices = indices[mask,:]
    if equal:
        noise = torch.rand(list(inp.shape[1:4])+[16], dtype=inp.dtype)
    else:
        lstNoise = []
        for i in range(0, 16):
            indices = (inp[:, :, :, :, 3:4] > 0.1)[0,:, :, :, 0].squeeze().nonzero(as_tuple=False)
            mask = torch.rand(indices.shape[0]) <= perc
            indices = indices[mask,:]
            noise = torch.rand(list(inp.shape[1:4]), dtype=inp.dtype)
            lstNoise += [noise]
            inp[:, indices[:,0], indices[:,1], indices[:,2],i] = noise[indices[:,0],indices[:,1],indices[:,2]]
        return inp    
    inp[:, indices[:,0], indices[:,1], indices[:,2],:] = noise[indices[:,0],indices[:,1],indices[:,2],:]
    return inp

cfgModel = ModelConfig()
ca_model = CAModel(cfgModel, min_step=140, max_step=180)
ca_model = ca_model.load_from_checkpoint(
    f"./checkpoints/{animal}/{animal}-{mode}.ckpt",
    hparams=cfgModel, min_step=150, max_step=180
    )

steps = 800
out = seed[None,...].clone()

#random.seed(42)

plot_spot = st.empty()
str_spot = st.empty()
flag = True
i = 0
while True:
    i += 1
    with str_spot:
        st.write("Growing")
    fig = visualizeGO(out.squeeze(), xyz=target.shape[:3])
    with plot_spot:
        st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
    out = ca_model(out, fire_rate=cfgModel.fire_rate)
    if i % 250 == 0 and i > 0 and damageType != "start from random seed" and flag:
        with str_spot:
            st.write("4 secs to rotate and see grown object before damaging it")
        time.sleep(4)
        with str_spot:
            st.write("4 secs to rotate and see the amount of damage")
        flag = False
        if damageType == "crop damage":
            out = take_cube(out)
        elif damageType == "simple damage":
            for i in range(0, dmgTimes):
                out = make_cube_damage(out)
        elif perc != None and damageType == "replace with random cells":
            out = percentageNoisyCellsChange(out, perc=perc, equal=False)
        fig = visualizeGO(out.squeeze(), xyz=target.shape[:3])
        with plot_spot:
            st.plotly_chart(fig, use_container_width=False, sharing="streamlit")
    if i % steps == 0:
        with str_spot:
            st.write("See grown/recovered object. Press R to re-run.")
        break