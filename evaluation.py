from utils import readPLY, getCentroid
import numpy as np
import torch
from CAModel import CAModel
from Config import ModelConfig
from random import randint, sample
from torch.nn.functional import mse_loss
import random
from utils import take_cube

def evaluate(animalName:str, plyFileName:str, trainMode:str, repeat_num:int=64) -> None:
    '''_summary_

    Args:
        animalName (str): type of animal
        plyFileName (str): name of PLY file to read
        trainMode (str): training mode checkpoint to use (distillWay, modifiedWay, modifiedNoNoiseWay)
        repeat_num (int, optional): number of examples to compute. Defaults to 64.

    Returns:
        None: no returned value, function used just to print things.
    '''    
    # setting all seed for reproducibility purposes
    print(f"### Starting evaluation for {animalName} with training mode {trainMode} ###")
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)

    listCoords = []
    target = readPLY(f"./plyFiles/{plyFileName}")
    coords = np.where(target[:, :, :, 3:4] == 1)
    for i in range(0, len(coords[0])):
        listCoords += [
            np.array([coords[0][i : i + 1][0], coords[1][i : i + 1][0], coords[2][i : i + 1][0]])
        ]

    centerCoords = getCentroid(listCoords, np.array(target.shape[:3])//2)
    print(centerCoords)
    seed = np.zeros(list(target.shape)[:-1] + [16], np.float32)
    c = centerCoords
    seed[c[0], c[1], c[2], 3:] = 1.0
    seed = torch.from_numpy(seed)
    target = torch.from_numpy(target).cuda()

    cfgModel = ModelConfig()
    min_steps = (max(target.shape)*2)+10
    ca_model = CAModel(cfgModel, min_step=min_steps, max_step=min_steps+40)#min_step=120, max_step=160)# BEST: MIN=70, MAX=105

    ca_model = ca_model.load_from_checkpoint(
        f"./checkpoints/{animalName}/{animalName}-{trainMode}.ckpt",
        hparams=cfgModel, min_step=min_steps, max_step=min_steps+40
    )
    ca_model = ca_model.cuda()

    steps = 250
    out = seed[None,...].detach().clone()

    inpSeed = seed[None,...].cuda()
    inpSeed = ca_model(inpSeed, steps=steps, fire_rate=cfgModel.fire_rate)
    growing_loss = mse_loss(inpSeed.squeeze()[:,:,:,:4], target)
    print(f"grow from simple central seed {growing_loss}")

    batch_x = torch.repeat_interleave(inpSeed, repeat_num, dim=0).cuda()

    batch_x_cropped = batch_x.clone()
    for i in range(0, repeat_num):
        batch_x_cropped[i] = take_cube(batch_x_cropped[i][None,...])
    # batch_x_cropped = take_cube_eval(batch_x.detach().clone(), repeat_num)
    reconstruct_crop_loss = 0
    batch_recovered_crop = batch_x_cropped.clone()
    for i in range(0, repeat_num):
        batch_recovered_crop[i] = ca_model(batch_x_cropped[i][None,...], steps=steps, fire_rate=cfgModel.fire_rate)
        reconstruct_crop_loss += mse_loss(batch_recovered_crop[i][:,:,:,:4], target)

    reconstruct_crop_loss /= repeat_num
    print(f"re-grow from cropped piece {reconstruct_crop_loss}")

    def make_damage_eval(inp, repeat_num):
        for i in range(0, repeat_num):
            lungh = len(torch.where(inp[i,:,:,:,3:4]>0.1)[0])
            num = sample(range(0,lungh-1), k=1)[0]
            coords = torch.where(inp[i,:,:,:,3:4]>0.1)
            x,y,z = coords[1][num],coords[2][num],coords[3][num]
            r = randint(5,8)
            inp[i,max(0,x-r):x+r, max(0,y-r):y+r, max(0,z-r):z+r,:] = 0
        return inp

    batch_x_damaged = make_damage_eval(batch_x.detach().clone(), repeat_num)
    reconstruct_damage_loss = 0
    for i in range(0, repeat_num):
        computed = ca_model(batch_x_damaged[i][None,...], steps=steps, fire_rate=cfgModel.fire_rate)
        reconstruct_damage_loss += mse_loss(computed.squeeze()[:,:,:,:4], target)

    reconstruct_damage_loss /= repeat_num
    print(f"recover from cube damage {reconstruct_damage_loss}")

    batch_seeds = torch.repeat_interleave(seed[None,...], repeat_num, dim=0).cuda()
    for i in range(0, repeat_num):
        center = sample(listCoords, k=1)[0]
        batch_seeds[i,center[0], center[1], center[2], 3:] = 1.0

    random_seed_loss = 0
    for i in range(0, repeat_num):
        computed = ca_model(batch_seeds[i][None,...], steps=steps, fire_rate=cfgModel.fire_rate)
        random_seed_loss += mse_loss(computed.squeeze()[:,:,:,:4], target)
    random_seed_loss /= repeat_num
    print(f"grow from random position seed {random_seed_loss}")
    print(f"### The End ###\n")

if __name__ == '__main__':
    evaluate("ostrich", "ostrich.ply", "distillWay", 64)
    evaluate("ostrich", "ostrich.ply", "modifiedWay", 64)
    evaluate("largepuffin", "largepuffin.ply", "distillWay", 64)
    evaluate("largepuffin", "largepuffin.ply", "modifiedWay", 64)
    evaluate("oryx", "oryx.ply", "distillWay", 64)
    evaluate("oryx", "oryx.ply", "modifiedWay", 64)
    evaluate("kangaroo", "kangaroo.ply", "distillWay", 64)
    evaluate("kangaroo", "kangaroo.ply", "modifiedWay", 64)
    evaluate("Wildebeest", "Wildebeest.ply", "distillWay", 64)
    evaluate("Wildebeest", "Wildebeest.ply", "modifiedWay", 64)