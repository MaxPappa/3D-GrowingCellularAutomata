# 3D-GrowingCellularAutomata
Growing cellular automata (as described in https://distill.pub/2020/growing-ca/) adapted to 3D objects.


To reproduce everything, setup a new environment using python 3.8 and then install all dependencies
```
python -m pip install -r requirements.txt
```

To run the streamlit demo:
```
streamlit run streamlit-demo.py
```
___

Animals in PLY format downloaded from [SketchFab-WaxFreeOintment](https://sketchfab.com/WaxFreeOintment/models)

Each model is trained for 3000-6000 steps (higher is the amount of pixels, more steps will require to converge) using
Colab GPUs. With a Nvidia T4 (Colab) the time required may vary from 30 minutes to 1 hour using a batch size of 4 and a pool size of 128.

For each animal object there are 3 checkpoints:

* **distillWay** which is the strategy reproduced from GNCA distill paper, adapted to 3D objects. To reproduce this, just set modifiedTrainingMode to False in DataModuleConfig.
* **modifiedNoNoiseWay** which consists in using a cropped/cut-out damage at the spot of the simple type of damage. To reproduce this, set modifiedTrainingMode to True and randomNoise to False.
* **modifiedWay** same as modifiedNoNoiseWay but using also a random substitution of cells with random values to let the model persist/recover even in very strange states. To reproduce this, set modifiedTrainingMode and randomNoise to True.
