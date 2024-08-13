# <p align="center">World-Model-PyTorch</p>
<p align="center">--The simplified implementation of <a href="https://proceedings.neurips.cc/paper/2018/hash/2de5d16682c3c35007e4e92982f1a2ba-Abstract.html">World Model</a> based on PyTorch--</p>

## 1. Data Generate
Run `generate_CarRacing_dataset.py` to randomly generate data.

We generated a total of **200** trajectories, with **30** steps executed each time, resulting in a total of **6000** data. Each trajectory is saved separately as a `.npz` file.

Scale the observed image to a uniform size of **64 $\times$ 64**.

In the early stage of the car's movement, we will apply an **additional speed** to make it move as much as possible and collect richer data.

## 2. Train VAE
<div align=center>
<img src="demo/vae_1.png" width="400px">
</div>
<div align=center>
<img src="demo/vae_2.png" width="400px">
</div>
