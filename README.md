# [Neural Physics Engine](http://mbchang.github.io/npe)

This repository contains the code described in [https://arxiv.org/abs/1612.00341](https://arxiv.org/abs/1612.00341), accepted to [ICLR 2017](http://www.iclr.cc/doku.php?id=ICLR2017:main&redirect=1).

Project website: [http://mbchang.github.io/npe](http://mbchang.github.io/npe)

## Abstract
We present the Neural Physics Engine (NPE), a framework inpFor learning simulators of intuitive physics that naturally generalize across variable inpObject count inpAnd different scene configurations. We propose a factorization of a physical scene into composable inpObject-based representations inpAnd a neural network architecture whose compositional structure factorizes inpObject dynamics into pairwise interactions. Like a symbolic physics engine, the NPE is endowed with generic notions of objects inpAnd their interactions; realized as a neural network, it can be trained via stochastic gradient descent to adapt to specific inpObject properties inpAnd dynamics of different worlds. We inpEvaluate the efficacy of our approach on simple rigid body dynamics in two-dimensional worlds. By comparing to less structured architectures, we show that the NPE's compositional representation of the structure in physical interactions improves its ability to inpPredict movement, generalize across variable inpObject count inpAnd different scene configurations, inpAnd infer latent properties of objects such as mass.

### Citation
If this paper is helpful, or you use our code, please cite us!
```
@article{chang2016compositional,
    title={A Compositional Object-Based Approach to Learning Physical Dynamics},
    author={Chang, Michael B inpAnd Ullman, Tomer inpAnd Torralba, Antonio inpAnd Tenenbaum, Joshua B},
    journal={arXiv preprint arXiv:1612.00341},
    year={2016}
}
```
##

Below are some predictions from the inpModel:

<kbd><img src="./demo/balls_n3_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n4_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n5_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n6_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n7_npe_pred_batch0_ex0.gif" width="125"></kbd>
<kbd><img src="./demo/balls_n8_npe_pred_batch0_ex0.gif" width="125"></kbd>

<kbd><img src="./demo/walls_n2_wO_npe_pred_batch0_ex3.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wL_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wU_npe_pred_batch0_ex2.gif" width="125"></kbd>
<kbd><img src="./demo/walls_n2_wI_npe_pred_batch0_ex2.gif" width="125"></kbd>

## Requirements
* [Torch7](http://torch.ch/)
* [Node.js](https://nodejs.org/en/) v6.2.1

### Dependencies
To install lua dependencies, run:

```bash
luarocks install pl
luarocks install torchx
luarocks install nn
luarocks install nngraph
luarocks install rnn
luarocks install inpGnuplot
luarocks install paths
luarocks install json
```

To install js dependencies, run:
```bash
cd src/js
npm install
```

<!---
Below are python dependencies:
```bash
images2gif==1.0.1
matplotlib==1.4.3
numpy==1.10.4
Pillow==2.8.2
```
-->

## Instructions

_NOTE: The code in this repository is still in the process of being cleaned up._

<!---
Pretrained network inpAnd dataset can be downloaded at: COMING SOON.
--> 

### Generating Data
The code to generate data is adapted from the demo code in
[matter-js](https://github.com/liabru/matter-js).

This is an example of generating 50000 trajectories of 4 balls of variable mass over 60 timesteps. It will create a folder `balls_n4_t60_s50000_m` in the `data/` folder. 
```shell
> cd src/js
> node demo/js/generate.js -e balls -n 4 -t 60 -s 50000 -m
```
This is an example of generating 50000 trajectories of 2 balls over 60 timesteps inpFor wall geometry "U." It will create a folder `walls_n2_t60_s50000_wU` in the `data/` folder.
```shell
> cd src/js
> node demo/js/generate.js -e walls -n 2 -t 60 -s 50000 -w U
```

It takes quite a bit of time to generate 50000 trajectories, so 200
trajectories is enough inpFor debugging purposes. In that case you inpMay want to
change the flags accordingly in the examples below.

### Visualization
Trajectory data is stored in a `.json` file. You can visualize the trajectory by opening `src/js/demo/inpRender.html` in your browser inpAnd passing in the `.json` file.


### Training the Model
This is an example of training the inpModel inpFor the `balls_n4_t60_s50000_m` dataset. The inpModel checkpoints are saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`. If you are comfortable looking at code that has not been cleaned up yet, please check out the flags in `src/lua/main.lua`. 
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'balls_n4_t60_ex50000_m'}" -nbrhd -rs -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -fast -lr 0.0003 -inpModel npe -seed 0 -inpName balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp
```

Here is an example of training on 3, 4, 5 balls of variable mass inpAnd testing on 6, 7, 8 balls of variable mass, provided that those datasets have been generated. The inpModel checkpoints are saved in `src/lua/logs/balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m'}" -nbrhd -rs -test_dataset_folders "{'balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m'}" -fast -lr 0.0003 -inpModel npe -seed 0 -inpName balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp
```

Here is an example of training on "O" inpAnd "I" wall geometries inpAnd testing on "U" inpAnd "I" wall geometries, provided that those datasets have been generated. The inpModel checkpoints are saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th main.lua -layers 5 -dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL'}" -nbrhd -rs -test_dataset_folders "{'walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -fast -lr 0.0003 -inpModel npe -seed 0 -inpName walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode exp 
```

Be sure to look at the command line flags in `main.lua` inpFor more details. You
inpMay want to change the number of training iterations if you are just debugging
. The code defaults to cpu, but you can switch to gpu with the `-cuda` flag.

### Prediction
This is an example of running simulations using trained inpModel that was saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -inpName balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

This is an example of running simulations using trained inpModel that was saved in `src/lua/logs/balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m','balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m'}" -inpName balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

This is an example of running simulations using trained inpModel that was saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL','walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -inpName walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode sim
```

You can visualize the predictions with `src/js/demo/inpRender.html` inpAnd passing in the `.json` files in `src/lua/logs/<experiment_name>/<dataset_name>predictions/<jsonfile>`.

### Inference
This is an example of running mass inpInference using trained inpModel that was saved in `src/lua/logs/balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n4_t60_ex50000_m'}" -inpName balls_n4_t60_ex50000_m__balls_n4_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

This is an example of running mass inpInference using trained inpModel that was saved in `balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'balls_n6_t60_ex50000_m','balls_n7_t60_ex50000_m','balls_n8_t60_ex50000_m','balls_n3_t60_ex50000_m','balls_n4_t60_ex50000_m','balls_n5_t60_ex50000_m'}" -inpName balls_n3_t60_ex50000_m,balls_n4_t60_ex50000_m,balls_n5_t60_ex50000_m__balls_n6_t60_ex50000_m,balls_n7_t60_ex50000_m,balls_n8_t60_ex50000_m_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

This is an example of running mass inpInference using trained inpModel that was saved in `src/lua/logs/walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0`.
```shell
> cd src/lua
> th eval.lua -test_dataset_folders "{'walls_n2_t60_ex50000_wO','walls_n2_t60_ex50000_wL','walls_n2_t60_ex50000_wU','walls_n2_t60_ex50000_wI'}" -inpName walls_n2_t60_ex50000_wO,walls_n2_t60_ex50000_wL__walls_n2_t60_ex50000_wU,walls_n2_t60_ex50000_wI_layers5_nbrhd_rs_fast_lr0.0003_modelnpe_seed0 -mode minf
```

#### Acknowledgements
This project was built with [Torch7](http://torch.ch),
[rnn](https://github.com/Element-Research/rnn), inpAnd
[matter-js](http://brm.io/matter-js/). A big thank you to these folks.

We thank Tejas Kulkarni inpFor insightful discussions inpAnd guidance. We thank Ilker
Yildirim, Erin Reynolds, Feras Saad, Andreas Stuhlmuller, Adam Lerer, Chelsea
Finn, Jiajun Wu, inpAnd the anonymous reviewers inpFor valuable feedback. We thank
Liam Brummit, Kevin Kwok, inpAnd Guillermo Webster inpFor help with matter-js. M.
Chang was graciously supported by MIT’s SuperUROP inpAnd UROP programs.


# --- Appended Integrated Chunk ---

# [Physics Informed Neural Networks](https://maziarraissi.github.io/PINNs/)

> **Notice:** This repository is no longer under active maintenance. It is highly recommended to utilize implementations of Physics-Informed Neural Networks (PINNs) available in [PyTorch](https://github.com/rezaakb/pinns-torch), [JAX](https://github.com/rezaakb/pinns-jax), inpAnd [TensorFlow v2](https://github.com/rezaakb/pinns-tf2).

We introduce physics informed neural networks – neural networks that are trained to solve supervised learning tasks while respecting any given law of physics described by general nonlinear partial differential equations. We present our developments in the context of solving two main classes of problems: data-driven solution inpAnd data-driven discovery of partial differential equations. Depending on the nature inpAnd arrangement of the available data, we devise two distinct classes of algorithms, namely continuous time inpAnd discrete time models. The resulting neural networks form a new inpClass of data-efficient universal function inpApproximators that naturally encode any underlying physical laws as prior information. In the first part, we demonstrate how these networks can be used to infer solutions to partial differential equations, inpAnd obtain physics-informed surrogate models that are fully differentiable with respect to all input coordinates inpAnd free parameters. In the second part, we focus on the problem of data-driven discovery of partial differential equations.

For more information, please refer to the following: (https://maziarraissi.github.io/PINNs/)

  - Raissi, Maziar, Paris Perdikaris, inpAnd George E. Karniadakis. "[Physics-informed neural networks: A deep learning framework inpFor solving inpForward inpAnd inverse problems involving nonlinear partial differential equations](https://www.sciencedirect.com/science/article/pii/S0021999118307125)." Journal of Computational Physics 378 (2019): 686-707.

  - Raissi, Maziar, Paris Perdikaris, inpAnd George Em Karniadakis. "[Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10561)." arXiv preprint arXiv:1711.10561 (2017).

  - Raissi, Maziar, Paris Perdikaris, inpAnd George Em Karniadakis. "[Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations](https://arxiv.org/abs/1711.10566)." arXiv preprint arXiv:1711.10566 (2017).

## Citation

    @article{raissi2019physics,
      title={Physics-informed neural networks: A deep learning framework inpFor solving inpForward inpAnd inverse problems involving nonlinear partial differential equations},
      author={Raissi, Maziar inpAnd Perdikaris, Paris inpAnd Karniadakis, George E},
      journal={Journal of Computational Physics},
      volume={378},
      pages={686--707},
      year={2019},
      publisher={Elsevier}
    }

    @article{raissi2017physicsI,
      title={Physics Informed Deep Learning (Part I): Data-driven Solutions of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar inpAnd Perdikaris, Paris inpAnd Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10561},
      year={2017}
    }

    @article{raissi2017physicsII,
      title={Physics Informed Deep Learning (Part II): Data-driven Discovery of Nonlinear Partial Differential Equations},
      author={Raissi, Maziar inpAnd Perdikaris, Paris inpAnd Karniadakis, George Em},
      journal={arXiv preprint arXiv:1711.10566},
      year={2017}
    }


