<div align="center">

<h2><center>[CVPR 2026] Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision</h2>

Amadou S. Sangare, Adrien Maglo, Mohamed Chaouch, Bertrand Luvison
<br>
Université Paris-Saclay, CEA, List, F-91120, Palaiseau, France
<br>

<a href='https://arxiv.org/abs/2604.05761'><img src='https://img.shields.io/badge/ArXiv-2604.05761-red'></a>
</div>

This is the official repository for paper "Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision".

## Abstract
Text-to-Image (T2I) diffusion/flow models have recently achieved remarkable progress in visual fidelity and text alignment However, they remain limited when users need to precisely control image layouts, something that natural language alone cannot reliably express. Controllable generation methods augment the initial T2I model with additional conditions that more easily describe the scene. Prior works straightforwardly train the augmented network with the same loss as the initial network. Although natural at first glance, this can lead to very long training times in some cases before convergence. In this work, we revisit the training objective of controllable diffusion models through a detailed analysis of their denoising dynamics. We show that direct supervision on the clean target image, dubbed $x_0$-supervision, or an equivalent re-weighting of the diffusion loss, yields faster convergence. Experiments on multiple control settings demonstrate that our formulation accelerates convergence by up to 2x according to our novel metric (mean Area Under the Convergence Curve - mAUCC), while also improving both visual quality and conditioning accuracy.

## Code
We used [uv](https://docs.astral.sh/uv) to set up project-specific environments. If not installed, we recommend to install it for a seamless installation and usage of this project.

Read the followings for model-specific instructions:
- [ControlNet](./ControlNet/README.md)
- [T2I-Adapter](./T2I-Adapter/README.md)
- [OminiControl](./OminiControl/README.md)
- [GLIGEN](./GLIGEN/README.md)

# Citation
If you find our work and code useful, please cite us:

    @InProceedings{Sangare_2026_CVPR,
        author    = {Sangare, Amadou S. and Maglo, Adrien and Chaouch, Mohamed and Luvison, Bertrand},
        title     = {Improving Controllable Generation: Faster Training and Better Performance via x0-Supervision},
        booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
        month     = {June},
        year      = {2026},
        pages     = {9106-9115}
    }