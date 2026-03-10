<div align="center">

<h2><center>[CVPR 2026] Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision</h2>

Amadou S. Sangare, Adrien Maglo, Mohamed Chaouch, Bertrand Luvison

<br>

<a href='/'><img src='https://img.shields.io/badge/ArXiv-2407.21705-red'></a>
</div>

This is the official repository for paper "Improving Controllable Generation: Faster Training and Better Performance via $x_0$-Supervision".

## Abstract
Text-to-Image (T2I) diffusion/flow models have recently achieved remarkable progress in visual fidelity and text alignment However, they remain limited when users need to precisely control image layouts, something that natural language alone cannot reliably express. Controllable generation methods augment the initial T2I model with additional conditions that more easily describe the scene. Prior works straightforwardly train the augmented network with the same loss as the initial network. Although natural at first glance, this can lead to very long training times in some cases before convergence. In this work, we revisit the training objective of controllable diffusion models through a detailed analysis of their denoising dynamics. We show that direct supervision on the clean target image, dubbed $x_0$-supervision, or an equivalent re-weighting of the diffusion loss, yields faster convergence. Experiments on multiple control settings demonstrate that our formulation accelerates convergence by up to 2x according to our novel metric (mean Area Under the Convergence Curve - mAUCC), while also improving both visual quality and conditioning accuracy.

## Code
Coming soon ⏳