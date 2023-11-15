# WiTraPresGAN

This repository is the codebase for the scientific paper: TimeGAN for Data-Driven AI in high dimensional Industrial Data

The code is a quasi fork of: https://github.com/birdx0810/timegan-pytorch and features an implementation of TimeGAN (Yoon et al., NIPS2019)

Abstract:
The availability of historical process data in predictive maintenance is often insufficient to train complex machine learning models. To address this issue, techniques for data augmentation and synthesis have been developed, including the use of Generative Adversarial Networks (GANs). In this paper, the authors apply the GAN-based approach to synthesize simulated time series data. Experiments are carried out to find a trade-off between the amount of labeled data needed and the accuracy of the synthetic data for downstream tasks. The authors find that using 40\,\% of the original data for training the GAN results in synthetic data that contains the same information for downstream tasks as the original data, leading to an estimated speedup of ~60\,\% in the initial computing time. The results of the evaluation for the authors' own FEM simulation data, as well as for the Tennessee-Eastman benchmark dataset, are presented, demonstrating the potential of GANs in reducing time and energy in process development, while additionally interpolating a fixed parameter grid that is used for simulation purposes.

The fulltext can be found under: [insert Link here]
