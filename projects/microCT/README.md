# Region Segmentation from microCT Scanning

Introduction
-------------

Indirect flight muscles (IFMs) in adult Drosophila provide the key power stroke for wing beating. They also serve as a valuable model for studying muscle
development. Such analyses are impeded by conventional histological
preparations and imaging techniques that limit exact morphometry of flight
muscles. In this tutorial, microCT scanning is employed on a tissue preparation
that retains muscle morphology under homeostatic conditions. Focusing on
a subset of IFMs called the dorsal longitudinal muscles (DLMs), it is found that
DLM volumes increase with age, partially due to the increased separation
between myofibrillar fascicles, in a sex-dependentmanner.The authors have uncovered
and quantified asymmetry in the size of these muscles on either side of the
longitudinal midline.Measurements of this resolution and scalemake substantive
studies that test the connection between form and function possible.

In this tutorial, you will learn how to predict the **Drosophila longitudinal muscles instance masks** on the CT Fly
dataset released by Chaturvedi, et al. in 2019.

Volumetric Instance Segmentation
----------------------

This section provides step-by-step guidance for CT Fly segmentation with the dataset released by Chaturvedi et al. 
We consider the task as 3D **instance segmentation** and predict the Drosophila longitudinal muscles instances with encoder-decoder ConvNets ``unet_res_3d``, similar to the one used in [neuron segmentation] (https://zudi-lin.github.io/pytorch_connectomics/build/html/tutorials/snemi.html).

The evaluation of the segmentation results is based on the F1-score.

    The dataset released by Chaturvedi et al. is completely different from other EM connectomics datasets used in the tutorials, 
    where we downsample the volumes to (112,112,112) to capture the whole feild of view instead of patches of volumes.
    Therefore a completely different Dataloader and preprocessing steps are preferred.

All the scripts needed for this tutorial can be found at ``pytorch_connectomics/projects/microCT``. Need to pass the argument ``--config-file configs/projects/microCT/configs/CT-Fly-No-Augmentation.yaml`` during training and inference to load the required configurations for this task. 


#. Get the dataset:

    Download the dataset from our server:

        
            http://rhoana.rc.fas.harvard.edu/dataset/
    

#. Run the training script:


        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -u projects/microCT/main.py \
          --config-file projects/microCT/configs/CT-Fly-No-Augmentation.yaml

#. Visualize the training progress:


        $ tensorboard --logdir runs

#. Run inference on test image volume:


        $ source activate py3_torch
        $ CUDA_VISIBLE_DEVICES=0,1,2,3 python -u projects/microCT/main.py \
          --config-file projects/microCT/configs/CT-Fly-No-Augmentation.yaml --inference \
          --checkpoint outputs/CT_Fly/volume_100000.pth.tar

Our pretained model achieves a F1 score of **0.89** on the test set.