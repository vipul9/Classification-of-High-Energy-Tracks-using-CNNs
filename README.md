# Classification High Energy Particles Using Convolutional Neural Networks
## Overview
This is the code for training and testing on image data for classification of electrons in a high energy physics (HEP) experiment.
## The Large Hadron Collider
The Large Hadron Collider (LHC) is the biggest and most powerful particle accelerator. It consists of a 27 km ring of superconducting magnets. In the accelerator, two high energy particle beams (proton beams) traveling in nearly the speed of lights are made to collide. Beams are made to go in the opposite directions in separate beam pipes. Beams are guided along the accelerator by a strong magnetic field produced using superconducting electromagnets. The collisions happen at four locations. At the collision points, the energy of particle collisions is transformed into mass resulting in the formation of several particles. Four detectors are placed at these collision points to detect this spray of particles. The biggest of these detectors, CMS and ATLAS, are general purpose detectors. ALICE and LHCb are detectors designed for the search of specific phenomena.
## CMS
Compact Muon Solenoid or CMS is a general-purpose detector designed to observe any new physics phenomena. Though most particles produced by high energy collisions at the LHC are unstable, they decay to stable particles which can be detected by CMS. By identifying these stable particles, their momenta and energy, the detector can recreate a model of the collision. It acts as a high-speed camera, taking 3D photographs of particle collisions.

![cms-event](https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/pp_collision.gif)

## Particle Reconstruction
The interaction of the particle with the different layers of the detector is used to reconstruct the nature and properties of the particle. The silicon tracker tracks particles near the collision region. The calorimeter stops some particles and measures their energy. The magnet bends the particle allowing for momentum measurement. Muon chambers detect muons.

![cms-reconstruction](https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/particle_reco.gif)

This work uses convolutional neural networks to classify electrons using their signature in the calorimeters. Data from the CMS detector is used for this work.

## High Energy Tracks Classification
In proton-proton collision huge amount of data gets created. The collison rate is far more than ability to write data. Ideal case is to store all the data but it is not possible due to hardware limitations. To takle this problem, there is a trigger system in place. This trigger system help to chose. This process of choosing the data is called triggering. Trigers are energy sensitive. If produces in an event has energy above the threshold, then it keeps that event.

Moving charged particles bend in the presense of magnetic field. Tracker is a unit in detector that reconstruct the particle trajectory. Reconstruction of tracks is an crucial process because it gives valuable information about energy of particle and charge of the particle. High energy particles are seldom created in nature hence they make interesting point of study. High Energy particles tend to have straight line path in magnetic field. This work shows the novel way of triggering over the events with high energy tracks using Convolutional neural networks (CNNs). 

This work follows three models namely 2D toy tracker model, 3D toy tracker model and finally CMS tracker. 

# 2D Toy Tracker Model
## 100 Tracks case:
<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/100trk_100per_hpt.png" width="400"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/100trk_100per_lpt.png" width="400"/>

Note that the track in red is the high energy particle's track and colour is just for representation. Actual images used for training and testing of neural networks were grayscale images. One thing to notice here is that the points are not aligned. This is because we never get et point of intersection from detector due to observational error.

Below is the output histogram and ROC of a convolutional neural network.
<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/histo_100trk_100per.png" width="425"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/ROC_100trk_100per.png" width="425"/>

## 200 tracks case:
Total 200 tracks per event.  
<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/high_200trk_100per.png" width="425"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/low_200trk_100per.png" width="425"/>

Below is the result of the CNN. Behaviour of ROC curve is expected. As complexity goes up, accuracy goes down. 
<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/histo_200trk_100per.png" width="425"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/ROC_200trk_100per.png" width="425"/>

# Simulated Data From CMS 

Image of tracker after proton-proton collsion.

<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/combine_high.png" width="425"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/combine_low.png" width="425"/>

Below is the output histogram of a convolutional neural network after training and testing on 8000 electron images of both categories. (Signal is isolated electrons and background is non-isolated electrons.)

<img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/histo_brrl_exl.png" width="425"/> <img src="https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/Images/ROC_brrl_exl.png" width="425"/>

# Instructions

All the information used for creating the images, i.e., the spatial coordinates, and energy of particles come from CERN lxplus servers. Data is generated using CMS software (CMSSW). Sample code for this conversion can be found in CMSSW repository. Finally, these images become input to a convolutional neural network implemented using Tensorflow. The code for this is (https://github.com/vipul9/Classification-of-High-Energy-Tracks-using-CNNs/blob/master/cnn_model.py). 
