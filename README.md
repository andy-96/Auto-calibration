# Camera Self-Calibration Research Project - 2020
@Contributors: Yunhai Han & Yuhan Liu


## Introduction
This research project focuses on (real-time) camera self-calibration in urban autonomous driving scenarios. Specifically, we experiment and analyze **intrinsic calibration** routines using only ordinary objects like **stop signs** and **license plates**.  

## Table of Contents
* [Experiments & Error Analysis - Winter 2020](https://github.com/y8han/Auto-calibration/blob/master/Overview/2020-2-20.md)  
**Abstract**: Experimented on varieties of chessboard calibration, analyzed the performances under various factors (e.g. number of points, number of frames, chessboard sizes, and magnitude of noises), and tried modeling the calibration error.  
**Comments**: Not practical to build an overall error model.

* [Feature Extraction & Line Estimation - Spring 2020](https://github.com/y8han/Auto-calibration/blob/master/Overview/2020-5-5.md)  
**Abstract**: Finding algorithms to extract feature points (with known 3D coordinates or physical sizes in world objects) from images, and using line estimation to attain sub-pixel accuracy.

* [Auto-calibraton Using Stop Signs - Summer 2020](https://github.com/y8han/Auto-calibration/blob/master/System/pipeline.png)  
**Abstract**: For intelligent vehicle applications, calibration is often an important component of sensor fusion, depth estimation and scene understanding. However, in many scenarios, the estimated calibration parameters can change over time as the result of temperature and vibrations. For this reason, we are actively developing tools and methods that leverage road furniture and geometric shapes such as stop signs to dynamically calibrate our cameras on board of our vehicles in real-time.

## Paper
[Auto-calibration Method Using Stop Signs for Urban Autonomous Driving Applications](https://arxiv.org/abs/2010.07441)  

## References
[1] Pellejero O.A., Sagüés C., Guerrero J.J. (2004) [Automatic Computation of the Fundamental Matrix from Matched Lines](https://webdiis.unizar.es/~csagues/english/publicaciones/03CAEPIA-cp.pdf). In: Conejo R., Urretavizcaya M., Pérez-de-la-Cruz JL. (eds) Current Topics in Artificial Intelligence. TTIA 2003. Lecture Notes in Computer Science, vol 3040.

[2] Faugeras O.D., Luong Q.T., Maybank S.J. (1992) [Camera self-calibration: Theory and experiments](https://link.springer.com/content/pdf/10.1007/3-540-55426-2_37.pdf). In: Sandini G. (eds) Computer Vision — ECCV'92. ECCV 1992. Lecture Notes in Computer Science, vol 588. 
