# Start
This code has NOT been written by me. I dont't take credit for this. 
All credits goes to Marco Cristani teams at University of Verona. 
## To extract the data:

Requirement:
 - habitat-lab
 - habitat-sim
 - matterport data


Descr:
This code will extract the RGB,DEPTH, and Semantic segmentation from the scene specified in the python source.
The name of the images are in the form of *x_y_z_rx_r_y_rz.[png, npy]*

1. `cd Data_Extraction`

2. Change the path to the data in line 43 of **img_extractor.py**
2. `python img_extractor.py`


____
## Reconstruction from rgb-d and semantic segmentation

This will extract a wall.mat file with the points of the extracted walls, using the rbg,depth and semantic information

Requirement:
 - open-3d (this library change a lot from version to version. could work on 0.8.0 or 0.9.0)

Run
1. `cd habitat_reconstruction`
2. change path in line 247 of **scene_reconstrucion.py**
3.`python scene_reconstruction.py`
