Requirements

Lasagne: https://github.com/Lasagne/Lasagne
Theano: http://deeplearning.net/software/theano/
NumPy, SciPy, Pandas, nolearn

Installed using pip. 
sudo pip install "packagename"

GPU specific operations: 
Install CUDA: https://developer.nvidia.com/cuda-toolkit
Install cuDNN https://developer.nvidia.com/cudnn

Make Theano use GPU: http://deeplearning.net/software/theano/library/config.html

The keypoints_detector_trainer.py builds the CNN and starts training on the training file provided. 
On a GPU the training for 1000 epochs will take about 30 minutes. 

The script will dump the weights and parameters for the NN to external files 

To predict keypoints on images, run the keypoint_detector.py with the generated files as arguments. 
The results would pe printed on the system.out which can be exported to a csv file used for phase 2. 
