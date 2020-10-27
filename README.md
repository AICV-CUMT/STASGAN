# STASGAN
# Semi-supervised Blockwisely Architecture Search for Efficient Lightweight Generative Adversarial Network

Byeongho Heo, Jeesoo Kim, Sangdoo Yun, Hyojin Park, Nojun Kwak, Jin Young Choi

Man Zhang; Yong Zhou; Jiaqi Zhao; Shixiong Xia; Jiaqi Wang; Zizheng Huang

| Here I present a baseline of the proposed method |


## 0. Environments
- Linux
- Cuda 9.0
- Tesla P100
- jetson nano


## 1. Requirements
- python
- tensorflow
- keras
- matplotlib
- scipy
- numpy
- pillow
- scikit-image


## 2. Train the model
- python sas_gan.py


## 3. Check the result
- The trained parameters can be found in the 'saved_model' file.
- The generated images can be found in the 'images' file.


## 4. Implementation on mobile device
- In our experiments, we used a nano to prove the effectiveness of the method. The implementation method is relatively simple. First, adjust the code to be accurate, then configure the nano environment to be consistent with the environment required for the experiment, and finally run the program on the nano device.
