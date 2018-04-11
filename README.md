Gated Recurrent Convolution Neural Network for OCR
======================================

This project is an implementation of the GRCNN for OCR. For details, please refer to the paper: https://papers.nips.cc/paper/6637-gated-recurrent-convolution-neural-network-for-ocr.pdf



Build
-----

The GRCNN is built upon the CRNN. The requirements are:

1. Ubuntu 14.04
2. CUDA 7.5
3. CUDNN 5

For the convenience of compiling, we provide the dependencies from here:
https://pan.baidu.com/s/1c21zl1e#list/path=%2F

It is more convenient if you use nivdia-docker image (@rremani supplied) : https://hub.docker.com/r/rremani/cuda_crnn_torch/

After installing the dependencies, go to ``src/`` and execute `` build_cpp.sh`` to build the C++ code. If successful, a file named ``libcrnn.so`` should be produced in the ``src/`` directory.


Inference
--------

We provide the pretrained model from [here](https://pan.baidu.com/s/1c21zl1e#list/path=%2F). Put the downloaded model file into directory ``model/GRCL/``. Moreover, we provide the IC03 dataset in the "./data/IC03" directory. You need to change the directories listed in the "test.txt". The "test_label.txt" is the ground truth of each image. The "lexicon_50.txt" is the lexicon of IC03. 

"src/evaluation.lua": Lexicon-free evaluation

"src/evaluation_lex.lua" Lexicon-based evaluation

The evaluation code will output the recognition accuracy.


Train a new model
-----------------

Follow the following steps to train a new model on your own dataset.

  1. Create a new LMDB dataset.`` src/create_own_dataset.py ``(need to ``pip install lmdb`` first).
  2. You can modify the configuration in ``model/GRCL/GRCL_LSTM_pretrain.lua``
  3. Go to ``src/`` and execute ``th main_train.lua ../model/GRCL/ ../model/saved_model``. Model snapshots will be saved into ``../model/saved_model``.



Citation
--------

@inproceedings{jianfeng2017deep, \<br> 
        title={Gated Recurrent Convolution Neural Network for OCR}, \<br> 
        author={Wang, Jianfeng and Hu, Xiaolin}, \<br> 
        booktitle={Advances in Neural Information Processing Systems}, \<br>        
        year={2017} \<br> 
       }
    
To-Do
----------------

Caffe implementation: Since the CRNN project is a little bit out-of-date, it might cannot compatible with Ubuntu 16.04. We will further implement a caffe version of GRCNN for OCR.

