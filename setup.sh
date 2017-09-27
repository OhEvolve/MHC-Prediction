#!/bin/bash

# This is a setup script for Tensorflow setup 
# Designed for Ubuntu 16.04

sudo su

### Install CUDA ###

echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda; then
  # The 16.04 installer works with 16.10.
  curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  dpkg -i ./cuda-repo-ubuntu1604_8.0.61-1_amd64.deb
  apt-get update
  apt-get install cuda -y
fi
# note: can use nvidia-smi command to check CUDA install 

# Create environmental variables for CUDA 

echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDA_HOME/lib64' >> ~/.bashrc
source ~/.bashrc

### Install cudNN ###

cd $HOME
tar xzvf cudnn-8.0-linux-x64-v5.1.tgz
sudo cp cuda/lib64/* /usr/local/cuda/lib64/
sudo cp cuda/include/cudnn.h /usr/local/cuda/include/
rm -rf ~/cuda
rm cudnn-8.0-linux-x64-v5.1.tgz

### Install Tensorflow ### 

sudo apt-get install python-dev python-pip libcupti-dev
sudo pip install tensorflow-gpu

### Test install with Python ### 
python test_gpu.py

