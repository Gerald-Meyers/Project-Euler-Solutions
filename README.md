# Dev Container NVIDIA based

This is an example VSCode devcontainer that successfully builds and has access to the GPU using tensorflow. Some additional python modules are included mostly for arbitrary reasons. 
The previous branch this came from was not particularly optimized to use the docker image building and caching service, and ran all installation steps after starting the image.
This resulted in significant overhead and run-time with each start.

## Prerequisites
- Docker engine (and setup .wslconfig to use more cores and memory than default)
- NVIDIA driver for the graphic card
- NVIDIA Container Toolkit (which is already included in Windows’ Docker Desktop; Linux users have to install it)
- VS Code with DevContainer extension installed

## Start the DevContainer
- Clone this repo.
- In VS Code press `Ctrl + Shift + P` to bring up the Command Palette. 
- Enter and find `Dev Containers: Reopen in Container`. 
- VS Code will starts to download the base image, building the docker image, install everything specified in the `dev.Dockerfile` & `requirements.txt`, and finish opening the directory in DevContainer.
- The DevContainer would then run nvidia-smi to show what GPU can be seen by the container. Be noted that this works even without setting up cuDNN or any environment variables.

## Test with keras script for MNIST
The file `./src/train.py` is a short AutoKeras test script for you, which trains with the MNIST handwriting digit dataset with a pre-defined CNN model.
Open a new terminal and enter:
```bash
 python3 src/autokeras_script.py
``` 


## Setup details
### Dev Container definition
DevContainer definition `.devcontainer/devcontainer.json` uses the official tensorflow developer image `tensorflow/tensorflow:2.16.1-gpu` (not base or runtime), which supports AMD64 and ARM64 and has CUDA installed. 

### Installing basic Linux tools, Python 3, Python packages and cuDNN
This is the original script for installing basic based on the original nvidia image. The file is mostly empty but included for the sake of posterity: `.devcontainer/install-dev-tools.sh`. 

```bash
# update system
apt-get update
apt-get upgrade -y
# install Linux tools and Python 3
apt-get install software-properties-common wget curl \
    python3-dev python3-pip python3-wheel python3-setuptools -y
# install Python packages
python3 -m pip install --upgrade pip
pip3 install --user -r .devcontainer/requirements.txt
# update CUDA Linux GPG repository key
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
dpkg -i cuda-keyring_1.0-1_all.deb
rm cuda-keyring_1.0-1_all.deb
# install cuDNN
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub
add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" -y
apt-get update
apt-get install libcudnn8=8.9.0.*-1+cuda11.8
apt-get install libcudnn8-dev=8.9.0.*-1+cuda11.8
# install recommended packages
apt-get install zlib1g g++ freeglut3-dev \
    libx11-dev libxmu-dev libxi-dev libglu1-mesa libglu1-mesa-dev libfreeimage-dev -y
# clean up
pip3 cache purge
apt-get autoremove -y
apt-get clean
```

### Third party Python packages
The file `.devcontainer/requirements.txt` contains all third party Python packages you wish to install. Modify the list as you like.

```
tensorflow==2.16.1
pandas
numpy
scipy
matplotlib
autokeras
ipykernel
scikit-learn
regex
```


**Source**: [Setup a NVIDIA DevContainer with GPU Support for Tensorflow/Keras on Windows](https://alankrantas.medium.com/setup-a-nvidia-devcontainer-with-gpu-support-for-tensorflow-keras-on-windows-d00e6e204630)