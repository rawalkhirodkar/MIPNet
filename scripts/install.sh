source ~/anaconda3/etc/profile.d/conda.sh

## cd to root
cd ..

conda create -n mipnet python=3.8 -y
conda activate mipnet
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch -y
pip install Cython h5py
pip install -r requirements.txt
