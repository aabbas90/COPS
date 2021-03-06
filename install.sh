conda install -y pytorch torchvision cudatoolkit=10.1 -c pytorch
pip install opencv-python
pip install -U pandas
pip install scipy
pip install scikit-image

# Install dependencies:
python -m pip install depends/panopticapi/
python -m pip install depends/cityscapesScripts/

# Install detectron2:
python -m pip install depends/detectron2

# Install AMWC, MWC solvers.
PACKAGES="mc" python3 -m pip install depends/LPMP