FROM mirrors.tencent.com/nvs/nerf-pytorch
RUN python3 -m pip install --upgrade pip \
    && pip3 install scikit-video \
    && pip3 install h5py \
    && pip3 install turtle \
    && pip3 install cmapy