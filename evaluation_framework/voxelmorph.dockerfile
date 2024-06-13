FROM tensorflow/tensorflow:2.10.0-gpu

# switch to root to install packages
USER root

RUN apt-get update && apt-get install -y \
    python3.9 python3.9-venv\
    libx11-dev libgl1-mesa-glx libglu1-mesa libxt6 libxrender1 libsm6 libice6 libgtk2.0-0 libosmesa6 libgl2ps-dev \
    && rm -rf /var/lib/apt/lists/*

# Yes this image already contains tensorflow 2.10, but it has it on
# python 3.8 and we need 3.9 for the evaluation script to work,
# so we install it again.
RUN python3.9 -m pip install --upgrade pip
RUN python3.9 -m pip install tensorflow==2.10 voxelmorph antspyx vtk

RUN useradd -ms /bin/bash bmet

USER bmet

# Set the working directory
WORKDIR /workspace

# Ensure the necessary volume mount points exist
VOLUME ["/workspace"]

# Default command to run
CMD ["bash"]