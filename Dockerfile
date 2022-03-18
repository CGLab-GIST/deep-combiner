# migrated version from TF1 to TF2
FROM tensorflow/tensorflow:2.5.0-gpu

# Install openexr for python
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    python3-pip \
	libopenexr-dev \
	zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*
RUN git clone https://github.com/jamesbowman/openexrpython.git
RUN pip3 install openEXR==1.3.0
WORKDIR /openexrpython
RUN python3 setup.py install

# Running codes
VOLUME /data
VOLUME /codes
WORKDIR /codes
CMD python3 tester.py