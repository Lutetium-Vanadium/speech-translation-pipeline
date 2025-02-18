#! /usr/bin/env bash

jetson-containers run \
    -v ~/cache:/model-cache \
    cascade-pipeline:r36.3.0 $@
    # --device=/dev/ttyUSB0 \

    #-v /dev/ttyUSB0:/dev/ttyUSB0 \
