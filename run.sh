#! /usr/bin/env bash

jetson-containers run \
    -v ~/cache:/model-cache \
    --device=/dev/ttyUSB0 \
    cascade-pipeline:r36.3.0 $@

    #-v /dev/ttyUSB0:/dev/ttyUSB0 \
