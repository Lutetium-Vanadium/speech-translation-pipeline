#! /usr/bin/env bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

jetson-containers build $@ --package-dirs $SCRIPT_DIR,$SCRIPT_DIR/../whisper_s2t cascade-pipeline
