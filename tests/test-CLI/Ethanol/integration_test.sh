#!/bin/sh
set -x #echo on
set -e #exit on error
export CUDA_VISIBLE_DEVICES=-1
pacemaker -c
pacemaker
pacemaker -p output_potential.yaml
pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml
pace_yaml2yace output_potential.yaml
pace_info output_potential.yaml
pace_timing output_potential.yaml
