#!/bin/sh
set -x #echo on
set -e #exit on error
export CUDA_VISIBLE_DEVICES=-1
# clear files
pacemaker -c
# initial fit
pacemaker
# upfit
pacemaker -p output_potential.yaml
# active set
pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml
# data augmentation
pace_augment -d fitting_data_info.pckl.gzip output_potential.yaml -a output_potential.asi
# upfit
pacemaker input_aug.yaml -p output_potential.yaml
# active set
pace_activeset -d fitting_data_info.pckl.gzip output_potential.yaml
# auto core-rep
pace_corerep output_potential.yaml -a output_potential.asi -d fitting_data_info.pckl.gzip
# utilities
pace_yaml2yace output_potential.yaml
pace_info output_potential.yaml
pace_timing output_potential.yaml
