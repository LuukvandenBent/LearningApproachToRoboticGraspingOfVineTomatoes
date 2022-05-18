#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
@author: taeke
"""

def compute_grasp(grasp_length_min_px=30.0,
                  grasp_length_min_mm=15.0,
                  finger_thinkness_mm=10.0,
                  open_dist_mm=2*(0.037 - 0.0195)*1000):

    settings = {'grasp_length_min_px': grasp_length_min_px,
                'grasp_length_min_mm': grasp_length_min_mm,
                'finger_thinkness_mm': finger_thinkness_mm,
                'open_dist_mm': open_dist_mm}

    return settings

def detect_peduncle(branch_length_min_px=15,
                    branch_length_min_mm=10):

    settings = {'branch_length_min_px': branch_length_min_px,
                'branch_length_min_mm': branch_length_min_mm}
    return settings


def detect_tomato(blur_size=3,
                  radius_min_frac=8,
                  radius_max_frac=6,
                  distance_min_frac=4,  
                  big_radius_min_mm=20,
                  big_radius_max_mm=35,
                  small_radius_min_mm=15,
                  small_radius_max_mm=23,
                  dp=4,
                  param1=20,
                  param2=80,
                  ratio_threshold=0.4,
                  ratio_max_dist=2):

    settings = {'radius_min_frac': radius_min_frac,
                'radius_max_frac': radius_max_frac,
                'distance_min_frac': distance_min_frac,
                'big_radius_min_mm': big_radius_min_mm,
                'big_radius_max_mm': big_radius_max_mm,
                'small_radius_min_mm': small_radius_min_mm,
                'small_radius_max_mm': small_radius_max_mm,
                'dp': dp,
                'param1': param1,
                'param2': param2,
                'ratio_threshold': ratio_threshold,
                'blur_size': blur_size,
                'ratio_max_dist': ratio_max_dist}

    return settings


def filter_segments(filter_diameter=3):
    settings = {'filter_diameter': filter_diameter}
    return settings


def segment_image(f=2,
                  hue_radius=1.5,
                  epsilon=0.01,
                  i_max=20):

    settings = {'f': f,
                'hue_radius': hue_radius,
                'epsilon': epsilon,
                'i_max': i_max}
    return settings


def initialize_all():

    settings = {'segment_image': segment_image(),
                'detect_tomato': detect_tomato(),
                'detect_peduncle': detect_peduncle(),
                'compute_grasp': compute_grasp(),
                'filter_segments': filter_segments()}

    return settings
