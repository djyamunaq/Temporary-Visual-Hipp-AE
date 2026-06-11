import numpy as np
import matplotlib.patches as patches
import matplotlib.pyplot as plt

import os

def get_view(scene, field, eye_position):
    "Crops the scene to the field view around the eye position"
    return scene[
        int(eye_position[0] - field[0]/2.):int(eye_position[0] + field[0]/2.), 
        int(eye_position[1] - field[1]/2.):int(eye_position[1] + field[1]/2.), 
        :]

def apply_saccade(eye_position, displacement, field, scaling):
    "Applies the displacement to the eye position, knowing the visual field size, and the FEF size."
    print('Previous eye position:', eye_position)
    print('Coordinates in FEF:', displacement)
    saccade = (displacement - scaling/2)*field/scaling
    print('Saccade direction:', saccade)
    new_eye_position = eye_position + saccade
    print('New eye position:', new_eye_position)
    return new_eye_position
