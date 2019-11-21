"""Do some quick testing, get images and then test color detection.

This is necessary for calibration later. This script serves two purposes: (1)
to help get data from the camera, to get sample color images for usage later,
and (2) to tune the color detection code. We have the option of working with
the physical robot (requires us to be on the dvrk arm machine), or not.

(c) November 2019 by Daniel Seita
"""
import os
import cv2
import pickle
import sys
import json
import argparse
import numpy as np
from os.path import join
np.set_printoptions(suppress=True)
# Temporary hacks because we don't have this in a normal python package.
#sys.path.append('..')  # Allows: `python collect_calib_data.py
sys.path.append('.')   # Allows: `python calibration/collect_calib_data.py`

# Stuff from our code base.
try:
    import rospy
    import camera
    from dvrkClothSim import dvrkClothSim
except:
    print('NOTE: Cannot import rospy, assume we are on machine w/out access to robot')
import utils as U

# To change width, use img[:,width_min:width_max], etc.
Y_MIN = 50
Y_MAX = 500
X_MIN = 600
X_MAX = 1200
WHITE = (255,255,255)
RED   = (0,0,255)
GREEN = (0,255,0)
BLUE  = (255,0,0)


def _crop(img):
    return img[Y_MIN:Y_MAX, X_MIN:X_MAX]


def _save(img, path, key):
    pth = path.replace('.png','_{}.png'.format(key))
    cv2.imwrite(pth, img)


def _get_contour(image, path):
    """Given `image`, find the largest contour's center.

    We should think of this as being close to the wrist position of the robot
    w.r.t. the camera pixels --- and ideally the camera *frame* if we know what
    to do with the camera intrinsics.

    AH ... we need the contours that are within the bounds. But, we cropped the
    image. Therefore, we need to adjust the resulting x and y by adding the
    appropriate x and y that we used for cropping.
    """
    image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    image_bgr = image.copy()
    _save(image_bgr, path, '05_before_contour')

    # Note that contour detection requires a single channel image.
    (cnts, _) = cv2.findContours(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                                 cv2.RETR_TREE,
                                 cv2.CHAIN_APPROX_SIMPLE)
    contained_cnts = []

    # Find the centroids of the contours in _pixels_, of the cropped image.
    for c in cnts:
        try:
            M = cv2.moments(c)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # Enforce it to be within bounding box.
            #if (xx < cX < xx+ww) and (yy < cY < yy+hh):
            # TODO: may consider doing it, but easier just to take all.
            contained_cnts.append(c)
        except:
            pass

    # Go from `contained_cnts` to a target contour (largest) and process it.
    if len(contained_cnts) > 0:
        target_contour = sorted(contained_cnts, key=cv2.contourArea, reverse=True)[0] # Index 0
        try:
            M = cv2.moments(target_contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            peri = cv2.arcLength(target_contour, True)
            approx = cv2.approxPolyDP(target_contour, 0.02*peri, True)

            # Draw them on the `image_bgr` to return, for visualization purposes.
            cv2.circle(image_bgr, (cX,cY), 50, RED, thickness=1)
            cv2.drawContours(image_bgr, [approx], -1, GREEN, 2)
            cv2.putText(img=image_bgr,
                        text="{},{}".format(cX,cY),
                        org=(cX+10,cY+10),
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=1,
                        color=WHITE,
                        thickness=2)
            _save(image_bgr, path, '06_after_contour')
        except:
            pass
        return image_bgr, (cX,cY)

    else:
        # Failed to find _any_ contour.
        return image_bgr, (-1,-1)


def _detect_region(img, path):
    """Given image, detect the region of interest (e.g., tape or ball).

    Note that the image is probably BGR since we directly used `imread`.
    Should be originally of size: (1200, 1920, 3).
    """
    #thresh = _get_thresholded_image(img.copy())
    image = img.copy()

    image = _crop(image)
    _save(image, path, '00_crop')

    # Looks indistinguishable.
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    _save(image, path, '01_rgb')

    # Doesn't change much but can smooth the balls (if we're using those).
    image = cv2.bilateralFilter(image, 7, 13, 13)
    _save(image, path, '02_bilateral')

    # Makes a weird image, as usual.
    hsv   = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    _save(hsv, path, '03_hsv')

    # ------------------------------------------------------------------------ #
    # Seems like tuning `lower` produces more notable differences.
    # Increasing lower[0], lower[1] from 10 -> 50 gets rid of some background.
    # Higher values mean more background is removed, helps clean it up.
    # To keep only the reddish-color from the ball, deal with lower[0].
    # Going from lower[0]: 60 -> 70 got rid of blocks, and kept the ball!
    # Increasing upper to 255 doesn't seem to make much difference.
    # ------------------------------------------------------------------------ #
    lower = np.array([70, 50, 50])
    upper = np.array([200, 200, 200])
    mask  = cv2.inRange(hsv, lower, upper)
    res   = cv2.bitwise_and(image, image, mask=mask)
    image = cv2.medianBlur(res, 9)
    _save(image, path, '04_masked')

    # Get contour center from the _masked_ image.
    image, center = _get_contour(image, path)
    assert isinstance(center, tuple)
    if center[0] == -1 and center[1] == -1:
        return

    # Correct for cropping.
    cX = center[0] + X_MIN
    cY = center[1] + Y_MIN

    # Then we can overlay on the real image.
    image_bgr = img.copy()
    #cv2.drawContours(image_bgr, [approx], -1, GREEN, 2)
    cv2.circle(image_bgr, (cX,cY), 50, GREEN, thickness=2)
    cv2.circle(image_bgr, (cX,cY), 3, GREEN, thickness=-1)
    cv2.putText(img=image_bgr,
                text="{},{}".format(cX,cY),
                org=(cX+10, cY+10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=1,
                color=BLUE,
                thickness=2)
    _save(image_bgr, path, '07_original_contour')


if __name__ == "__main__":
    use_dvrk = False

    if use_dvrk:
        # Set up the dvrk, close gripper, dvrk should be gripping some tape or ball.
        p = dvrkClothSim()
        p.arm.set_jaw(jaw=0)
        print('current pose:  {}'.format(p.arm.get_current_pose(unit='deg')))
        cam = camera.RGBD()
        print('Initialized `dvrkClothSim()`.\n')

        # Get image as directly seen from the robot's camera.
        c_img_raw = None
        d_img_raw = None
        while c_img_raw is None:
            c_img_raw = cam.read_color_data()
        while d_img_raw is None:
            d_img_raw = cam.read_depth_data()

        num = len([x for x in os.listdir('.') if 'c_img_' in x])
        cv2.imwrite('c_img_{}.png'.format(str(num).zfill(2)), c_img_raw)
    else:
        # Get set of images, run some tests. Restrict to 12 char stuff: c_img_XX.png
        img_paths = sorted(
                [join('data',x) for x in os.listdir('data/')
                if 'c_img_' in x and '.png' in x and len(x) == 12]
        )
        for pth in img_paths:
            print('\nTesting: {}'.format(pth))
            img = cv2.imread(pth)
            _detect_region(img, pth)
