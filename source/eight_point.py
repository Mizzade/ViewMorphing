'''
Implementation of the Paper 'In Defense Of The 8-Point Algoithm.
'''
import cv2
import numpy as np
from helpers import (get_good_flann_matches, normalize_points,
                     find_fundamental_matrix, get_rotation_matrix,
                     find_projective_transformations)

'''
First off, get matching points u_i and u_i^prime
'''
if __name__ == '__main__':
    im_src = cv2.imread('./datasets/poptarts/p1.jpg')
    im_dst = cv2.imread('./datasets/poptarts/p2.jpg')
    sift = cv2.xfeatures2d.SIFT_create()

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(im_src, None)
    kp2, des2 = sift.detectAndCompute(im_dst, None)

    good_matches = get_good_flann_matches(des1, des2)

    # min number of matches need to construct fundamental matrix is 8
    MIN_MATCH_COUNT = 8

    if len(good_matches) < MIN_MATCH_COUNT:
        print('Not enough matches found: ', len(good_matches), '/',
              MIN_MATCH_COUNT)
    else:
        # ndArrays
        PTS_SRC = np.float32([kp1[m.queryIdx].pt for m in good_matches][:9])
        PTS_DST = np.float32([kp2[m.trainIdx].pt for m in good_matches][:9])

        # Normalize points and get the transformation matrix
        T_SRC, PTS_SRC_NORMALIZED = normalize_points(PTS_SRC)
        T_DST, PTS_DST_NORMALIZED = normalize_points(PTS_DST)

        # Get F_PRIME, a special form of the fundamental matrix that is
        # singular and of rank 2.
        F_PRIME = find_fundamental_matrix(PTS_SRC_NORMALIZED,
                                          PTS_DST_NORMALIZED)

        F = T_DST * F_PRIME * T_SRC.T

        # Get image prewarping transformations like descriped in appendix
        # of `View Morphing`
        H0, H1 = find_projective_transformations(F)

        # TODO Does H0 or H1 have to be transposed?
        # Because: (H1^-1)^T F H0^-1 = F_hat
        H1_inverse = np.linalg.inv(H1)
        H0_inverse = np.linalg.inv(H0)

        F_hat = H1_inverse * F * H0_inverse
        F_hat2 = H1_inverse.T * F * H0_inverse

        shape_src = im_src.shape

        warped1 = cv2.warpPerspective(im_src, F_hat, shape_src[::-1][1:])
        warped2 = cv2.warpPerspective(im_src, F_hat2, shape_src[::-1][1:])

        out = np.concatenate((im_src, warped1, warped2), axis=1)
        cv2.imshow('OUT', out)

    # cv2.imshow('Src', im_src)
    # cv2.imshow('Dst', im_dst)
    cv2.waitKey(0)
