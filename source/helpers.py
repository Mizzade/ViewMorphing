import cv2
import numpy as np

INDEX_PARAMS = dict(
    algorithm=0,  # FLANN_INDEX_KDTREE
    trees=5
)

SEARCH_PARAMS = dict(check=50)

FLANN = cv2.FlannBasedMatcher(INDEX_PARAMS, SEARCH_PARAMS)


def get_good_flann_matches(des1, des2):
    MATCHES = FLANN.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Loew's ratio test.
    GOOD_MATCHES = []
    for m, n in MATCHES:
        if m.distance < 0.7 * n.distance:
            GOOD_MATCHES.append(m)

    return GOOD_MATCHES


def normalize_points(points):
    '''
    Return transformation matrix and transformed points in homogenous form.
    Computing normalization (translation and scaling) of the coordinates of
    matched points.

    arguments
    -------------
    points -- np.array of 2D coordinates of form [n x 2] where n is the number
              of points.

    returns
    -------------
    transformation_matrix -- np.matrix of form [3 x 3]
    transformed_homogenous_points -- np.matrix of form [n x 3]
    '''

    ''' Transform input points by translating and scaling them.
    Return transformed points in homogenous form as well as Transformation
    matrix T.
    '''
    # mean values of x and y axis seperately: [mean_x, mean_y]
    MEAN = np.mean(points, axis=0)

    # [x, y] -> [x - mean_x, y - mean_y]
    mean_shifted_points = points - MEAN

    '''
    As a first step, the coordinates in each image are translated (by a
    different translation for each image) so as to bring the centroid of the
    set of all points to the origin. The coordinates are also scaled. The
    previous sections suggested that the best results will be obtained if the
    coordinates are scaled, so that on the average a point u is of the form
    u = (1, 1, 1)^T.
    Such a point will lie a distance √2 from the origin. Rather than choose
    different scale factors for each direction, an isotropic scaling factor is
    chosen so that the u and v coordinates of a point are scaled equally. The
    transformation is as follows:
    1. The points are translated so that their centroid is at the origin.
    2. The points are then scaled isotropically so that the average distance
        from the origin is equal to √2.
    '''
    # sqrt(sum(x^2 + y^2)) for all points
    # Norm of all vectors OR of each vector?
    xs = mean_shifted_points[:, 0]
    ys = mean_shifted_points[:, 1]

    FROBENIUS_NORM = np.mean(np.sqrt(xs * xs + ys * ys))
    # FROBENIUS_NORM = np.linalg.norm(mean_shifted_points, ord='fro')
    SCALING_FACTOR = np.sqrt(2) / FROBENIUS_NORM

    '''
    Transformation matrix in normalized form. This matrix will scale all
    points isotropically and translate them so that their centroid is at the
    origin.
    '''
    t_normalized = np.matrix([
        [SCALING_FACTOR, 0, -SCALING_FACTOR * MEAN.item(0)],
        [0, SCALING_FACTOR, -SCALING_FACTOR * MEAN.item(1)],
        [0, 0, 1]
    ], np.float32)

    # Convert points from euklidian to homogenous form: [x, y] -> [x, y, 1]
    homogenous_points = cv2.convertPointsToHomogeneous(points)[:, 0]
    normalized_points = (t_normalized * homogenous_points.T).T

    # TODO average distance to origin seems to be 1/3

    return t_normalized, normalized_points


def get_equation_matrix(pts_src, pts_dst):
    '''
    Calculate a equation matrix to find the fundamental matrix given two sets
    of points, u' and u.

    arguments:
    ----
    pts_src -- [m x 3] np.matrix. m is number of points in array
    pts_dst -- [m x 3] np.matrix. m is number of points in array

    returns:
    --------
    A --  [m x 9] np.matrix. Equation matrix.
    '''
    '''
    The fundamental matrix is defined by the equationu Fu = 0 (1) for any pair
    of matching points u' ↔ u in two images. Given sufficiently many point
    matches ui' ↔ ui, (at  least  8) this equation (1) can be used to compute
    the unknown matrix F.
    In particular, writing u=(u, v, 1) and u'=(u', v', 1) each point
    match gives rise to one linear equation in the unknown entries of F. The
    coefficients of this equation are easily written in terms of the known
    coordinates u and u'. Specifically, the equation corresponding to a pair
    of points (u, v, 1) and (u', v', 1) will be
    uu'f11 + uv'f21 + uf31 + vu'f12 + vv'f22 + vf32 + u'f13 + v'f23 + f33 = 0.
    The row of the equation matrix may be represented asa vector
    (uu', uv', u, vu', vv', v, u', v', 1).
    From all the point matches, we obtain a set of linear equations of the form
    Af = 0 (2), where f is a 9-vector containing the entries of the matrix F,
    and A is the equation matrix. The fundamental matrix F, and hence the
    solution vector f is defined only up to an unknown scale. For this reason,
    and to avoid the trivial solution f, we make the additional constraint
    ||f|| = 1 where ||f||, is the norm of f. Under these conditions, it is
    possible to find a solution to the system (2) with as few as 8 point
    matches.
    '''
    number_of_matches, _ = pts_src.shape

    u = pts_src[:, 0].getA1()
    v = pts_src[:, 1].getA1()
    up = pts_dst[:, 0].getA1()
    vp = pts_dst[:, 1].getA1()

    ONES = np.matrix(np.ones((number_of_matches))).getA1()

    # A = [m x 9], where as m = number of matching points
    A = np.matrix([u * up, u * vp, u, v * up, v * vp, v, up, vp, ONES]).T

    return A


def get_linear_solution_of_F(pts_src, pts_dst):
    '''
    Calculate a fundamental matrix for two specific images

    arguments:
    ----
    pts_src -- [m x 3] ndarray. m is number of points in array
    pts_dst -- [m x 3] ndarray. m is number of points in array

    returns:
    --------
    F -- fundamental matrix transforms u_i of pts_src to the corresponding
          points u`_i of pts_dst.
          [3 x 3], np.matrix
    '''
    A = get_equation_matrix(pts_src, pts_dst)

    '''
    In  fact, because of inaccuracies in the measurement or specification of
    the matched points, the  matrix A will not be rank-deficient – it will have
    rank 9. In this case, we will not be able to find a non-zero solution to
    the equations Af = 0. Instead, we seek a least-squares solution to this
    equation set. In particular, we seek the vector f that minimizes ||Af||
    subject to the constraint ||f|| = f^T f = 1.
    It is well known (and easily derived using Lagrange multipliers) that the
    solution to this problem is the unit eigenvector of A^T A corresponding to
    the smallest eigenvalue of A.
    Note that since A^T A is positive semi-definite and symmetric, all its
    eigenvectors are real and positive, or zero. For convenience, (though
    somewhat inexactly), we will call this eigenvector the least eigenvector of
    A^T A. An appropriate algorithm for finding this eigenvector is the
    algorithm of Jacobi or the Singular Value Decomposition.
    '''

    ''' Excerpt from help(np.linalg.svd):
    The SVD is commonly written as ``a = U S V.H``.  The `v` returned
    by this function is ``V.H`` and ``u = U``.
    The rows of `v` are the eigenvectors of ``a.H a``. The columns
    of `u` are the eigenvectors of ``a a.H``.  For row ``i`` in
    `v` and column ``i`` in `u`, the corresponding eigenvalue is
    ``s[i]**2``.
    '''

    # Singular value decomposition of A
    _, _, V = np.linalg.svd(A, full_matrices=0)
    # Since np.linalg.svd returns V.T, we have to transpose V again.
    # VT = V.T

    # Extract the smalles singular value, that is the last row of V since, V is
    # acutally v^T.
    f = V.T[:, -1]

    # Reshape f to become F, a 3x3 matrix
    F = f.reshape((3, 3))

    return F


def constrain_fundamental_matrix(F):
    '''
    Constrain a fundamental matrix F to be singular and of rank 2.

    arugments:
    -----------
    F -- fundamental matrix, [3 x 3], np.matrix

    returns:
    -----------
    FP -- F prime. [3 x 3], np.matrix. F prime is of rank 2 and minimizes the
          Frobenius norm || F - FP ||, subject to the condition det(FP) = 0.
    '''
    '''
    Let F = UDV^T be the Singular Value Decomposition of F, where D is a
    diagonal matrix D = diag(r, s, t) satisfying r ≥ s ≥ t. We let
    F` = Udiag(r, s, 0)V^T. This method was suggested by Tsai and Huang and has
    been proven to minimize the Frobenius norm of F − F`, as required.
    '''

    # Enforce singularity and rank contraint.
    U, S, VT = np.linalg.svd(F, full_matrices=0)
    S[2] = 0
    D = np.diag(S)

    # F prime
    FP = U * D * VT

    return FP


def find_fundamental_matrix(pts_src, pts_dst):
    '''
    8-point algorithm for computation of the fundamental matrix consisting of
    two steps:
    1. Linear solution
    2. Constraint enforcement.

    arguments:
    -----------
    pts_src -- [m x 3] ndarray. m is number of points in array
    pts_dst -- [m x 3] ndarray. m is number of points in array

    returns:
    --------
    FP -- fundamental matrix transforms u_i of pts_src to the corresponding
          points u`_i of pts_dst.
          [3 x 3], np.matrix of rank 2, and det(FP) = 0
    '''
    F = get_linear_solution_of_F(pts_src, pts_dst)
    FP = constrain_fundamental_matrix(F)

    return FP


def get_rotation_matrix(rotation_vector, theta):
    '''
    Return rotation matrix from rotation vector and rotation angle theta.

    arugments:
    ----------
    rotation_vector: [1x3] or [3x1] vector
    theta: angle in radians

    returns:
    ----------
    R: a 3x3 rotation matrix
    '''

    '''
    Rodiguez Rotation Formula
    I = Identitymatrix
    k = axis of rotation as normalized vector
    K = [[ 0,  -k_z,  k_y],
            [ k_z,  0,  -k_x],
            [-k_y, k_x,   0 ]]
    R = I + (sin(Theta) * K) + (1 - cos(Theta) * K^2)
    '''
    norm = np.linalg.norm(rotation_vector, ord='fro')
    if (norm == 0):
        return None

    k = rotation_vector / norm
    I = np.matrix(np.eye(3))
    K = np.matrix([[0, -k.item(2), k.item(1)],
                  [k.item(2), 0, -k.item(0)],
                  [-k.item(1), k.item(0), 0]])

    R = I + (np.sin(theta) * K) + (1 - np.cos(theta)) * K * K

    return R


def get_theta(r, e):
    '''
    Given an epipole and a rotation axis, compute the rotation angle theta in
    radians.

    arguments:
    ----------
    r -- rotationsAxis, [1 x 3] or [3 x 1], np.matrix
    e -- epipole coordinates, [1 x 3] or [3 x 1], np.matrix

    returns:
    theta -- float, angle in radians
    '''
    theta = -np.pi / 2 - np.arctan((r.item(1) * e.item(0) - r.item(0) *
                                    e.item(1)) / e.item(2))

    return theta


def find_projective_transformations(F):
    '''
    Given fundamental matrix F, find the projective transformations H0, H1,
    such that (H1^-1)^T F H0^-1 = F_hat and F_hat is of the form
    F_hat = [[0,  0,  0],
             [0,  0, -1],           (8)
             [0,  1,  0]]

    arguments:
    ----------
    F -- fundamental matrix, np.matrix of form [3 x 3]

    returns:
    --------
    [H0, H1] -- H_i is a projective transformation matrix.
                np.matrix of form [3 x 3]
    '''
    '''
    A sufficient condition for two views to be parallel is that their
    fundamental matrix have the form:
    F_hat = [[0,  0,  0],
            [0,  0, -1],           (8)
            [0,  1,  0]]
    Consequently, any two images with fundamental matrix F may be prewarped
    (i.e., made parallel) by choosing any two projective transforms H0 and
    H1 such that (H1^-1)^T F H0^-1 = F_hat
    '''

    '''
    Here we describe one method that applies a rotation in depth to make
    the images planes parallel, followed by an affine transformation to
    align corresponding scanlines. The procedure is determined by choosing
    an (arbitrary) axis of rotation d0 = [d0^x, d0^y, 0]^T element of I0.
    Given [x y z]^T = Fd0, the corresponding axis in I1 is determined
    according to d1 = [-y x 0]^T. To compute the angles of depth rotation
    we need the epipoles, also known as vanishing points, e0 element of I0
    and e1 element of I1.

    e0 = [e0^x e0^y e0^z]^T and
    e1 = [e1^x e1^y e1^z]^T are the unit eigenvectors of F and F^T
    respectively, coresponding to eigenvalues of 0.

    The entire procedure is determined by selecting d0. A suitable choice
    is to select d0 orthogonal to e0, i.e., d0 = [-e_0^y e_0^x 0]^T.
    '''

    '''
    np.linalg.eig:
    Returns:
    w : (..., M) array
    The eigenvalues, each repeated according to its multiplicity. The
    eigenvalues are not necessarily ordered. The resulting array will be of
    complex type, unless the imaginary part is zero in which case it will be
    cast to a real type. When a is real the resulting eigenvalues will be real
    (0 imaginary part) or occur in conjugate pairs
    v : (..., M, M) array
    The normalized (unit “length”) eigenvectors, such that the column v[:,i] is
    the eigenvector corresponding to the eigenvalue w[i].
    '''
    _, E0 = np.linalg.eig(F)
    _, E1 = np.linalg.eig(F.T)

    # Last column should be the Eigenvector corresponding to Eigenvalue of 0.
    e0 = E0[:, -1]
    e1 = E1[:, -1]

    '''
    The procedure is determined by choosing
    an (arbitrary) axis of rotation d0 = [d0^x, d0^y, 0]^T element of I0.
    Given [x y z]^T = Fd0, the corresponding axis in I1 is determined
    according to d1 = [-y x 0]^T.
    '''
    d0 = np.matrix([-e0.item(1), e0.item(0), 0]).T
    Fd0 = F * d0
    d1 = np.matrix([-Fd0.item(1), Fd0.item(0), 0]).T

    '''
    A view's epipole represents the projection of the optical center of
    the other view. The following procedure will work provided the views
    are not singular, i.e., the epipoles are outside the image borders and
    therefore not withing the field of view.
    The angles of rotation in depth about d_i are given by
    Theta_i = -PI/2 - tan^-1 (( d_i^y e_i^x - d_i^x e_i^y) / (e_i^z)).
    '''
    theta0 = get_theta(d0, e0)
    theta1 = get_theta(d1, e1)

    '''
    We denote as R_{Theta_i}^{d_i} the 3x3 matrix corresponding to a
    rotation of angle Theta_i about axis d_i. Applying R_{Theta_0}^{d_0} to
    I0 and R_{Theta_1}^{d_1} to I1 makes the two images planes parallel.
    Although this is technically sufficent for prewarping it is useful to
    add an additional affine warp to align the scanlines. This simplifies
    the morph step to a scanline interpolation and also avoids bottleneck
    problems that arise as a result of image plane rotations.
    '''

    R_theta_0 = get_rotation_matrix(d0, theta0)
    R_theta_1 = get_rotation_matrix(d1, theta1)

    '''
    The next step is to rotate the images so that epipolar lines are
    horinzontal. The new epipoles are
    [e_tilde_{i}^{x} e_tilde_{i}^y} 0] ^T = R_{Theta_i}^{d_i}e_i.
    '''

    e_tilde_0 = R_theta_0 * e0  # epipole of Image0
    e_tilde_1 = R_theta_1 * e1  # epipole of Image1

    '''
    The angles of rotation phi_0 and phi_1 are given by
    phi_i = -tan^-1 (e_tilde_{i}^{y}/e_tilde_{i}^{x}).
    '''
    phi0 = - np.arctan(e_tilde_0.item(1) / e_tilde_0.item(0))
    phi1 = - np.arctan(e_tilde_1.item(1) / e_tilde_1.item(0))

    '''
    After applying thes image plane rotations, the fundamental matrix has
    the form
    F_tilde =
    R_{phi_1} R_{Theta_1}^{d_1} F R_{-Theta_0}^{d_0} R_{-phi_0} =
        [[0 0 0],
         [0 0 a],
         [0 b c]]
    The 3x3 matrix R_Theta denots the image plane (z axis) rotation of
    angle Theta.
    '''

    R_phi_0 = get_rotation_matrix(np.matrix([0, 0, 1]), phi0)
    R_phi_1 = get_rotation_matrix(np.matrix([0, 0, 1]), phi1)

    R_minus_theta_0 = get_rotation_matrix(d0, -theta0)
    R_minus_phi_0 = get_rotation_matrix(np.matrix([0, 0, 1]), -phi0)

    F_tilde = R_phi_1 * R_theta_1 * F * R_minus_theta_0 * R_minus_phi_0

    '''
    Finally, to get F into the for of Equation (8), the
    second image is translated and vertically scaled by matrix
    T = [[1  0  0 ],
         [0 -a  -c],
         [0  0  b ]]
    '''

    T = np.matrix([[1, 0, 0],
                  [0, -F_tilde.item(5), -F_tilde.item(8)],
                  [0, 0, F_tilde.item(8)]])

    '''
    In summary, the prewarping transforms H0 and H1 are
    H0 = R_{theta_0} R_{Theta_0}^{d_0}
    H1 = T R_{theta_1} R_{Theta_1}^{d_1}
    '''
    H0 = R_phi_0 * R_theta_0
    H1 = T * R_phi_1 * R_theta_1

    return H0, H1




