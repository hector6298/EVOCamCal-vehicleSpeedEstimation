from skimage import feature, color, transform, io
import numpy as np
import logging

def parseParams(param_file):
    file = open(param_file)
    params = file.readlines()[3:]
    params = params[0].rstrip()
    params = params.split(' ')
    params = np.array(params).astype(np.float64)
    return params

def parseHomography(param_file):
    homographyMat = np.zeros((3,3), 'float32')
    file = open(param_file)
    params = file.readlines()[0]
    matRows = params.rstrip().split(';')
    tof = lambda i : np.float32(i)
    homographyMat[0] = list(map(tof, matRows[0].split(' ')))
    homographyMat[1] = list(map(tof, matRows[1].split(' ')))
    homographyMat[2] = list(map(tof, matRows[2].split(' ')))
    return homographyMat

def homography2Dto3D(o2dpt, homoMat):
    homogen2dPt = [*o2dpt, 1]
    matInv = np.linalg.inv(homoMat)
    hh = np.dot(matInv, homogen2dPt)
    scalar = hh[2]
    return [hh[0]/scalar, hh[1]/scalar]


def reproject2Dto3D(o2dPt,  afProjMat, nLenUnit:int=1000, nCoordSysTyp:int=1):
    oMatA = np.empty((3,3), dtype=np.float64)

    if nCoordSysTyp == 0:
        oMatA[0,0] = afProjMat[0]
        oMatA[0,1] = -o2dPt[1]
        oMatA[0,2] = afProjMat[2]
        oMatA[1,0] = afProjMat[4]
        oMatA[1,1] = -o2dPt[0]
        oMatA[1,2] = afProjMat[6]
        oMatA[2,0] = afProjMat[8]
        oMatA[2,1] = -1.0
        oMatA[2,2] = afProjMat[10]
    
    if nCoordSysTyp == 1:
        oMatA[0,0] = afProjMat[0]
        oMatA[0,1] = afProjMat[1]
        oMatA[0,2] = -o2dPt[1]
        oMatA[1,0] = afProjMat[4]
        oMatA[1,1] = afProjMat[5]
        oMatA[1,2] = -o2dPt[0]
        oMatA[2,0] = afProjMat[8]
        oMatA[2,1] = afProjMat[9]
        oMatA[2,2] = -1.0

    oMatAInv = np.linalg.inv(oMatA)
    oMatB = np.empty((3,1), dtype=np.float64)
    oMatB[0,0] = -afProjMat[3]
    oMatB[1,0] = -afProjMat[7]
    oMatB[2,0] = -afProjMat[11]

    oMatM = np.matmul(oMatAInv, oMatB)

    o3dPt = None

    if nCoordSysTyp == 0:
        o3dPt = (oMatM[0,0]/nLenUnit, 0.0, oMatM[2,0]/nLenUnit)
    elif nCoordSysTyp == 1:
        o3dPt = (oMatM[0,0]/nLenUnit, oMatM[1,0]/nLenUnit, 0.0)

    return o3dPt

def getFocal(vp1, vp2, pp):
    return np.sqrt(- np.dot(vp1[0:2]-pp[0:2], vp2[0:2]-pp[0:2]))

def homogenizeAndNormalize(p):
    p = np.array(p)
    p = p.flatten()
    assert len(p) == 2 or len(p) == 3
    if len(p) == 2:
        return np.concatenate((p, [1]))
    else:
        return p/p[2]

def computeCameraCalibration(_vp1, _vp2, _pp):
    vp1 = homogenizeAndNormalize(_vp1)
    vp2 = homogenizeAndNormalize(_vp2)
    pp = homogenizeAndNormalize(_pp)
    focal = getFocal(vp1, vp2, pp)
    vp1W = np.concatenate((vp1[0:2], [focal]))    
    vp2W = np.concatenate((vp2[0:2], [focal]))    
    ppW = np.concatenate((pp[0:2], [0])) 
    vp3W = np.cross(vp1W-ppW, vp2W-ppW)
    vp3 = np.concatenate((vp3W[0:2]/vp3W[2]*focal + ppW[0:2], [1]))
    vp3Direction = np.concatenate((vp3[0:2], [focal]))-ppW
    roadPlane = np.concatenate((vp3Direction/np.linalg.norm(vp3Direction), [10]))
    return vp1, vp2, vp3, pp, roadPlane, focal

def img2wrldPoint(pt2d, poseMat, focal, pp):
    Projection_mtx=mtx.dot(poseMat)
    Projection_mtx = np.delete(Projection_mtx, 2, 1)
    Inv_Projection = np.linalg.inv(Projection_mtx)
    img_point=np.array([pts1_blue[0]])
    img_point=np.vstack((img_point,np.array(1)))
    pt3d=Inv_Projection.dot(img_point)
    return pt3d

def getPrincipalPoint(imgSize):
    return (imgSize[1]//2, imgSize[0]//2)

def compute_vp(lines, threshold_inlier=5, num_ransac_iter=5000, reestimate_model=False):
    edgelets = compute_edgelets(lines)
    vp1 = ransac_vanishing_point(edgelets, num_ransac_iter=num_ransac_iter, 
                             threshold_inlier=threshold_inlier)
    if reestimate_model:
        vp1 = reestimate_model(vp1, edgelets, threshold_reestimate=threshold_inlier)
    vp1 = vp1 / vp1[2]
    return vp1

def compute2vps(linesH, linesV, threshold_inlier=5, num_ransac_iter=5000, reestimate_model=False):
    vpH = compute_vp(linesH, threshold_inlier, num_ransac_iter, reestimate_model)
    vpV = compute_vp(linesV, threshold_inlier, num_ransac_iter, reestimate_model)
    return vpH, vpV
    
def compute_edgelets(lines, sigma=3):
    """Create edgelets as in the paper.
    Uses canny edge detection and then finds (small) lines using probabilstic
    hough transform as edgelets.
    Parameters
    ----------
    image: ndarray
        Image for which edgelets are to be computed.
    sigma: float
        Smoothing to be used for canny edge detection.
    Returns
    -------
    locations: ndarray of shape (n_edgelets, 2)
        Locations of each of the edgelets.
    directions: ndarray of shape (n_edgelets, 2)
        Direction of the edge (tangent) at each of the edgelet.
    strengths: ndarray of shape (n_edgelets,)
        Length of the line segments detected for the edgelet.
    """

    locations = []
    directions = []
    strengths = []

    for p0, p1 in lines:
        p0, p1 = np.array(p0), np.array(p1)
        locations.append((p0 + p1) / 2)
        directions.append(p1 - p0)
        strengths.append(np.linalg.norm(p1 - p0))

    # convert to numpy arrays and normalize
    locations = np.array(locations)
    directions = np.array(directions)
    strengths = np.array(strengths)

    directions = np.array(directions) / \
        np.linalg.norm(directions, axis=1)[:, np.newaxis]

    return (locations, directions, strengths)


def edgelet_lines(edgelets):
    """Compute lines in homogenous system for edglets.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    Returns
    -------
    lines: ndarray of shape (n_edgelets, 3)
        Lines at each of edgelet locations in homogenous system.
    """
    locations, directions, _ = edgelets
    normals = np.zeros_like(directions)
    normals[:, 0] = directions[:, 1]
    normals[:, 1] = -directions[:, 0]
    p = -np.sum(locations * normals, axis=1)
    lines = np.concatenate((normals, p[:, np.newaxis]), axis=1)
    return lines


def compute_votes(edgelets, model, threshold_inlier=5):
    """Compute votes for each of the edgelet against a given vanishing point.
    Votes for edgelets which lie inside threshold are same as their strengths,
    otherwise zero.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    model: ndarray of shape (3,)
        Vanishing point model in homogenous cordinate system.
    threshold_inlier: float
        Threshold to be used for computing inliers in degrees. Angle between
        edgelet direction and line connecting the  Vanishing point model and
        edgelet location is used to threshold.
    Returns
    -------
    votes: ndarry of shape (n_edgelets,)
        Votes towards vanishing point model for each of the edgelet.
    """
    vp = model[:2] / model[2]

    locations, directions, strengths = edgelets

    est_directions = locations - vp
    dot_prod = np.sum(est_directions * directions, axis=1)
    abs_prod = np.linalg.norm(directions, axis=1) * \
        np.linalg.norm(est_directions, axis=1)
    abs_prod[abs_prod == 0] = 1e-5

    cosine_theta = dot_prod / abs_prod
    theta = np.arccos(np.abs(cosine_theta))

    theta_thresh = threshold_inlier * np.pi / 180
    return (theta < theta_thresh) * strengths


def ransac_vanishing_point(edgelets, num_ransac_iter=2000, threshold_inlier=5):
    """Estimate vanishing point using Ransac.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    best_model: ndarry of shape (3,)
        Best model for vanishing point estimated.
    Reference
    ---------
    Chaudhury, Krishnendu, Stephen DiVerdi, and Sergey Ioffe.
    "Auto-rectification of user photos." 2014 IEEE International Conference on
    Image Processing (ICIP). IEEE, 2014.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 2]

    best_model = None
    best_votes = np.zeros(num_pts)

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]

        current_model = np.cross(l1, l2)

        if np.sum(current_model**2) < 1 or current_model[2] == 0:
            # reject degenerate candidates
            continue

        current_votes = compute_votes(
            edgelets, current_model, threshold_inlier)

        if current_votes.sum() > best_votes.sum():
            best_model = current_model
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes.sum(), ransac_iter))

    return best_model


def ransac_3_line(edgelets, focal_length, num_ransac_iter=2000,
                  threshold_inlier=5):
    """Estimate orthogonal vanishing points using 3 line Ransac algorithm.
    Assumes camera has been calibrated and its focal length is known.
    Parameters
    ----------
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    focal_length: float
        Focal length of the camera used.
    num_ransac_iter: int
        Number of iterations to run ransac.
    threshold_inlier: float
        threshold to be used for computing inliers in degrees.
    Returns
    -------
    vp1: ndarry of shape (3,)
        Estimated model for first vanishing point.
    vp2: ndarry of shape (3,)
        Estimated model for second vanishing point, which is orthogonal to
        first vanishing point.
    Reference
    ---------
    Bazin, Jean-Charles, and Marc Pollefeys. "3-line RANSAC for orthogonal
    vanishing point detection." 2012 IEEE/RSJ International Conference on
    Intelligent Robots and Systems. IEEE, 2012.
    """
    locations, directions, strengths = edgelets
    lines = edgelet_lines(edgelets)

    num_pts = strengths.size

    arg_sort = np.argsort(-strengths)
    first_index_space = arg_sort[:num_pts // 5]
    second_index_space = arg_sort[:num_pts // 5]
    third_index_space = arg_sort[:num_pts // 2]

    best_model = (None, None)
    best_votes = 0

    for ransac_iter in range(num_ransac_iter):
        ind1 = np.random.choice(first_index_space)
        ind2 = np.random.choice(second_index_space)
        ind3 = np.random.choice(third_index_space)

        l1 = lines[ind1]
        l2 = lines[ind2]
        l3 = lines[ind3]

        vp1 = np.cross(l1, l2)
        # The vanishing line polar to v1
        h = np.dot(vp1, [1 / focal_length**2, 1 / focal_length**2, 1])
        vp2 = np.cross(h, l3)

        if np.sum(vp1**2) < 1 or vp1[2] == 0:
            # reject degenerate candidates
            continue

        if np.sum(vp2**2) < 1 or vp2[2] == 0:
            # reject degenerate candidates
            continue

        vp1_votes = compute_votes(edgelets, vp1, threshold_inlier)
        vp2_votes = compute_votes(edgelets, vp2, threshold_inlier)
        current_votes = (vp1_votes > 0).sum() + (vp2_votes > 0).sum()

        if current_votes > best_votes:
            best_model = (vp1, vp2)
            best_votes = current_votes
            logging.info("Current best model has {} votes at iteration {}".format(
                current_votes, ransac_iter))

    return best_model


def reestimate_model(model, edgelets, threshold_reestimate=5):
    """Reestimate vanishing point using inliers and least squares.
    All the edgelets which are within a threshold are used to reestimate model
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
        All edgelets from which inliers will be computed.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    restimated_model: ndarry of shape (3,)
        Reestimated model for vanishing point in homogenous coordinates.
    """
    locations, directions, strengths = edgelets

    inliers = compute_votes(edgelets, model, threshold_reestimate) > 0
    locations = locations[inliers]
    directions = directions[inliers]
    strengths = strengths[inliers]

    lines = edgelet_lines((locations, directions, strengths))

    a = lines[:, :2]
    b = -lines[:, 2]
    est_model = np.linalg.lstsq(a, b)[0]
    return np.concatenate((est_model, [1.]))


def remove_inliers(model, edgelets, threshold_inlier=95):
    """Remove all inlier edglets of a given model.
    Parameters
    ----------
    model: ndarry of shape (3,)
        Vanishing point model in homogenous coordinates which is to be
        reestimated.
    edgelets: tuple of ndarrays
        (locations, directions, strengths) as computed by `compute_edgelets`.
    threshold_inlier: float
        threshold to be used for finding inlier edgelets.
    Returns
    -------
    edgelets_new: tuple of ndarrays
        All Edgelets except those which are inliers to model.
    """
    inliers = compute_votes(edgelets, model, threshold_inlier) > 0
    locations, directions, strengths = edgelets
    locations = locations[~inliers]
    directions = directions[~inliers]
    strengths = strengths[~inliers]
    edgelets = (locations, directions, strengths)
    return edgelets


def vis_edgelets(image, edgelets, show=True, ax=None):
    """Helper function to visualize edgelets."""
    import matplotlib.pyplot as plt
    if ax is not None:
        ax.imshow(image)
    else:
        plt.figure(figsize=(10, 10))
        plt.imshow(image)
    locations, directions, strengths = edgelets
    for i in range(locations.shape[0]):
        xax = [locations[i, 0] - directions[i, 0] * strengths[i] / 2,
               locations[i, 0] + directions[i, 0] * strengths[i] / 2]
        yax = [locations[i, 1] - directions[i, 1] * strengths[i] / 2,
               locations[i, 1] + directions[i, 1] * strengths[i] / 2]

        if ax is not None:
            ax.plot(xax,yax, 'r-')
        else:
            plt.plot(xax, yax, 'r-')

    if show:
        plt.show()


def vis_model(image, lines, model, ax=None, show=True):
    """Helper function to visualize computed model."""
    import matplotlib.pyplot as plt
    edgelets = compute_edgelets(lines)
    locations, directions, strengths = edgelets
    inliers = compute_votes(edgelets, model, 10) > 0

    edgelets = (locations[inliers], directions[inliers], strengths[inliers])
    locations, directions, strengths = edgelets
    vis_edgelets(image, edgelets, False, ax=ax)
    vp = model / model[2]
    
    if ax is not None:
        ax.plot(vp[0], vp[1], 'bo')
    else:
        plt.plot(vp[0], vp[1], 'bo')
    for i in range(locations.shape[0]):
        xax = [locations[i, 0], vp[0]]
        yax = [locations[i, 1], vp[1]]
        if ax is not None:
            ax.plot(xax, yax, 'b-.')
        else:
            plt.plot(xax, yax, 'b-.')

    if show:
        plt.show()
