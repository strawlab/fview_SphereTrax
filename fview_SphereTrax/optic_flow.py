# optic_flow.py
#
# Optic flow velocity estimation for fview_SphereTrax
#
# Will Dickson 09/08/2006
# ------------------------------------------------------------------
import numpy
import numpy.numarray as nx
import scipy.linalg
import scipy.ndimage
from numpy.numarray.linear_algebra import solve_linear_equations as solve
from numpy.numarray.nd_image import gaussian_filter

# Least squares derivative kernels
X_KERNEL = nx.array([
    [ 0.0,  0.0, 0.0, 0.0, 0.0],
    [ 0.0,  0.0, 0.0, 0.0, 0.0],
    [-2.0, -1.0, 0.0, 1.0, 2.0],
    [ 0.0,  0.0, 0.0, 0.0, 0.0],
    [ 0.0,  0.0, 0.0, 0.0, 0.0]
    ])/(10.0*1.0)
X_KERNEL = X_KERNEL.astype( nx.Float32 )

Y_KERNEL = nx.array([
    [0.0, 0.0,  2.0, 0.0, 0.0],
    [0.0, 0.0,  1.0, 0.0, 0.0],
    [0.0, 0.0,  0.0, 0.0, 0.0],
    [0.0, 0.0, -1.0, 0.0, 0.0],
    [0.0, 0.0, -2.0, 0.0, 0.0]
    ])/(10.0*1.0)
Y_KERNEL = Y_KERNEL.astype( nx.Float32 )

def lsq_deriv(a):
    """
    Least squares partial derivative
    """

    ### Could this be sped up by doing multiplication in Fourier
    ### domain?

    # XXX Note: switching from numarray to scipy convolve
    # implementation probably switched the calling conventions. Need
    # to double check this. - ADS 20071102

    # array size error fixed by WBD 20071207

    a_x = -scipy.ndimage.convolve(a,X_KERNEL,mode='constant')
    a_y = scipy.ndimage.convolve(a,Y_KERNEL,mode='constant')
    return a_x, a_y

def get_optic_flow(im0,im1,pix,wnd,dt):
    """
    Compute optic flow at pixel location

    Inputs:
        im0 -- image 0
        im1 -- image 1
        pix -- pixel locations
        wnd -- computation window
        dt  -- time step between images
    """
    # Get index location of pixel
    i0,i1 = pix[1], pix[0]

    # Get regions for optic flow calculation
    E1_w_bndry = im1[i0-wnd-2:i0+wnd+3,i1-wnd-2:i1+wnd+3]
    E0 = im0[i0-wnd:i0+wnd+1,i1-wnd:i1+wnd+1]

    # Cast to Float for computation
    E1_w_bndry = nx.asarray(E1_w_bndry, nx.Float32)
    E0 = nx.asarray(E0, nx.Float32)
    E1 = E1_w_bndry[2:-2,2:-2]

    # Gaussian Filter
    #std = 1.5
    #E1_w_bndry = gaussian_filter(E1_w_bndry,std)
    #E1 = E1_w_bndry[2:-2,2:-2]
    #E0 = gaussian_filter(E0, std)

    # Get time derivatives
    dE_dt = (E1-E0)/dt

    # Get spacial derivatives
    Ex_w_bndry, Ey_w_bndry = lsq_deriv(E1_w_bndry)
    Ex = Ex_w_bndry[2:-2,2:-2]
    Ey = Ey_w_bndry[2:-2,2:-2]


    # Create Problem matrices
    n = 2*wnd+1
    A = nx.concatenate((nx.ravel(Ex)[:,None], nx.ravel(Ey)[:,None]), axis=1)
    b = -nx.ravel(dE_dt)

    ## Solution method 1 (fastest?) ------------------------
    #At = nx.transpose(A)
    #AtA = nx.matrixmultiply(At,A)
    #Atb = nx.matrixmultiply(At,b)
    #try:
    #    x = solve(AtA,Atb)
    #except:
    #    x = nx.array([0.0,0.0])

    # Solution method 2 ------------------------------------
    # New method
    x,resids,rank,s = scipy.linalg.lstsq(A,b)

    return x[0], x[1]


# ----------------------------------------------------------------------
def get_ang_vel(img_pts, img_dpix, cal, radius, sphere_pos):
    """
    Compute angular velocity vector of the sphere given a set of pixel
    displacement velocities and the corresponding image coordinates.

    """
    n = len(img_pts)
    f0,f1,c0,c1 = cal
    a,b,c = sphere_pos
    # Get points on sphere corresponding to image points
    sph_pts = []
    for p in img_pts:
        #print p
        q = img2sphere(p,cal,radius,sphere_pos)
        if q == None:
            # If image point is off sphere just return zero
            return 0.0, 0.0, 0.0

        sph_pts.append(q)

    # Create problem vector and matrix
    A = numpy.zeros((2*n,3), numpy.float)
    B = numpy.zeros((2*n,), numpy.float)
    for i in range(0,2*n):
        j = i/2

        # Image pts w/ camera calibration removed
        u = (img_pts[j][0] - c0)/f0
        v = (img_pts[j][1] - c1)/f1

        # Image velocities w/ camera calibration removed
        du = img_dpix[j][0]/f0
        dv = img_dpix[j][1]/f1

        # Point on sphere corresponding to image point
        x,y,z = sph_pts[j]

        if i%2 == 0:
            A[i,0] = -u*(y-b)
            A[i,1] = (z-c) + u*(x-a)
            A[i,2] = -(y-b)
            B[i] = z*du
        else:
            A[i,0] = -(z-c) - v*(y-b)
            A[i,1] = v*(x-a)
            A[i,2] = (x-a)
            B[i] = z*dv
    omega, resids, ranks, s = scipy.linalg.lstsq(A,B)
    return omega


def img2sphere(img_pt,cal,radius,sphere_pos):
    """
    Get points on near surface of sphere (w/smalles z value) corresponding
    to the given image point p.
    """
    # Convert points to world coords
    f0,f1,c0,c1 = cal
    p = (img_pt[0] - c0)/f0, (img_pt[1] - c1)/f1
    # Get coefficients in quadratic equation
    a,b,c = tuple(sphere_pos)
    A = p[0]**2 + p[1]**2 + 1
    B = -2.0*(p[0]*a + p[1]*b + c)
    C = a**2 + b**2 + c**2 - radius**2
    # Check if there is a solution
    root_term = B**2 - 4.0*A*C
    if root_term < 0.0:
        return None
    else:
        # Get near solution (smallest z value)
        z = (-B - numpy.sqrt(root_term))/(2.0*A)
        x = p[0]*z
        y = p[1]*z
        return x,y,z


def get_hfs_rates(w,orient_vecs,sphere_raduis):
    """
    Calculate the heading, forward and side rates/velocities
    given the angular velocity, the orientation basis vectors,
    and the raduis of the sphere.
    """
    u_vec, f_vec, s_vec = orient_vecs
    fly_pos = sphere_raduis*u_vec
    vel_vec = numpy.cross(w,fly_pos)
    head_rate = numpy.dot(u_vec,w)
    forw_rate = numpy.dot(vel_vec, f_vec)
    side_rate = numpy.dot(vel_vec, s_vec)
    return head_rate, forw_rate, side_rate





