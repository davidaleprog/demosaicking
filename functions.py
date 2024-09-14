
import numpy as np


# interpolation kernels
w1 = np.array([[0,0,-1,0,0], [0,0,2,0,0], [-1,2,4,2,-1], [0,0,2,0,0], [0,0,-1,0,0]])/8 # G at R location
w2 = w1 #G at B location
w3 = np.array([[0,0,0.5,0,0], [0,-1,0,-1,0], [-1,4,5,4,-1], [0,-1,0,-1,0], [0,0,0.5,0,0]])/8 # R at G location (Brow, Rcol)
w4 = np.array([[0,0,-1,0,0], [0,-1,4,-1,0], [0.5,0,5,0,0.5], [0,-1,4,-1,0], [0,0,-1,0,0]])/8 # R at G location (Rrow, Bcol)
w5 = np.array([[0,0,-3/2,0,0], [0,2,0,2,0], [-3/2,0,6,0,-1], [0,2,0,2,0], [0,0,-3/2,0,0]])/8 # R at B location (Brow, Bcol)
w6 = w3 #R at G location (Brow, Rcol)
w7 = w4 #B at G location (Rrow, Bcol)
w8 = w5 #B at R location (Rrow, Rcol)
w_s = [w1, w2, w3, w4, w5, w6, w7, w8]

def conv(A,B):
    return np.sum(A*B)

def bayer_gradient_interpolation(y, op, w=w_s):
    """
    Second interpolation method: convolution with a 2D kernels (5x5) using multi-channel interpolations
    y: input image
    w: interpolation kernels
    """
    y_padded = np.pad(y, ((2,2),(2,2)), 'constant', constant_values=0)
    size_padded = y_padded.shape

    #separate channels
    y_0 = y*op.mask[:, :, 0]
    y_1 = y*op.mask[:, :, 1]
    y_2 = y*op.mask[:, :, 2]

    for i in range(2,size_padded[0]-2):
        for j in range(2,size_padded[1]-2):

            # determine the location of the pixel
            if op.mask[i-2,j-2,1] == 1: # at G
                if op.mask[i-2,j-3,0] == 1: #R row
                    y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w3)
                    y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w7)
                else: #B row
                    y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w4)
                    y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w6)

            elif op.mask[i-2,j-2,0] == 1: # at R
                y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w1)
                y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w8)

            else: # at B
                y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w2)
                y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w5)

    y_res = np.zeros((y.shape[0], y.shape[1], 3))
    y_res[:, :, 0] = y_0
    y_res[:, :, 1] = y_1
    y_res[:, :, 2] = y_2

    return y_res



import numpy as np
def conv(A,B):
    return np.sum(A*B)
def quad_gradient_interpolation(y, op):
    """
    Second interpolation method: convolution with a 2D kernels (5x5) using multi-channel interpolations
    y: input image
    w: interpolation kernels
    """
    y_padded = np.pad(y, ((2,2),(2,2)), 'constant', constant_values=0)
    size_padded = y_padded.shape

    mask_padded = np.pad(op.mask, ((2,2),(2,2),(0,0)), 'constant', constant_values=-1)

    #separate channels
    y_0 = y*op.mask[:, :, 0]
    y_1 = y*op.mask[:, :, 1]
    y_2 = y*op.mask[:, :, 2]

    #interpolation kernels
    # green quad kernels
    wg11 = np.zeros((5,5))
    wg11[0,2] = -3
    wg11[1,2] = 13
    wg11[2,4] = 2
    wg11 += wg11.T
    wg11 = wg11/24

    # rotate the kernel
    wg12 = np.rot90(wg11,3)
    wg22 = np.rot90(wg11,2)
    wg21 = np.rot90(wg11,1)

    #red/blue quad kernels
        # center part 
    w_c11 = np.zeros((5,5))
    w_c11[0:2,0:2] = np.array([[-1,-1],[-1,9]])/6
    w_c21 = np.rot90(w_c11,1)
    w_c22 = np.rot90(w_c11,2)
    w_c12 = np.rot90(w_c11,3)
        # left part
    w_l11 = np.zeros((5,5))
    w_l11[0:2, 2] = np.array([-3,13])
    w_l11[4,2] = 2
    w_l11 = w_l11/12
    w_l22 = np.rot90(w_l11,2)
        # top part
    w_t11 = np.rot90(w_l11,1)
    w_t22 = np.rot90(w_t11,2)


    for i in range(2,size_padded[0]-2):
        for j in range(2,size_padded[1]-2):

            # determine the location of the pixel
            if mask_padded[i,j,1] == 1: # at G
                if mask_padded[i-1,j,0] + mask_padded[i+1,j,0] == 1: # at R column
                    if mask_padded[i,j-1,2] == 1:
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_t11)
                    elif mask_padded[i,j+1,2] == 1:
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_t22)
                    if mask_padded[i-1,j,0] == 1:
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_l11)
                    elif mask_padded[i+1,j,0] == 1:
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_l22)

                else:
                    if mask_padded[i,j-1,0] == 1:
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_t11)
                    elif mask_padded[i,j+1,0] == 1:
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_t22)
                    if mask_padded[i-1,j,2] == 1:
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_l11)
                    elif mask_padded[i+1,j,2] == 1:
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_l22)


            elif mask_padded[i,j,1] == 0: # at R or B
                if mask_padded[i-1,j,1] ==1:
                    if mask_padded[i,j-1,1] ==1:
                        y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],wg11)
                    else:
                        y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],wg12)
                else:
                    if mask_padded[i,j-1,1] ==1:
                        y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],wg21)
                    else:
                        y_1[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],wg22)
                
                if mask_padded[i,j,0] == 1: #at R
                    if mask_padded[i-1,j-1,2] ==1: #center (1,1)
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c11)
                    elif mask_padded[i-1,j+1,2] ==1: #center (1,2)
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c12)
                    elif mask_padded[i+1,j-1,2] ==1: #center (2,1)
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c21)
                    elif mask_padded[i+1,j+1,2] ==1: #center (2,2)
                        y_2[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c22)
                    
                elif mask_padded[i,j,2] == 1: #at B
                    if mask_padded[i-1,j-1,0] ==1: #center (1,1)
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c11)   
                    elif mask_padded[i-1,j+1,0] ==1: #center (1,2)
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c12)
                    elif mask_padded[i+1,j-1,0] ==1: #center (2,1)
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c21)
                    elif mask_padded[i+1,j+1,0] ==1: #center (2,2)
                        y_0[i-2,j-2] = conv(y_padded[i-2:i+3, j-2:j+3],w_c22)

    y_res = np.zeros((y.shape[0], y.shape[1], 3))
    y_res[:, :, 0] = y_0
    y_res[:, :, 1] = y_1
    y_res[:, :, 2] = y_2

    return y_res

from scipy import signal
def interpolate(y, op, mode = 'bayer'):
    """
    First interpolation method: convolution with a 2D kernel
    y: input image
    op: object of class CFA
    """
    #separate channels
    y_0 = y*op.mask[:, :, 0]
    y_1 = y*op.mask[:, :, 1]
    y_2 = y*op.mask[:, :, 2]

    #convolution 2D
    
    if mode == 'bayer':
        a = np.array([1,2,1])
    elif mode == 'quad_bayer':
        a = np.array([1,2,3,2,1])
    else:
        raise Exception('Mode not supported')
    
    # create the 2D kernel
    w = a[:, None] * a[None, :]
    w = w / np.sum(w)

    #interpolate green
    y_1_interpolated = signal.convolve2d(y_1, w, mode='same', boundary='fill', fillvalue=0)

    #interpolate red
    y_0_interpolated = signal.convolve2d(y_0, w, mode='same', boundary='fill', fillvalue=0)

    # interpolate blue
    y_2_interpolated = signal.convolve2d(y_2, w, mode='same', boundary='fill', fillvalue=0)

    y_res = np.zeros((y.shape[0], y.shape[1], 3))   
    y_res[:,:,0] = y_0_interpolated*4
    y_res[:,:,1] = y_1_interpolated*2
    y_res[:,:,2] = y_2_interpolated*4

    return y_res
