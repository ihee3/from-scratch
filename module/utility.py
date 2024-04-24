import numpy as np

def im2col(array, filter_size:int, stride=1, pad = 0):

    padded_array = np.pad(array, [(0,0), (0,0), (pad,pad), (pad,pad)], mode='constant')

    filter_size = filter_size

    N, C, H, W = padded_array.shape

    out_h = (H  - filter_size)//stride + 1
    out_w = (W - filter_size)//stride + 1

    num_sliding_w = W - filter_size - (stride -1)
    num_slideing_h = H - filter_size - (stride -1)

    col_img = np.zeros((N, C, num_slideing_h, num_sliding_w, filter_size, filter_size))
    

    for y in range(filter_size):
        y_lim = y + out_h
        for x in range(filter_size):
            x_lim = x + out_w
            col_img[:,:,:,:,y,x] = padded_array[:,:,y:y_lim, x:x_lim]

    col_img = col_img.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)

    return col_img

def col2im(col, input_shape, filter_size:int, stride=1, pad=0):
    N, C, H, W = input_shape
    out_h = (H + 2*pad - filter_size)//stride + 1
    out_w = (W + 2*pad - filter_size)//stride + 1
    
    col = col.reshape(N, C, out_h, out_w, filter_size, filter_size).transpose(0, 3, 4, 5, 1, 2)
    img = np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))

    print(img.shape) 
