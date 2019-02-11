#include <math.h>
#include <stdio.h>
#include "conv_net.h"
/**
 *  These subroutines work only for row-major storage order
 */

// model specific constants
#define CONV_FORWARD_WORK_LEN (MAX_LAYER_OUTP_SIZE) //
#define N_C_MAX (2048) // this model has a max of 1024 channels

int f_dbg = 0; // To be switched on manually : 1=debug trace

// Python interface: X_pad = convNetC.zero_pad(X, pad)
float*  zero_pad(float *X, int n_H_prev, int n_W_prev, int n_C, int pad, float *X_pad, int *n_H, int *n_W)
{
    /**
    Pad with zeros all images of the dataset X. The padding is applied to the 
    height and width of an image.
    
    Argument:
    X -- python numpy array of shape (n_H, n_W, n_C) representing a image
    pad -- integer, amount of padding around each image on vertical and horizontal dimensions
    
    Returns:
    X_pad -- padded image of shape (n_H + 2*pad, n_W + 2*pad, n_C), = X_pad in input
    *n_H, *n_W: resulting height and width of X_pad. input and output have the same n_C 
    */
    float* X_pad_ori=X_pad; // save to return the same addr
    *n_H = n_H_prev+pad+pad;
    *n_W = n_W_prev+pad+pad;
    int h=0,w=0,c=0;
    
    for (h=0;h<pad;h++) 
        for (w=0;w<*n_W;w++) 
            for (c=0;c<n_C;c++)
                *X_pad++ = 0;
    for (h=pad;h<pad+n_H_prev;h++) {
        for (w=0;w<pad;w++) 
            for (c=0;c<n_C;c++)
                *X_pad++ = 0;
        for (w=pad;w<pad+n_W_prev;w++) 
            for (c=0;c<n_C;c++)
                *X_pad++ = *X++;
        for (w=pad+n_W_prev;w<*n_W;w++)
            for (c=0;c<n_C;c++) 
                *X_pad++ = 0;
    }
    for (h=pad+n_H_prev;h<*n_H;h++) 
        for (w=0;w<*n_W;w++) 
            for (c=0;c<n_C;c++)
                *X_pad++ = 0;
    return X_pad_ori;
}


float conv_single_step_ori(float *a_slice_prev_ori, int h_stride, float *W, int W_stride, int f1,int f2,int n_C_prev, float b)
{
    /**
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    h_stride -- element on next row in a_slice_prev = n_W_prev*n_C_prev
    W -- Weight parameters contained in a window - matrix of shape (f1, f2, n_C_prev)
    W_stride -- nb of floats to skip in memory for the next W element, = n_C
    b -- Bias parameters contained in a window - matrix of shape (1, 1, 1)
    
    Returns:
    s -- a scalar value, result of convolving the sliding window (W, b) on a slice x of the input data
    */

    // Element-wise product between a_slice and W. Do not add the bias yet.
    float* a_slice_prev; // iterate over rows
    float s = 0;
    for (int i=0;i<f1;i++, a_slice_prev_ori += h_stride) {
        a_slice_prev = a_slice_prev_ori;
        for (int j=0;j<f2;j++)
            for (int c=0;c<n_C_prev;c++,W += W_stride) {
                // equivalent to: s += (*a_slice_prev++) * W_ori[((i*f2+j)*n_C_prev+c)*W_stride];
                s += (*a_slice_prev++) * (*W);
            }
    }
    // Add bias b to the sum of all elements.
    s = s + b;

    return s;
}

// Python interface: Z = convNetC.conv_forward(A_prev, W, b, pad,stride})
float _work[CONV_FORWARD_WORK_LEN];    
//float* conv_forward_ori(float* A_prev, int n_H_prev, int n_W_prev, int n_C_prev, 
//                    float* W, int f1, int f2,
//                    float* b, 
//                    int stride, int pad,
//                    float* Z, int n_H, int n_W, int n_C)
//{
//    /**
//    Implements the forward propagation for a convolution function
//    
//    Arguments:
//    A_prev -- output activations of the previous layer, 
//              numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
//    W -- Weights, numpy array of shape (f1, f2, n_C_prev, n_C)
//    b -- Biases, numpy array of shape (n_C)
//    hparameters -- "stride" and "pad"
//        
//    Returns:
//    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C), = Z in input
//    */        
//    if (f_dbg) printf("conv_forward A_prev 0x%08llX,n_H_prev %d,n_W_prev %d,n_C_prev %d,W 0x%08llX,f1 %d,f2 %d,b 0x%08llX,stride %d,pad %d,Z 0x%08llX,n_H %d,n_W %d,n_C %d\n",
//            (u64)A_prev,n_H_prev,n_W_prev,n_C_prev,(u64)W,f1,f2,(u64)b,stride,pad,(u64)Z,n_H,n_W,n_C);
//    
//    // Create A_prev_pad by padding A_prev
//    float* Z_ori = Z;// save to return the same addr
//    float *a_prev_pad = A_prev;
//    int n_H_prev_pad=n_H_prev, n_W_prev_pad=n_W_prev;
//
//    if (pad != 0) {
//        a_prev_pad = _work;
//        zero_pad(A_prev, n_H_prev, n_W_prev, n_C_prev, pad, a_prev_pad, &n_H_prev_pad, &n_W_prev_pad);
//    }
//
//    
//    for (int h=0;h<n_H;h++)         // loop over vertical axis of the output volume
//        for (int w=0;w<n_W;w++)     // loop over horizontal axis of the output volume
//            for (int c=0;c<n_C;c++) // loop over channels (= #filters) of the output volume
//            {    
//                //printf("DEBUG h %d, w %d, c %d\n", h,w,c);
//                // Find the corners of the current "slice"
//                int vert_start = h*stride;
//                int vert_end = vert_start+f1; // the nearest pixel outside the slice
//                int horiz_start = w*stride;
//                int horiz_end = horiz_start+f2;
//                
//                // Use the corners to define the (3D) slice of a_prev_pad 
//                //a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
//                float *a_slice_prev = &a_prev_pad[(vert_start*n_W_prev_pad+horiz_start)*n_C_prev];
//                
//                // Convolve the (3D) slice with the correct filter W and bias b, 
//                // to get back one output neuron.
//                // W[c], W[c+n_C], W[c+2*n_C], .., correspond to W[ W[:,:,c]
//                *Z++ = conv_single_step(a_slice_prev, n_W_prev_pad*n_C_prev, &W[c], n_C, f1,f2,n_C_prev, b[c]);
//            } 
//    return Z_ori;                               
//} 

float* conv_single_step(float *a_slice_prev_ori, int h_stride, 
                       float *W, int f1,int f2,int n_C_prev, int n_C, float *b,
                       float* Z)
{
    /**
    Apply one filter defined by parameters W on a single slice (a_slice_prev) of the output activation 
    of the previous layer.
    
    Arguments:
    a_slice_prev -- slice of input data of shape (f, f, n_C_prev)
    h_stride -- element on next row in a_slice_prev = n_W_prev*n_C_prev
    W -- Weight parameters contained in a window - matrix of shape (f1, f2, n_C_prev, n_C)
    b -- Bias parameters contained in a window - matrix of shape (n_C)
    Z -- start position in memory where to put output
    
    Returns:
    Z -- next to the last output float
    */

    // Add the bias first.
    for (int c2=0;c2<n_C;c2++) {
        Z[c2] = b[c2];
    }
    // Element-wise product between a_slice and W.
    float* a_slice_prev; // iterate over rows
    for (int i=0;i<f1;i++, a_slice_prev_ori += h_stride) {
        a_slice_prev = a_slice_prev_ori;
        for (int j=0;j<f2;j++,a_slice_prev += n_C_prev)
            for (int c=0;c<n_C_prev;c++,W += n_C) {            
                for (int c2=0;c2<n_C;c2++) {
                    // equivalent to: s += (*a_slice_prev++) * W_ori[((i*f2+j)*n_C_prev+c)*W_stride];
                    Z[c2] += a_slice_prev[c] * W[c2];
                }
            }
    }

    return Z+n_C;
}
// Python interface: Z = convNetC.conv_forward(A_prev, W, b, pad,stride})
float _work[CONV_FORWARD_WORK_LEN];    
float* conv_forward(float* A_prev, int n_H_prev, int n_W_prev, int n_C_prev, 
                    float* W, int f1, int f2,
                    float* b, 
                    int stride, int pad,
                    float* Z, int n_H, int n_W, int n_C)
{
    /**
    Implements the forward propagation for a convolution function
    
    Arguments:
    A_prev -- output activations of the previous layer, 
              numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    W -- Weights, numpy array of shape (f1, f2, n_C_prev, n_C)
    b -- Biases, numpy array of shape (n_C)
    hparameters -- "stride" and "pad"
        
    Returns:
    Z -- conv output, numpy array of shape (m, n_H, n_W, n_C), = Z in input
    */        
    if (f_dbg) printf("conv_forward A_prev 0x%08llX,n_H_prev %d,n_W_prev %d,n_C_prev %d,W 0x%08llX,f1 %d,f2 %d,b 0x%08llX,stride %d,pad %d,Z 0x%08llX,n_H %d,n_W %d,n_C %d\n",
            (u64)A_prev,n_H_prev,n_W_prev,n_C_prev,(u64)W,f1,f2,(u64)b,stride,pad,(u64)Z,n_H,n_W,n_C);
    
    // Create A_prev_pad by padding A_prev
    float* Z_ori = Z;// save to return the same addr
    float *a_prev_pad = A_prev;
    int n_H_prev_pad=n_H_prev, n_W_prev_pad=n_W_prev;

    if (pad != 0) {
        a_prev_pad = _work;
        zero_pad(A_prev, n_H_prev, n_W_prev, n_C_prev, pad, a_prev_pad, &n_H_prev_pad, &n_W_prev_pad);
    }

    
    for (int h=0;h<n_H;h++)         // loop over vertical axis of the output volume
        for (int w=0;w<n_W;w++)     // loop over horizontal axis of the output volume
            //for (int c=0;c<n_C;c++) // loop over channels (= #filters) of the output volume
            {    
                //printf("DEBUG h %d, w %d, c %d\n", h,w,c);
                // Find the corners of the current "slice"
                int vert_start = h*stride;
                //int vert_end = vert_start+f1; // the nearest pixel outside the slice
                int horiz_start = w*stride;
                //int horiz_end = horiz_start+f2;
                
                // Use the corners to define the (3D) slice of a_prev_pad 
                //a_slice_prev = a_prev_pad[vert_start:vert_end,horiz_start:horiz_end,:]
                float *a_slice_prev = &a_prev_pad[(vert_start*n_W_prev_pad+horiz_start)*n_C_prev];
                
                // Convolve the (3D) slice with the correct filter W and bias b, 
                // to get back one output neuron.
                // W[c], W[c+n_C], W[c+2*n_C], .., correspond to W[ W[:,:,c]
                //*Z++ = conv_single_step(a_slice_prev, n_W_prev_pad*n_C_prev, &W[c], n_C, f1,f2,n_C_prev, b[c]);
                Z=conv_single_step(a_slice_prev, n_W_prev_pad*n_C_prev, W, f1,f2,n_C_prev, n_C, b, Z);
            } 
    return Z_ori;                               
}

// Python interface: Z = bn_forward(x,mv_mean,mv_var,epsilon,beta,gamma)
float* bn_forward(float* x, int n_H, int n_W, int n_C,
                  float* mv_mean, float* mv_var, float epsilon, float* beta, float*gamma)
{
    /**
    Normalization carried out over each of the ie 32 planes (channel), 
    with params (mean,var,beta,gamma) independent from each-other channels.
    Modify 'x' in-place !!!!
    arguments
        x -- input to layer i.e. (n_H, n_W, n_C) = (608,608,32) where 32 is nb of channels
        mv_mean -- moving average saved in the trained model (n_C,)
        mv_var -- moving variance saved in the trained model (n_C,)
        epsilon -- epsilon used in training to avoid dividing by zero (scalar)
        beta -- hyperparam obtained in training (n_C,)
        gamma -- hyperparam obtained in training (n_C,)
    return
        'x' modified in-place, i.e. (608,608,32), normalized in each of 32 channels
    */
    if (f_dbg) printf("bn_forward x 0x%08llX,n_H %d,n_W %d,n_C %d,mv_mean 0x%08llX,mv_var 0x%08llX,epsilon %f,beta 0x%08llX,gamma 0x%08llX\n",
            (u64)x,n_H,n_W,n_C,(u64)mv_mean,(u64)mv_var,epsilon,(u64)beta,(u64)gamma);    
    
    float* x_ori = x;// save to return the same addr
    float std[N_C_MAX];
    for (int c=0;c<n_C;c++) {
        std[c] = sqrtf(mv_var[c]+epsilon);
    }
    for (int i=0;i<n_H;i++) {
        for (int j=0;j<n_W;j++) {
            for (int c=0;c<n_C;c++) {
                *x = (*x-mv_mean[c])*gamma[c]/std[c]+beta[c];
                x++;
            }
        }
    }
    return x_ori;
}

// Python interface: Z = leaky_re_lu(x,alpha)
float* leaky_re_lu(float* x, int len, float alpha)
{
    /**
    Relu activation carried out over x , (n_H, n_W, n_C)
    Modify 'x' in-place !!!!
    arguments
        x -- input to layer i.e. (n_H, n_W, n_C) = (608,608,32) where 32 is nb of channels
        len -- x.size = n_H*n_W*n_C
    return
        'x' modified in-place, i.e. (608,608,32): v = v if v>0, v = alpha*v if v<0
    */
    if (f_dbg) printf("leaky_re_lu x 0x%08llX,len %d,alpha %f\n",(u64)x,len,alpha);    
    
    float* x_ori = x;// save to return the same addr
    for (int i=0;i<len; i++, x++) {
        if (*x < 0) *x = alpha * (*x);
    }
    return x_ori;
}

// Python interface: Z = pool_forward(A_prev, f, stride, mode = "max"):
float* pool_forward(float* A_prev, int n_H_prev, int n_W_prev, int n_C_prev, int f, int stride, char* mode, 
                    float* Z, int* o_n_H, int* o_n_W, int* o_n_C)
{
    /**
    Implements the forward pass of the pooling layer. 
    Max pooling pools within a channel, it does not pool across channels.
    
    Arguments:
    A_prev -- Input data, numpy array of shape (n_H_prev, n_W_prev, n_C_prev)
    hyper parameters -- filter size "f", "stride"
    mode -- the pooling mode you would like to use, defined as a char ('m'-"max" or 'a'-"average")
    Z, o_n_H, o_n_W, o_n_C -- output array data, dimension (h,w,c) of the array. Z may the same as x
    
    Returns:
    *o_n_H, *o_n_W, *o_n_C : dimension of output volume
    Z  -- output of the pool layer, a numpy array of shape (n_H, n_W, n_C=n_C_prev)
    */
    if (f_dbg) printf("pool_forward A_prev 0x%llX,n_H_prev %d,n_W_prev %d,n_C_prev %d,f %d,stride %d,mode %c,Z 0x%llX, ",
                      (u64)A_prev,n_H_prev,n_W_prev,n_C_prev,f,stride,*mode,(u64)Z);    
    
    float* Z_ori = Z;      // output
    if (*mode != 'm') {
        printf("pool_forward only supports max pooling ! \n");
        //return NULL;
    }
    // Define the dimensions of the output
    int n_H = (int)(1 + (n_H_prev - f) / stride);
    int n_W = (int)(1 + (n_W_prev - f) / stride);
    int n_C = n_C_prev;
    *o_n_H=n_H; *o_n_W=n_W; *o_n_C=n_C;      
    if (f_dbg) printf("out: n_H %d, n_W %d, n_C %d\n",n_H, n_W, n_C);

    // start poolingprintf
    int vert_start = 0;     // h*stride; 
    const int h_stride = stride*n_W_prev*n_C_prev;// memory disp. in input for vert. pixel disp in output  
    const int w_stride = stride*n_C_prev;         // memory disp. in input for hor. pixel disp in output
    const int a_row_stride = n_W_prev*n_C_prev;
    for (int h=0;h<n_H;h++,vert_start += h_stride) {           // loop on the vertical axis of the output volume
        int horiz_start = vert_start;   // position in input
        for (int w=0;w<n_W;w++,horiz_start += w_stride) {      // loop on the horizontal axis of the output volume
            // Use the corners to define the current slice on A_prev, starting channel c at 0.
            float* a_prev_slice = &A_prev[horiz_start]; // disp in input at channel 0
            for (int c=0;c<n_C;c++,a_prev_slice++) {   // loop over the channels - the same in output&input volume                                  
                // Compute the pooling operation on the slice. 
                float v = -1e9f;      
                float* a_row = a_prev_slice; // 1st row in the slice                            
                for (int i=0;i<f;i++,a_row += a_row_stride) {
                    float* a = a_row;
                    for (int j=0;j<f;j++, a += n_C_prev) {
                        if (*a > v)
                            v = *a;                       
                    }
                     // next row in the slice in input
                }
                //*Z++ = *a_prev_slice; 
                *Z++ = v;
            }
        }
    }   

    return Z_ori;   
}

// Python interface: Z = concatenate2(x1,x2, axis=-1)
float* concatenate2(float* arr1,int nd1,int *shape1, float* arr2,int nd2,int *shape2, int axis, float* Z)
{
//    Concatenate two arrays into one by merging its 'axis', in the order of arr1 then arr2. 
//    Argument:
//        arr1, arr2 -- numpy arrays of nd dimension, of shape[0:nd]
//        Z -- storage for the merged array
//    Returns:
//        Z -- A numpy array shape (...,shape1_at_axis+shape1_at_axis,...)
    if (f_dbg) {
        printf("concatenate2 arr1 0x%llX arr2 0x%llX Z 0x%llX nd1 %d nd1 %d, axis %d, ", (u64)arr1,(u64)arr2,(u64)Z,nd1,nd2,axis);
        for (int i=0;i<nd1;i++) printf("shape1[%d] %d ", i, shape1[i]);
        printf(", ");
        for (int i=0;i<nd2;i++) printf("shape2[%d] %d ", i, shape2[i]);
        printf("\n");
    }

    if (nd1 != nd2) {
        printf("concatenate2 nd1 %d != nd2 %d \n", nd1,nd2);
        return NULL; // will crash
    }
    if (axis == -1) axis = nd1-1;
    int m = 1; 
    for (int i=0;i<axis;i++) {
        if (shape1[i] != shape2[i]) {
            printf("concatenate2 shape1[%d] %d != shape2[%d] %d \n", i,shape1[i],i,shape2[i]);
            return NULL; // will crash
        }
        m *= shape1[i];
    }
    int n = 1;
    for (int i=axis+1;i<nd1;i++) {
        if (shape1[i] != shape2[i]) {
            printf("concatenate2 shape1[%d] %d != shape2[%d] %d \n", i,shape1[i],i,shape2[i]);
            return NULL; // will crash
        }
        n *= shape1[i];
    }
    int sz1 = n*shape1[axis];
    int sz2 = n*shape2[axis];
    if (f_dbg) printf("concatenate2 at axis %d: sz1 %d + sz2 %d = %d\n", axis,sz1,sz2,(sz1+sz2));

    
    float* Z_ori = Z;  
    for (int i=0;i<m;i++) {
        for (int j=0;j<sz1;j++) *Z++ = *arr1++;
        for (int j=0;j<sz2;j++) *Z++ = *arr2++;
    }
    return Z_ori;
}

// Python interface: Z = space_to_depth(x, block_size)
//    space_to_depth( array(6,6,2), block_size=2) => array(3, 3, 8)
//    x = np.reshape(range(0,72),(6,6,2)) : array(6,6,2) 
//    x= [    [[ 0,  1],[ 2,  3],[ 4,  5],[ 6,  7],[ 8,  9],[10, 11]],
//            [[12, 13],[14, 15],[16, 17],[18, 19],[20, 21],[22, 23]],
//            [[24, 25],[26, 27],[28, 29],[30, 31],[32, 33],[34, 35]],
//            [[36, 37],[38, 39],[40, 41],[42, 43],[44, 45],[46, 47]],
//            [[48, 49],[50, 51],[52, 53],[54, 55],[56, 57],[58, 59]],
//            [[60, 61],[62, 63],[64, 65],[66, 67],[68, 69],[70, 71]] ]
//
//    y = space_to_depth(x, n_H_prev=6,n_W_prev=6, n_C_prev=2, block_size=2) => array(3, 3, 8) 
//    y= [  [[ 0  1  2  3 12 13 14 15],[ 4  5  6  7 16 17 18 19],[ 8  9 10 11 20 21 22 23]]    
//          [[24 25 26 27 36 37 38 39],[28 29 30 31 40 41 42 43],[32 33 34 35 44 45 46 47]]    
//          [[48 49 50 51 60 61 62 63],[52 53 54 55 64 65 66 67],[56 57 58 59 68 69 70 71]]  ]    
//
//    z = space_to_depth(x, n_H_prev=6,n_W_prev=6, n_C_prev=2, block_size=3) => array(2, 2, 18)
//    z = [ [[ 0,  1,  2,  3,  4,  5, 12, 13, 14, 15, 16, 17, 24, 25, 26, 27, 28, 29],[ 6,  7,  8,  9, 10, 11, 18, 19, 20, 21, 22, 23, 30, 31, 32, 33, 34, 35]],
//          [[36, 37, 38, 39, 40, 41, 48, 49, 50, 51, 52, 53, 60, 61, 62, 63, 64, 65],[42, 43, 44, 45, 46, 47, 54, 55, 56, 57, 58, 59, 66, 67, 68, 69, 70, 71]] ]

float* space_to_depth(float* x, int n_H_prev,int n_W_prev,int n_C_prev, int block_size, float* z)
{
    /**   
     Combine multple 'pixels' spaces into one of higher channel depth.
     Based on Warren Weckesser/stackoverflow
     aguments
        x -- array(n_H_prev, n_W_prev, n_C_prev)
        block_size -- 
    return: 
        Z -- content of x re-arranged: 
             reduced height  = n_H_prev / block_size
             reduced width   = n_W_prev / block_size
             increased depth = n_C_prev * block_size * block_size
    */
    if (f_dbg) printf("space_to_depth x 0x%llX n_H_prev %d n_W_prev %d n_C_prev %d block_size %d z 0x%llX\n", (u64)x,n_H_prev,n_W_prev,n_C_prev,block_size,(u64)z);
    
    float* z_ori = z;
    int n_H = n_H_prev/block_size;
    int n_W = n_W_prev/block_size;
    int n_C = n_C_prev*block_size*block_size;
    int h_prev_row = n_W_prev*n_C_prev; 
    //vertical stride in input volume when looping over height of output volume
    int h_prev_stride = n_W_prev*block_size*n_C_prev; 
    //horizontal stride in input volume when looping over width of output volume
    int w_prev_stride = block_size*n_C_prev; 
    float* ph = x;
    for (int h=0;h<n_H;h++, ph += h_prev_stride) {           // loop in output volume
        float* pw = ph;                                      // position in input volume for output ch 0
        for (int w=0;w<n_W;w++, pw += w_prev_stride) {       // loop in output volume
            int c_prev = 0;
            int offset = 0;             // count total offset of memeory in input volume resulting from channel folding
            int offset_h = 0;           // count total offset of memeory in input volume resulting from row folding
            int delta_w = 0;            // displacement on the same row resulting from channel folding
            //int delta_h = 0;            // displacement to same row resulting from channel folding
            for (int c=0;c<n_C;c++) {   // loop in output volume
                *z++ = pw[offset + c_prev];
                if (++c_prev==n_C_prev) {
                    c_prev = 0;
                    if (++delta_w == block_size) {
                        delta_w = 0;
                        offset_h += h_prev_row; // next row in the input volume
                        offset = offset_h;
                    } else {
                        offset += n_C_prev;   // next point on the same row
                    }
                } 
            }
        }
    }
    return z_ori;
}  

//Python:
//def sigmoid(x):
//    x=1/(1+np.exp(-x))
//    return x
//
float* sigmoid(float* x_ori, int len) 
{
/**
    modify in-place the array x to sigmoid(x)
*/
    for (float* x=x_ori;x<(x_ori+len);x++) {
        *x = (float)(1/(1+exp(-*x)));
    }  
    return x_ori;     
}

//Python:
//def softmax(x):
//    """
//    Calculates the softmax for each row of the input x.
//    Good for a row vector and also for matrices of shape (n, m)
//    
//    Argument:
//        x -- A numpy matrix of shape (n,m)
//    Returns:
//        s -- A numpy matrix equal to the softmax of x, of shape (n,m)
//    """
//    x_exp = np.exp(x)
//    # Create a vector x_sum that sums each row of x_exp. 
//    x_sum = np.sum(x_exp,axis=-1, keepdims=True)
//    # Compute softmax(x) by dividing x_exp by x_sum. Automatically use numpy broadcasting.
//    s = x_exp/x_sum
//    return s
//
float* softmax(float* x, int m, int n) // m -- nomber of rows, n -- number of columns
{    
/**
    modify in-place the array x to softmax(x)
*/
    float* x_ori = x;
    for (int i=0;i<m;i++) {
        double s = 1e-12;
        float* save_x = x;
        for (int j=0;j<n;j++, x++) {
            *x = (float)exp(*x);
            s += *x;
        }
        x = save_x;
        for (int j=0;j<n;j++, x++) {
            *x = (float)(*x/s);
        }
    }
    return x_ori;
}












