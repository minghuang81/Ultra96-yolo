#if !defined(CONV_NET_H)
#define CONV_NET_H

typedef unsigned int u32;
typedef unsigned char u8;
typedef int s32;
typedef char s8;
//typedef size_t u64;
typedef long long unsigned int u64;

// intermediate buffers for inference through conv net, sized to the max of layers
#define MAX_LAYER_OUTP_SIZE 11829248
// cumulatif nb of floats in all weights, to store model weights
#define TOTAL_WEIGHTS 50993897 
// max number of candidate boxes, subject to change depending on model
#define MAX_NB_OF_CANDIDATE_BOXES (19*19*5)
// max number of selected boxes
#define MAX_NB_OF_OUTPUT_BOXES (20)
extern int f_dbg; // To be switched on manually : 1=debug trace
extern float* buf_yolo_head;
	
// Python interface: X_pad = convNetC.zero_pad(X, pad)
float*  zero_pad(float *X, int n_H_prev, int n_W_prev, int n_C, int pad, float *X_pad, int *n_H, int *n_W);

// Python interface: Z = convNetC.conv_forward(A_prev, W, b, pad,stride})
float*  conv_forward(float* A_prev, int n_H_prev, int n_W_prev, int n_C_prev, 
                    float* W, int f1, int f2,
                    float* b, 
                    int stride, int pad,
                    float* Z, int n_H, int n_W, int n_C);
                    
// Python interface: Z = bn_forward(x,mv_mean,mv_var,epsilon,beta,gamma), output inplace of input
float* bn_forward(float* x, int n_H, int n_W, int n_C,
                float* mv_mean, float* mv_var, float epsilon, float* beta, float*gamma);

// Python interface: Z = leaky_re_lu(x,alpha), output inplace of input
float* leaky_re_lu(float* x, int len, float alpha);

// Python interface: Z = pool_forward(A_prev, f, stride, mode = "max"):
float* pool_forward(float* A_prev, int n_H_prev, int n_W_prev, int n_C_prev, int f, int stride, char* mode,
                    float* Z, int* o_n_H, int* o_n_W, int* o_n_C);
                    
// Python interface: Z = space_to_depth(x, block_size)
float* space_to_depth(float* x, int n_H_prev,int n_W_prev,int n_C_prev, int block_size, float* z);

// Python interface: Z = concatenate2(x1,x2, axis=-1)
float* concatenate2(float* arr1,int nd1,int *shape1, float* arr2,int nd2,int *shape2, int axis, float* Z);

// Python interface: Z = sigmoid(x), output inplace of input
float* sigmoid(float* x, int len) ;   

// Python interface: Z = softmax(x), output inplace of input
float* softmax(float* x, int m, int n) ;                 


#define CHECK(X)  if(!(X)) { printf(#X " not asserted at %s:%d",__FILE__,__LINE__);}
	

int isLE(void);
void conv_init(void);


#endif