// Python 3.6.5 :: Anaconda, Inc.
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include "numpy/arrayobject.h"
#include "conv_net.h"
#include "yolo.h"

extern float* conv_predict(float* image_data,int stoplayer,int* n_H,int* n_W,int* n_C);

// Python interface: Z = convNetC.yolo(image_data,anchors,num_classes,image_shape,stoplayer=-1)
// arguments:
//   image_data -- input image of shape (n_H, n_W, n_C) = (608,608,3) fixed per model
//   anchors -- anchors boxes of shape (nb of anchors, 2) where '2' elements are anchor box (width,height). nb of anchors=5
//   num_classes -- candidate object classes for detection. num_classes = 80
//   stoplayer  -- after execution at this layer, exit. '-1' designates the last output layer of conv net
// return:
//   Z -- of shape (None, 6) where 'None' is the number of detected objets. 
//        Each box that outline a detected object is described by 6 numbers:
//        - score (detection probability) = Z[:,0:1].flatten()
//        - box corners (top-left y,x, bottom right y,x) = Z[:,1:5] note: y=height; x=width
//        - object class = Z[:,5:6].astype('int').flatten()

static PyObject* convNetC_yolo(PyObject* self, PyObject *args, PyObject *keywds) {
    PyObject *arg1=NULL,*arg2=NULL,*arg3=NULL;
    int stoplayer = -1,num_classes;
    static char *kwlist[] = {"image_data","anchors","num_classes","image_shape","stoplayer", NULL};
    if (!PyArg_ParseTupleAndKeywords(args,keywds, "OOiO|i", kwlist, &arg1,&arg2,&num_classes,&arg3,&stoplayer)) {
        printf("PyArg_ParseTupleAndKeywords failed. Arg(s) are missing/incorrect\n");
        return NULL;
    }
    // image data
    PyArrayObject *arr1 = NULL;
    arr1 = (PyArrayObject*)PyArray_FROM_OTF(arg1, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr1 == NULL) {
        printf("missing arg 'image_data' ");
        return NULL;
    }    
    int nd = PyArray_NDIM(arr1);   //number of dimensions
    if (nd != 3) {
        printf("expecting PyArray_NDIM of A_prev being 3, but got %d ", nd);
        Py_XDECREF(arr1);
        return NULL;
    }
    //npy_intp *shape = PyArray_DIMS(arr1);  // npy_intp array of length nd showing length in each dim.
    //int n_H_prev=(int)shape[0], n_W_prev=(int)shape[1], n_C_prev=(int)shape[2]; 
    float* A_prev = (float*)PyArray_DATA(arr1);

    // anchors
    PyArrayObject *arr2 = NULL;
    arr2 = (PyArrayObject*)PyArray_FROM_OTF(arg2, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr2 == NULL) {
        Py_XDECREF(arr1); 
        printf("missing arg 'anchors' ");
        return NULL;
    }    
    int nd2 = PyArray_NDIM(arr2);   //number of dimensions
    if (nd2 != 2) {
        printf("expecting PyArray_NDIM of anchors being 2, but got %d ", nd2);
        Py_XDECREF(arr1); Py_XDECREF(arr2);
        return NULL;
    }
    npy_intp *shape2 = PyArray_DIMS(arr2);  // npy_intp array of length nd showing length in each dim.
    int num_anchors=(int)shape2[0], anchor_dim=(int)shape2[1]; 
    if (anchor_dim!=2) {
        printf("expecting shape of anchors being (:,2), but got (:,%d) ", anchor_dim);
        Py_XDECREF(arr1); Py_XDECREF(arr2);
        return NULL;
    }
    float* anchors = (float*)PyArray_DATA(arr2);

    // displayed image shape
    PyArrayObject *arr3 = NULL;
    arr3 = (PyArrayObject*)PyArray_FROM_OTF(arg3, NPY_FLOAT32, NPY_ARRAY_IN_ARRAY);
    if (arr3 == NULL) {
        printf("missing arg 'image_shape' ");
        Py_XDECREF(arr1); Py_XDECREF(arr2);
        return NULL;
    }    
    int nd3 = PyArray_NDIM(arr3);   //number of dimensions
    if (nd3 != 1) {
        printf("expecting PyArray_NDIM of image_shape being 1, but got %d ", nd3);
        Py_XDECREF(arr1); Py_XDECREF(arr2); Py_XDECREF(arr3);
        return NULL;
    }
    npy_intp *shape3 = PyArray_DIMS(arr3);  // npy_intp array of length nd showing length in each dim.
    int img_dim=(int)shape3[0]; 
    if (img_dim!=2) {
        printf("expecting shape of image_shape being (2), but got (%d) ", img_dim);
        Py_XDECREF(arr1); Py_XDECREF(arr2); Py_XDECREF(arr3);
        return NULL;
    }
    float* image_shape = (float*)PyArray_DATA(arr3); // h=image_shape[0], w=image_shape[1]
    
    // conv net output
    // --------------
        
    if (f_dbg) printf("calling  conv_predict(A_prev, stoplayer %d, &n_H, &n_W, &n_C)\n",stoplayer);
    
    int n_H,n_W,n_C;       
    // output array has shape of (19, 19, 425), hard-coded per model
    float* Z = conv_predict(A_prev,stoplayer,&n_H,&n_W,&n_C);
  

    // Output boxes from Yolo encoding
    // ------------------
    
    // copy to new buffer which will be modified in-place with yolo boxes
    memcpy(buf_yolo_head,Z,n_H*n_W*n_C*4);
    if (f_dbg) printf("calling  yolo_head_1(buf_yolo_head, n_H %d,n_W %d,n_C %d, anchors,num_anchors %d, num_classes %d)\n",n_H,n_W,n_C,num_anchors, num_classes);
    // output array has shape of (19, 19, 5, 85), hard-coded per model
    yolo_head_1(buf_yolo_head,n_H,n_W,n_C, anchors,num_anchors, num_classes);    
    if (f_dbg) printf("return from  yolo_head_1()\n");

    // Yolo evalution
    // ----------------------
    
    float* out_boxes;   
    int max_boxes=10;
    float score_threshold=.6f; float iou_threshold=.5f;
    if (f_dbg) printf("calling  yolo_eval()\n");
    int boxes_cnt = yolo_eval(buf_yolo_head,n_H,n_W,num_anchors, num_classes,&out_boxes,
                 (int)image_shape[0], (int)image_shape[1], 
                 max_boxes, score_threshold, iou_threshold);
    if (f_dbg) printf("return from  yolo_eval() boxes_cnt=%d\n",boxes_cnt);
                 
    // output array has shape of (None, 6), each (,-) describes a box
    npy_intp dims3[2] = {boxes_cnt,6};
    PyArrayObject *arr6 = (PyArrayObject *)PyArray_SimpleNewFromData(2, dims3, NPY_FLOAT,out_boxes);   

    Py_DECREF(arr1);
    return Py_BuildValue("O", arr6);            
}

// the module's method table
static char convNetC_docs[] = "C extention of conv Net\n";
static PyMethodDef helloworld_funcs[] = {
   {"yolo", (PyCFunction)convNetC_yolo, METH_VARARGS|METH_KEYWORDS, convNetC_docs},
   {NULL}
};

#ifdef PY3K
// module definition structure for python3
// --------------------------------------

// Our Module Definition struct
static struct PyModuleDef myModule = {
    PyModuleDef_HEAD_INIT,
    "convNetC", // name of module
    "C extention of convNet", // comment
    -1, // all state in global variables
    helloworld_funcs
};

// create module - module definition above
PyMODINIT_FUNC PyInit_convNetC(void) {
    import_array();
    //import_ufunc();
    //Py_InitModule3("helloworld", helloworld_funcs,"example!");
    return PyModule_Create(&myModule);
}

#else
// module definition structure for python2.7
// --------------------------------------

// module initializer for python2
PyMODINIT_FUNC initconvNetC(void) {
    import_array();
    Py_InitModule("convNetC", helloworld_funcs);
}

#endif