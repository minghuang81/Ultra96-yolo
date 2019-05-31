# Ultra96-yolo
Yolo object detection for Avnet/Xilinx Ultra96 board (Arm53 cores+fpga), based on a school assignment by deeplearning.ai.

This is a basic Yolo implementation in C, without hardware acceleration - Usage Instruction
===========================================================================================
    Performance: execution on mono-core takes about 2 minutes for detection in a 780x1024 image.
    
    1) Clone the repository onto Ultra96 board, in home directory "/home/root". After cloning:
         /home/root/Ultra96-yolo
    
    2) To compile all in Ultra96-yolo:
        cd /home/root/Ultra96-yolo/itf
        chmod a+x compile.sh
        ./compile.sh
        
            Display in console  is like:
            running build_ext
            ...
            aarch64-xilinx-linux-gcc ... -o /home/root/Ultra96-yolo/itf/convNetC.so	
    
    3) to run yolo with input taken from ./images, output placed into ./out:    
        cd /home/root/Ultra96-yolo/
        python test_C.py
    	
            Display in the console while running "test_C.py" is like:
            root@Ultra96:~/Ultra96-yolo# python test_C.py
             ---------------- input image: test.jpg in ./images, output image: test.jpg in ./out -------------
            Have detected 6 objects. consomed time: 2018-11-15 00:24:06.529748 -> 2018-11-15 00:25:48.236639
            
             ---------------- input image: 0019.jpg in ./images, output image: 0019.jpg in ./out -------------
            Have detected 2 objects. consomed time: 2018-11-15 00:25:48.346454 -> 2018-11-15 00:27:29.844256
    
    4) Description of the files
    	./README.md
    	./Drive.ai+Dataset+Sample+LICENSE.html
    	./LICENSE.txt
    	
    	./test_C.py: run this C implementation of Yolo on input images ./images/* to produce the out/*. 
    	             In the output image, the recognized objets are surrounded by boxes.
    	
    	./itf: interface for Python user applications such as 'test_C.py'.
    	./itf/compile.sh: compile Python interface module 'convNetC'.
    	./itf/convNetC_itf.c: source code of Python interface. Calling conv_net.c.
    	./itf/setup.py: 'makefile' to compile this C implementation, used by "compile".
    	./itf/__init__.py: empty file for python to take "itf" as a importable module.	
    		
    	./lib:  includes C implementation of convolutional net for inference.
    	./lib/conv_net.c: C subroutines used when making inference. Calling convNetC_fct.c.
    	./lib/conv_net.h: prototype for #include.
    	./lib/utils.py: image resizing, image reading and image saving in Python.
    	./lib/__init__.py: empty file for python to take "lib" as a importable module.	
    	
    	./model/convNetC_fct.c: C API to the model. Called by conv_net.c.
    	./model/convNetC_var.c: variables and static memory used for the model.
    	./model/convNetC_w.BE: weights of the model in big-endian.
    	./model/convNetC_w.LE: weights of the model in little-endian.
    
References
==========
1) python to c interface
The C programs (the collection of files suffixed by .c and .h) provides a call interface to Python programs (files suffixed by .py) through the C code in ./itf/convNetC_itf.c. C being the "service provider", the interface is thus programmed in C. In that file towards the end, the C functions are packed in a Python module named "convNetC", and that module contains only one Python function named "yolo". So the C programms have only a single access point (export) to Python, convNetC.yolo():
    Py_InitModule("convNetC", helloworld_funcs);
    static PyMethodDef helloworld_funcs[] = {
        {"yolo", (PyCFunction)convNetC_yolo, METH_VARARGS|METH_KEYWORDS, convNetC_docs},        
Then Python programs access the C functions by using the following Python instructions (see ./test_C.py):
    from itf import convNetC
    yolo_evals = convNetC.yolo(resized_image[0],anchors,len(class_names),image_shape)   
Tutorials and manuals: 
    https://docs.python.org/2/extending/extending.html
    https://docs.python.org/3/c-api/
Note the slight difference between Python 2.7 and 3.x, distinguished by "PY3K" keyword in ./itf/convNetC_itf.c

2) handling tensors as flat array
https://en.wikipedia.org/wiki/Row-_and_column-major_order  
C programs handle a multidimensional tensor in row-major order. For instance, a image is of 2x4 pixels, each pixel in RGB, having a height of 2 pixles and a width of 4, in another word the image has two rows and 4 columns. It cab be represented by the tensor of shape [2,4,3]:
    [
        [[11,12,13],[14,15,16],[17,18,19],[10,11,12]]
        [[21,22,23],[24,25,26],[27,28,29],[20,21,22]]
    ] 
where [11,12,13] is the top-left pixel and [20,21,22] is the bottom-right pixel.
The tesor is stored in memory as:
    lower_addr -> 11 12 13 14 15 16 17 18 19 10 11 12 21 22 23 24 25 26 27 28 29 20 21 22
and can be accessed in C code as lower_addr[0] .. lower_addr[23]
    
    
