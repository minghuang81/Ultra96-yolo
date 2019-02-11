#include "../lib/conv_net.h"
// intermediate buffers for inference through conv net, sized to the max of layers. defined in conv_net.h
float convNetC_buf1[MAX_LAYER_OUTP_SIZE], convNetC_buf2[MAX_LAYER_OUTP_SIZE], convNetC_buf3[MAX_LAYER_OUTP_SIZE], convNetC_buf4[MAX_LAYER_OUTP_SIZE];

// cumulatif nb of floats in all weights, to store model weights. defined in conv_net.h
float buf_w[TOTAL_WEIGHTS];

// C-conv, P-max poolig, N-BN, R-leaky relu, D-space_to_depth_x2, A-concAtenate 2 to 1, (1)-buf1, (2)-buf2,(3)-buf3

//	                                                                                                                                                                                                                                                                                                                                                    
//	(1)         (2) (1)         (2) (1)           (2)            (1)            (2)  (1)            (2)            (1)            (2)  (1)            (2)            (1)            (2)            (3)            (2)  (1)            (2)            (1)            (2)            (1)            (2)                       (1)                     buf1   
//	L1C-L2N-L3R-L4P-L5C-L6N-L7R-L8P-L9C-L10N-L11R-L12C-L13N-L14R-L15C-L16N-L17R-L18P-L19C-L20N-L21R-L22C-L23N-L24R-L25C-L26N-L27R-L28P-L29C-L30N-L31R-L32C-L33N-L34R-L35C-L36N-L37R-L38C-L39N-L40R-L41C-L42N-L43R-L44P-L45C-L46N-L47R-L48C-L49N-L50R-L51C-L52N-L53R-L54C-L55N-L56R-L57C-L58N-L59R-L60C-L61N-------L63R------L65C------L67N------L69R-| 
//	                                                                                                                                                                                                           |                                                                                                                                     |(2) (1)            (2) 
//	                                                                                                                                                                                                           |buf3                                                                                                                                 L70A-L71C-L72N-L73R-L74C-YoloHead                
//	                                                                                                                                                                                                           |                                                                                             (4)                           (3)       |               
//	                                                                                                                                                                                                           \---------------------------------------------------------------------------------------------L62C------L64N------L66R------L68D------| 
//	                                                                                                                                                                                                                                                                                                                                                buf3

float* buf_conv2d_1=convNetC_buf1; // [11829248] // L1
float* buf_max_pooling2d_1=convNetC_buf2; // [2957312] // L4
float* buf_conv2d_2=convNetC_buf1; // [5914624] // L5
float* buf_max_pooling2d_2=convNetC_buf2; // [1478656] // L8
float* buf_conv2d_3=convNetC_buf1; // [2957312] // L9
float* buf_conv2d_4=convNetC_buf2; // [1478656] // L12
float* buf_conv2d_5=convNetC_buf1; // [2957312] // L15
float* buf_max_pooling2d_3=convNetC_buf2; // [739328] // L18
float* buf_conv2d_6=convNetC_buf1; // [1478656] // L19
float* buf_conv2d_7=convNetC_buf2; // [739328] // L22
float* buf_conv2d_8=convNetC_buf1; // [1478656] // L25
float* buf_max_pooling2d_4=convNetC_buf2; // [369664] // L28
float* buf_conv2d_9=convNetC_buf1; // [739328] // L29
float* buf_conv2d_10=convNetC_buf2; // [369664] // L32
float* buf_conv2d_11=convNetC_buf1; // [739328] // L35
float* buf_conv2d_12=convNetC_buf2; // [369664] // L38
float* buf_conv2d_13=convNetC_buf3; // [739328] // L41: conv2d_13->L42 batch_normalization_13->L43 leaky_re_lu_13
float* buf_max_pooling2d_5=convNetC_buf2; // [184832] // L44
float* buf_conv2d_14=convNetC_buf1; // [369664] // L45
float* buf_conv2d_15=convNetC_buf2; // [184832] // L48
float* buf_conv2d_16=convNetC_buf1; // [369664] // L51
float* buf_conv2d_17=convNetC_buf2; // [184832] // L54
float* buf_conv2d_18=convNetC_buf1; // [369664] // L57
float* buf_conv2d_19=convNetC_buf2; // [369664] // L60
float* buf_conv2d_21=convNetC_buf4; // [92416] // L62: layer43 leaky_re_lu_13-> conv2d_21 -> layer64 BN
float* buf_conv2d_20=convNetC_buf1; // [369664] // L65: L60->L61->L63->L65->L67->L69-> L70
float* buf_space_to_depth_x2=convNetC_buf3; // [92416] // L68:                      -> L70
float* buf_concatenate_1=convNetC_buf2; // [369664] // L70
float* buf_conv2d_22=convNetC_buf1; // [369664] // L71
float* buf_conv2d_23=convNetC_buf2; // [153425] // L74

float* buf_yolo_head=convNetC_buf3;

#define _CRT_SECURE_NO_DEPRECATE
#include <stdio.h>
#include <stdlib.h>

extern int f_dbg;

// return 1 for little endian, 0 for big endian 
int isLE()
{
    volatile unsigned int i=0x01234567;
    return (*((unsigned char*)(&i))) == 0x67;
}

void conv_init()
{
	if (f_dbg) {
		printf("convNetC_buf1 0x%llX, buf2 0x%llX, buf3 0x%llX, buf_w 0x%llX\n", (u64)convNetC_buf1,(u64)convNetC_buf2,(u64)convNetC_buf3, (u64)buf_w);
	}
	// load model weights
	FILE *file=NULL;
	if (isLE()) {
		if (f_dbg) printf("conv_init little-endian\n");
		file = fopen("../model/convNetC_w.LE","rb");  // r for read, b for binary
		if (file == NULL) {
			file = fopen("./model/convNetC_w.LE","rb");  // in case test is run from parent directory
		}
	} else {
		if (f_dbg) printf("conv_init big-endian\n");
		file = fopen("../model/convNetC_w.BE","rb");  // r for read, b for binary
		if (file == NULL) {
			file = fopen("./model/convNetC_w.BE","rb");  // in case test is run from parent directory
		}
	}
	if (file == NULL) {
		printf("conv_init model file not found\n");
		exit(0);
	}
	
	fread(buf_w,TOTAL_WEIGHTS*4,1,file); // read 10 bytes to our buffer
	fclose(file);
	
//	printf("debug conv_init buf_w=");
//  for (int i=0;i<425;i++) printf(" %f", buf_w[i]);
//  printf("debug Z\n");
}
 