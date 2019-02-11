#include "../lib/conv_net.h"
#include "convNetC_var.c"
#include <time.h>   	// for clock_t, clock(), CLOCKS_PER_SEC
float* conv_predict(float* image_data,int stoplayer,int *n_H,int *n_W,int *n_C) 
{
////   arguments:
//   image_data -- tensor of shape (608, 608, 3) hard-coded per model
//   stoplayer -- stop at and return the output of this layer. -1 means the last layer
//   returns:
//   lastConvLayer -- output from the last conv net layer, of shape (n_H,n_W,n_C)//

	clock_t begin=0;
	conv_init();
	if (stoplayer==-1) stoplayer=74;
	float* input_1=image_data;

	// layer1 conv2d_1 (None, 608, 608, 3)->(None, 608, 608, 32)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer1 conv2d_1 (None, 608, 608, 3)->(None, 608, 608, 32)\n");
	if (f_dbg) begin = clock();

	float* conv2d_1_w=&buf_w[0];
	float* conv2d_1_b=&buf_w[864];
	float* conv2d_1 = conv_forward(input_1,608,608,3,conv2d_1_w,3,3,conv2d_1_b,1,1,buf_conv2d_1,608,608,32);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=608;*n_W=608;*n_C=32;
	if (stoplayer==1) return conv2d_1;


	// layer2 batch_normalization_1 (None, 608, 608, 32)->(None, 608, 608, 32)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer2 batch_normalization_1 (None, 608, 608, 32)->(None, 608, 608, 32)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_1_m=&buf_w[896];
	float* batch_normalization_1_v=&buf_w[928];
	float batch_normalization_1_e=0.001f;
	float* batch_normalization_1_b=&buf_w[960];
	float* batch_normalization_1_g=&buf_w[992];
	float* batch_normalization_1=bn_forward(conv2d_1,608,608,32,batch_normalization_1_m,batch_normalization_1_v,batch_normalization_1_e,batch_normalization_1_b,batch_normalization_1_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=608;*n_W=608;*n_C=32;
	if (stoplayer==2) return batch_normalization_1;


	// layer3 leaky_re_lu_1 (None, 608, 608, 32)->(None, 608, 608, 32)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer3 leaky_re_lu_1 (None, 608, 608, 32)->(None, 608, 608, 32)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_1=leaky_re_lu(batch_normalization_1,11829248,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=608;*n_W=608;*n_C=32;
	if (stoplayer==3) return leaky_re_lu_1;


	// layer4 max_pooling2d_1 (None, 608, 608, 32)->(None, 304, 304, 32)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer4 max_pooling2d_1 (None, 608, 608, 32)->(None, 304, 304, 32)\n");
	if (f_dbg) begin = clock();

	int o_n_H_4,o_n_W_4,o_n_C_4;
	float* max_pooling2d_1 = pool_forward(leaky_re_lu_1,608,608,32,2,2,"m",buf_max_pooling2d_1,&o_n_H_4,&o_n_W_4,&o_n_C_4);
	CHECK(o_n_H_4==304 && o_n_W_4==304 );
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=304;*n_W=304;*n_C=32;
	if (stoplayer==4) return max_pooling2d_1;


	// layer5 conv2d_2 (None, 304, 304, 32)->(None, 304, 304, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer5 conv2d_2 (None, 304, 304, 32)->(None, 304, 304, 64)\n");
	if (f_dbg) begin = clock();

	float* conv2d_2_w=&buf_w[1024];
	float* conv2d_2_b=&buf_w[19456];
	float* conv2d_2 = conv_forward(max_pooling2d_1,304,304,32,conv2d_2_w,3,3,conv2d_2_b,1,1,buf_conv2d_2,304,304,64);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=304;*n_W=304;*n_C=64;
	if (stoplayer==5) return conv2d_2;


	// layer6 batch_normalization_2 (None, 304, 304, 64)->(None, 304, 304, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer6 batch_normalization_2 (None, 304, 304, 64)->(None, 304, 304, 64)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_2_m=&buf_w[19520];
	float* batch_normalization_2_v=&buf_w[19584];
	float batch_normalization_2_e=0.001f;
	float* batch_normalization_2_b=&buf_w[19648];
	float* batch_normalization_2_g=&buf_w[19712];
	float* batch_normalization_2=bn_forward(conv2d_2,304,304,64,batch_normalization_2_m,batch_normalization_2_v,batch_normalization_2_e,batch_normalization_2_b,batch_normalization_2_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=304;*n_W=304;*n_C=64;
	if (stoplayer==6) return batch_normalization_2;


	// layer7 leaky_re_lu_2 (None, 304, 304, 64)->(None, 304, 304, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer7 leaky_re_lu_2 (None, 304, 304, 64)->(None, 304, 304, 64)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_2=leaky_re_lu(batch_normalization_2,5914624,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=304;*n_W=304;*n_C=64;
	if (stoplayer==7) return leaky_re_lu_2;


	// layer8 max_pooling2d_2 (None, 304, 304, 64)->(None, 152, 152, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer8 max_pooling2d_2 (None, 304, 304, 64)->(None, 152, 152, 64)\n");
	if (f_dbg) begin = clock();

	int o_n_H_8,o_n_W_8,o_n_C_8;
	float* max_pooling2d_2 = pool_forward(leaky_re_lu_2,304,304,64,2,2,"m",buf_max_pooling2d_2,&o_n_H_8,&o_n_W_8,&o_n_C_8);
	CHECK(o_n_H_8==152 && o_n_W_8==152 );
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=64;
	if (stoplayer==8) return max_pooling2d_2;


	// layer9 conv2d_3 (None, 152, 152, 64)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer9 conv2d_3 (None, 152, 152, 64)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* conv2d_3_w=&buf_w[19776];
	float* conv2d_3_b=&buf_w[93504];
	float* conv2d_3 = conv_forward(max_pooling2d_2,152,152,64,conv2d_3_w,3,3,conv2d_3_b,1,1,buf_conv2d_3,152,152,128);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==9) return conv2d_3;


	// layer10 batch_normalization_3 (None, 152, 152, 128)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer10 batch_normalization_3 (None, 152, 152, 128)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_3_m=&buf_w[93632];
	float* batch_normalization_3_v=&buf_w[93760];
	float batch_normalization_3_e=0.001f;
	float* batch_normalization_3_b=&buf_w[93888];
	float* batch_normalization_3_g=&buf_w[94016];
	float* batch_normalization_3=bn_forward(conv2d_3,152,152,128,batch_normalization_3_m,batch_normalization_3_v,batch_normalization_3_e,batch_normalization_3_b,batch_normalization_3_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==10) return batch_normalization_3;


	// layer11 leaky_re_lu_3 (None, 152, 152, 128)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer11 leaky_re_lu_3 (None, 152, 152, 128)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_3=leaky_re_lu(batch_normalization_3,2957312,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==11) return leaky_re_lu_3;


	// layer12 conv2d_4 (None, 152, 152, 128)->(None, 152, 152, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer12 conv2d_4 (None, 152, 152, 128)->(None, 152, 152, 64)\n");
	if (f_dbg) begin = clock();

	float* conv2d_4_w=&buf_w[94144];
	float* conv2d_4_b=&buf_w[102336];
	float* conv2d_4 = conv_forward(leaky_re_lu_3,152,152,128,conv2d_4_w,1,1,conv2d_4_b,1,0,buf_conv2d_4,152,152,64);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=64;
	if (stoplayer==12) return conv2d_4;


	// layer13 batch_normalization_4 (None, 152, 152, 64)->(None, 152, 152, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer13 batch_normalization_4 (None, 152, 152, 64)->(None, 152, 152, 64)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_4_m=&buf_w[102400];
	float* batch_normalization_4_v=&buf_w[102464];
	float batch_normalization_4_e=0.001f;
	float* batch_normalization_4_b=&buf_w[102528];
	float* batch_normalization_4_g=&buf_w[102592];
	float* batch_normalization_4=bn_forward(conv2d_4,152,152,64,batch_normalization_4_m,batch_normalization_4_v,batch_normalization_4_e,batch_normalization_4_b,batch_normalization_4_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=64;
	if (stoplayer==13) return batch_normalization_4;


	// layer14 leaky_re_lu_4 (None, 152, 152, 64)->(None, 152, 152, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer14 leaky_re_lu_4 (None, 152, 152, 64)->(None, 152, 152, 64)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_4=leaky_re_lu(batch_normalization_4,1478656,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=64;
	if (stoplayer==14) return leaky_re_lu_4;


	// layer15 conv2d_5 (None, 152, 152, 64)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer15 conv2d_5 (None, 152, 152, 64)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* conv2d_5_w=&buf_w[102656];
	float* conv2d_5_b=&buf_w[176384];
	float* conv2d_5 = conv_forward(leaky_re_lu_4,152,152,64,conv2d_5_w,3,3,conv2d_5_b,1,1,buf_conv2d_5,152,152,128);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==15) return conv2d_5;


	// layer16 batch_normalization_5 (None, 152, 152, 128)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer16 batch_normalization_5 (None, 152, 152, 128)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_5_m=&buf_w[176512];
	float* batch_normalization_5_v=&buf_w[176640];
	float batch_normalization_5_e=0.001f;
	float* batch_normalization_5_b=&buf_w[176768];
	float* batch_normalization_5_g=&buf_w[176896];
	float* batch_normalization_5=bn_forward(conv2d_5,152,152,128,batch_normalization_5_m,batch_normalization_5_v,batch_normalization_5_e,batch_normalization_5_b,batch_normalization_5_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==16) return batch_normalization_5;


	// layer17 leaky_re_lu_5 (None, 152, 152, 128)->(None, 152, 152, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer17 leaky_re_lu_5 (None, 152, 152, 128)->(None, 152, 152, 128)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_5=leaky_re_lu(batch_normalization_5,2957312,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=152;*n_W=152;*n_C=128;
	if (stoplayer==17) return leaky_re_lu_5;


	// layer18 max_pooling2d_3 (None, 152, 152, 128)->(None, 76, 76, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer18 max_pooling2d_3 (None, 152, 152, 128)->(None, 76, 76, 128)\n");
	if (f_dbg) begin = clock();

	int o_n_H_18,o_n_W_18,o_n_C_18;
	float* max_pooling2d_3 = pool_forward(leaky_re_lu_5,152,152,128,2,2,"m",buf_max_pooling2d_3,&o_n_H_18,&o_n_W_18,&o_n_C_18);
	CHECK(o_n_H_18==76 && o_n_W_18==76 );
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=128;
	if (stoplayer==18) return max_pooling2d_3;


	// layer19 conv2d_6 (None, 76, 76, 128)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer19 conv2d_6 (None, 76, 76, 128)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* conv2d_6_w=&buf_w[177024];
	float* conv2d_6_b=&buf_w[471936];
	float* conv2d_6 = conv_forward(max_pooling2d_3,76,76,128,conv2d_6_w,3,3,conv2d_6_b,1,1,buf_conv2d_6,76,76,256);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==19) return conv2d_6;


	// layer20 batch_normalization_6 (None, 76, 76, 256)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer20 batch_normalization_6 (None, 76, 76, 256)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_6_m=&buf_w[472192];
	float* batch_normalization_6_v=&buf_w[472448];
	float batch_normalization_6_e=0.001f;
	float* batch_normalization_6_b=&buf_w[472704];
	float* batch_normalization_6_g=&buf_w[472960];
	float* batch_normalization_6=bn_forward(conv2d_6,76,76,256,batch_normalization_6_m,batch_normalization_6_v,batch_normalization_6_e,batch_normalization_6_b,batch_normalization_6_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==20) return batch_normalization_6;


	// layer21 leaky_re_lu_6 (None, 76, 76, 256)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer21 leaky_re_lu_6 (None, 76, 76, 256)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_6=leaky_re_lu(batch_normalization_6,1478656,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==21) return leaky_re_lu_6;


	// layer22 conv2d_7 (None, 76, 76, 256)->(None, 76, 76, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer22 conv2d_7 (None, 76, 76, 256)->(None, 76, 76, 128)\n");
	if (f_dbg) begin = clock();

	float* conv2d_7_w=&buf_w[473216];
	float* conv2d_7_b=&buf_w[505984];
	float* conv2d_7 = conv_forward(leaky_re_lu_6,76,76,256,conv2d_7_w,1,1,conv2d_7_b,1,0,buf_conv2d_7,76,76,128);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=128;
	if (stoplayer==22) return conv2d_7;


	// layer23 batch_normalization_7 (None, 76, 76, 128)->(None, 76, 76, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer23 batch_normalization_7 (None, 76, 76, 128)->(None, 76, 76, 128)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_7_m=&buf_w[506112];
	float* batch_normalization_7_v=&buf_w[506240];
	float batch_normalization_7_e=0.001f;
	float* batch_normalization_7_b=&buf_w[506368];
	float* batch_normalization_7_g=&buf_w[506496];
	float* batch_normalization_7=bn_forward(conv2d_7,76,76,128,batch_normalization_7_m,batch_normalization_7_v,batch_normalization_7_e,batch_normalization_7_b,batch_normalization_7_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=128;
	if (stoplayer==23) return batch_normalization_7;


	// layer24 leaky_re_lu_7 (None, 76, 76, 128)->(None, 76, 76, 128)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer24 leaky_re_lu_7 (None, 76, 76, 128)->(None, 76, 76, 128)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_7=leaky_re_lu(batch_normalization_7,739328,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=128;
	if (stoplayer==24) return leaky_re_lu_7;


	// layer25 conv2d_8 (None, 76, 76, 128)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer25 conv2d_8 (None, 76, 76, 128)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* conv2d_8_w=&buf_w[506624];
	float* conv2d_8_b=&buf_w[801536];
	float* conv2d_8 = conv_forward(leaky_re_lu_7,76,76,128,conv2d_8_w,3,3,conv2d_8_b,1,1,buf_conv2d_8,76,76,256);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==25) return conv2d_8;


	// layer26 batch_normalization_8 (None, 76, 76, 256)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer26 batch_normalization_8 (None, 76, 76, 256)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_8_m=&buf_w[801792];
	float* batch_normalization_8_v=&buf_w[802048];
	float batch_normalization_8_e=0.001f;
	float* batch_normalization_8_b=&buf_w[802304];
	float* batch_normalization_8_g=&buf_w[802560];
	float* batch_normalization_8=bn_forward(conv2d_8,76,76,256,batch_normalization_8_m,batch_normalization_8_v,batch_normalization_8_e,batch_normalization_8_b,batch_normalization_8_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==26) return batch_normalization_8;


	// layer27 leaky_re_lu_8 (None, 76, 76, 256)->(None, 76, 76, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer27 leaky_re_lu_8 (None, 76, 76, 256)->(None, 76, 76, 256)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_8=leaky_re_lu(batch_normalization_8,1478656,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=76;*n_W=76;*n_C=256;
	if (stoplayer==27) return leaky_re_lu_8;


	// layer28 max_pooling2d_4 (None, 76, 76, 256)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer28 max_pooling2d_4 (None, 76, 76, 256)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	int o_n_H_28,o_n_W_28,o_n_C_28;
	float* max_pooling2d_4 = pool_forward(leaky_re_lu_8,76,76,256,2,2,"m",buf_max_pooling2d_4,&o_n_H_28,&o_n_W_28,&o_n_C_28);
	CHECK(o_n_H_28==38 && o_n_W_28==38 );
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==28) return max_pooling2d_4;


	// layer29 conv2d_9 (None, 38, 38, 256)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer29 conv2d_9 (None, 38, 38, 256)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* conv2d_9_w=&buf_w[802816];
	float* conv2d_9_b=&buf_w[1982464];
	float* conv2d_9 = conv_forward(max_pooling2d_4,38,38,256,conv2d_9_w,3,3,conv2d_9_b,1,1,buf_conv2d_9,38,38,512);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==29) return conv2d_9;


	// layer30 batch_normalization_9 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer30 batch_normalization_9 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_9_m=&buf_w[1982976];
	float* batch_normalization_9_v=&buf_w[1983488];
	float batch_normalization_9_e=0.001f;
	float* batch_normalization_9_b=&buf_w[1984000];
	float* batch_normalization_9_g=&buf_w[1984512];
	float* batch_normalization_9=bn_forward(conv2d_9,38,38,512,batch_normalization_9_m,batch_normalization_9_v,batch_normalization_9_e,batch_normalization_9_b,batch_normalization_9_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==30) return batch_normalization_9;


	// layer31 leaky_re_lu_9 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer31 leaky_re_lu_9 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_9=leaky_re_lu(batch_normalization_9,739328,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==31) return leaky_re_lu_9;


	// layer32 conv2d_10 (None, 38, 38, 512)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer32 conv2d_10 (None, 38, 38, 512)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* conv2d_10_w=&buf_w[1985024];
	float* conv2d_10_b=&buf_w[2116096];
	float* conv2d_10 = conv_forward(leaky_re_lu_9,38,38,512,conv2d_10_w,1,1,conv2d_10_b,1,0,buf_conv2d_10,38,38,256);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==32) return conv2d_10;


	// layer33 batch_normalization_10 (None, 38, 38, 256)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer33 batch_normalization_10 (None, 38, 38, 256)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_10_m=&buf_w[2116352];
	float* batch_normalization_10_v=&buf_w[2116608];
	float batch_normalization_10_e=0.001f;
	float* batch_normalization_10_b=&buf_w[2116864];
	float* batch_normalization_10_g=&buf_w[2117120];
	float* batch_normalization_10=bn_forward(conv2d_10,38,38,256,batch_normalization_10_m,batch_normalization_10_v,batch_normalization_10_e,batch_normalization_10_b,batch_normalization_10_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==33) return batch_normalization_10;


	// layer34 leaky_re_lu_10 (None, 38, 38, 256)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer34 leaky_re_lu_10 (None, 38, 38, 256)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_10=leaky_re_lu(batch_normalization_10,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==34) return leaky_re_lu_10;


	// layer35 conv2d_11 (None, 38, 38, 256)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer35 conv2d_11 (None, 38, 38, 256)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* conv2d_11_w=&buf_w[2117376];
	float* conv2d_11_b=&buf_w[3297024];
	float* conv2d_11 = conv_forward(leaky_re_lu_10,38,38,256,conv2d_11_w,3,3,conv2d_11_b,1,1,buf_conv2d_11,38,38,512);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==35) return conv2d_11;


	// layer36 batch_normalization_11 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer36 batch_normalization_11 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_11_m=&buf_w[3297536];
	float* batch_normalization_11_v=&buf_w[3298048];
	float batch_normalization_11_e=0.001f;
	float* batch_normalization_11_b=&buf_w[3298560];
	float* batch_normalization_11_g=&buf_w[3299072];
	float* batch_normalization_11=bn_forward(conv2d_11,38,38,512,batch_normalization_11_m,batch_normalization_11_v,batch_normalization_11_e,batch_normalization_11_b,batch_normalization_11_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==36) return batch_normalization_11;


	// layer37 leaky_re_lu_11 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer37 leaky_re_lu_11 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_11=leaky_re_lu(batch_normalization_11,739328,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==37) return leaky_re_lu_11;


	// layer38 conv2d_12 (None, 38, 38, 512)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer38 conv2d_12 (None, 38, 38, 512)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* conv2d_12_w=&buf_w[3299584];
	float* conv2d_12_b=&buf_w[3430656];
	float* conv2d_12 = conv_forward(leaky_re_lu_11,38,38,512,conv2d_12_w,1,1,conv2d_12_b,1,0,buf_conv2d_12,38,38,256);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==38) return conv2d_12;


	// layer39 batch_normalization_12 (None, 38, 38, 256)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer39 batch_normalization_12 (None, 38, 38, 256)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_12_m=&buf_w[3430912];
	float* batch_normalization_12_v=&buf_w[3431168];
	float batch_normalization_12_e=0.001f;
	float* batch_normalization_12_b=&buf_w[3431424];
	float* batch_normalization_12_g=&buf_w[3431680];
	float* batch_normalization_12=bn_forward(conv2d_12,38,38,256,batch_normalization_12_m,batch_normalization_12_v,batch_normalization_12_e,batch_normalization_12_b,batch_normalization_12_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==39) return batch_normalization_12;


	// layer40 leaky_re_lu_12 (None, 38, 38, 256)->(None, 38, 38, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer40 leaky_re_lu_12 (None, 38, 38, 256)->(None, 38, 38, 256)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_12=leaky_re_lu(batch_normalization_12,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=256;
	if (stoplayer==40) return leaky_re_lu_12;


	// layer41 conv2d_13 (None, 38, 38, 256)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer41 conv2d_13 (None, 38, 38, 256)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* conv2d_13_w=&buf_w[3431936];
	float* conv2d_13_b=&buf_w[4611584];
	float* conv2d_13 = conv_forward(leaky_re_lu_12,38,38,256,conv2d_13_w,3,3,conv2d_13_b,1,1,buf_conv2d_13,38,38,512);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==41) return conv2d_13;


	// layer42 batch_normalization_13 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer42 batch_normalization_13 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_13_m=&buf_w[4612096];
	float* batch_normalization_13_v=&buf_w[4612608];
	float batch_normalization_13_e=0.001f;
	float* batch_normalization_13_b=&buf_w[4613120];
	float* batch_normalization_13_g=&buf_w[4613632];
	float* batch_normalization_13=bn_forward(conv2d_13,38,38,512,batch_normalization_13_m,batch_normalization_13_v,batch_normalization_13_e,batch_normalization_13_b,batch_normalization_13_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==42) return batch_normalization_13;


	// layer43 leaky_re_lu_13 (None, 38, 38, 512)->(None, 38, 38, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer43 leaky_re_lu_13 (None, 38, 38, 512)->(None, 38, 38, 512)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_13=leaky_re_lu(batch_normalization_13,739328,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=512;
	if (stoplayer==43) return leaky_re_lu_13;


	// layer44 max_pooling2d_5 (None, 38, 38, 512)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer44 max_pooling2d_5 (None, 38, 38, 512)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	int o_n_H_44,o_n_W_44,o_n_C_44;
	float* max_pooling2d_5 = pool_forward(leaky_re_lu_13,38,38,512,2,2,"m",buf_max_pooling2d_5,&o_n_H_44,&o_n_W_44,&o_n_C_44);
	CHECK(o_n_H_44==19 && o_n_W_44==19 );
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==44) return max_pooling2d_5;


	// layer45 conv2d_14 (None, 19, 19, 512)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer45 conv2d_14 (None, 19, 19, 512)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_14_w=&buf_w[4614144];
	float* conv2d_14_b=&buf_w[9332736];
	float* conv2d_14 = conv_forward(max_pooling2d_5,19,19,512,conv2d_14_w,3,3,conv2d_14_b,1,1,buf_conv2d_14,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==45) return conv2d_14;


	// layer46 batch_normalization_14 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer46 batch_normalization_14 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_14_m=&buf_w[9333760];
	float* batch_normalization_14_v=&buf_w[9334784];
	float batch_normalization_14_e=0.001f;
	float* batch_normalization_14_b=&buf_w[9335808];
	float* batch_normalization_14_g=&buf_w[9336832];
	float* batch_normalization_14=bn_forward(conv2d_14,19,19,1024,batch_normalization_14_m,batch_normalization_14_v,batch_normalization_14_e,batch_normalization_14_b,batch_normalization_14_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==46) return batch_normalization_14;


	// layer47 leaky_re_lu_14 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer47 leaky_re_lu_14 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_14=leaky_re_lu(batch_normalization_14,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==47) return leaky_re_lu_14;


	// layer48 conv2d_15 (None, 19, 19, 1024)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer48 conv2d_15 (None, 19, 19, 1024)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* conv2d_15_w=&buf_w[9337856];
	float* conv2d_15_b=&buf_w[9862144];
	float* conv2d_15 = conv_forward(leaky_re_lu_14,19,19,1024,conv2d_15_w,1,1,conv2d_15_b,1,0,buf_conv2d_15,19,19,512);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==48) return conv2d_15;


	// layer49 batch_normalization_15 (None, 19, 19, 512)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer49 batch_normalization_15 (None, 19, 19, 512)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_15_m=&buf_w[9862656];
	float* batch_normalization_15_v=&buf_w[9863168];
	float batch_normalization_15_e=0.001f;
	float* batch_normalization_15_b=&buf_w[9863680];
	float* batch_normalization_15_g=&buf_w[9864192];
	float* batch_normalization_15=bn_forward(conv2d_15,19,19,512,batch_normalization_15_m,batch_normalization_15_v,batch_normalization_15_e,batch_normalization_15_b,batch_normalization_15_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==49) return batch_normalization_15;


	// layer50 leaky_re_lu_15 (None, 19, 19, 512)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer50 leaky_re_lu_15 (None, 19, 19, 512)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_15=leaky_re_lu(batch_normalization_15,184832,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==50) return leaky_re_lu_15;


	// layer51 conv2d_16 (None, 19, 19, 512)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer51 conv2d_16 (None, 19, 19, 512)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_16_w=&buf_w[9864704];
	float* conv2d_16_b=&buf_w[14583296];
	float* conv2d_16 = conv_forward(leaky_re_lu_15,19,19,512,conv2d_16_w,3,3,conv2d_16_b,1,1,buf_conv2d_16,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==51) return conv2d_16;


	// layer52 batch_normalization_16 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer52 batch_normalization_16 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_16_m=&buf_w[14584320];
	float* batch_normalization_16_v=&buf_w[14585344];
	float batch_normalization_16_e=0.001f;
	float* batch_normalization_16_b=&buf_w[14586368];
	float* batch_normalization_16_g=&buf_w[14587392];
	float* batch_normalization_16=bn_forward(conv2d_16,19,19,1024,batch_normalization_16_m,batch_normalization_16_v,batch_normalization_16_e,batch_normalization_16_b,batch_normalization_16_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==52) return batch_normalization_16;


	// layer53 leaky_re_lu_16 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer53 leaky_re_lu_16 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_16=leaky_re_lu(batch_normalization_16,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==53) return leaky_re_lu_16;


	// layer54 conv2d_17 (None, 19, 19, 1024)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer54 conv2d_17 (None, 19, 19, 1024)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* conv2d_17_w=&buf_w[14588416];
	float* conv2d_17_b=&buf_w[15112704];
	float* conv2d_17 = conv_forward(leaky_re_lu_16,19,19,1024,conv2d_17_w,1,1,conv2d_17_b,1,0,buf_conv2d_17,19,19,512);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==54) return conv2d_17;


	// layer55 batch_normalization_17 (None, 19, 19, 512)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer55 batch_normalization_17 (None, 19, 19, 512)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_17_m=&buf_w[15113216];
	float* batch_normalization_17_v=&buf_w[15113728];
	float batch_normalization_17_e=0.001f;
	float* batch_normalization_17_b=&buf_w[15114240];
	float* batch_normalization_17_g=&buf_w[15114752];
	float* batch_normalization_17=bn_forward(conv2d_17,19,19,512,batch_normalization_17_m,batch_normalization_17_v,batch_normalization_17_e,batch_normalization_17_b,batch_normalization_17_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==55) return batch_normalization_17;


	// layer56 leaky_re_lu_17 (None, 19, 19, 512)->(None, 19, 19, 512)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer56 leaky_re_lu_17 (None, 19, 19, 512)->(None, 19, 19, 512)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_17=leaky_re_lu(batch_normalization_17,184832,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=512;
	if (stoplayer==56) return leaky_re_lu_17;


	// layer57 conv2d_18 (None, 19, 19, 512)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer57 conv2d_18 (None, 19, 19, 512)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_18_w=&buf_w[15115264];
	float* conv2d_18_b=&buf_w[19833856];
	float* conv2d_18 = conv_forward(leaky_re_lu_17,19,19,512,conv2d_18_w,3,3,conv2d_18_b,1,1,buf_conv2d_18,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==57) return conv2d_18;


	// layer58 batch_normalization_18 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer58 batch_normalization_18 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_18_m=&buf_w[19834880];
	float* batch_normalization_18_v=&buf_w[19835904];
	float batch_normalization_18_e=0.001f;
	float* batch_normalization_18_b=&buf_w[19836928];
	float* batch_normalization_18_g=&buf_w[19837952];
	float* batch_normalization_18=bn_forward(conv2d_18,19,19,1024,batch_normalization_18_m,batch_normalization_18_v,batch_normalization_18_e,batch_normalization_18_b,batch_normalization_18_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==58) return batch_normalization_18;


	// layer59 leaky_re_lu_18 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer59 leaky_re_lu_18 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_18=leaky_re_lu(batch_normalization_18,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==59) return leaky_re_lu_18;


	// layer60 conv2d_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer60 conv2d_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_19_w=&buf_w[19838976];
	float* conv2d_19_b=&buf_w[29276160];
	float* conv2d_19 = conv_forward(leaky_re_lu_18,19,19,1024,conv2d_19_w,3,3,conv2d_19_b,1,1,buf_conv2d_19,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==60) return conv2d_19;


	// layer61 batch_normalization_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer61 batch_normalization_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_19_m=&buf_w[29277184];
	float* batch_normalization_19_v=&buf_w[29278208];
	float batch_normalization_19_e=0.001f;
	float* batch_normalization_19_b=&buf_w[29279232];
	float* batch_normalization_19_g=&buf_w[29280256];
	float* batch_normalization_19=bn_forward(conv2d_19,19,19,1024,batch_normalization_19_m,batch_normalization_19_v,batch_normalization_19_e,batch_normalization_19_b,batch_normalization_19_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==61) return batch_normalization_19;


	// layer62 conv2d_21 (None, 38, 38, 512)->(None, 38, 38, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer62 conv2d_21 (None, 38, 38, 512)->(None, 38, 38, 64)\n");
	if (f_dbg) begin = clock();

	float* conv2d_21_w=&buf_w[29281280];
	float* conv2d_21_b=&buf_w[29314048];
	float* conv2d_21 = conv_forward(leaky_re_lu_13,38,38,512,conv2d_21_w,1,1,conv2d_21_b,1,0,buf_conv2d_21,38,38,64);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=64;
	if (stoplayer==62) return conv2d_21;


	// layer63 leaky_re_lu_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer63 leaky_re_lu_19 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_19=leaky_re_lu(batch_normalization_19,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==63) return leaky_re_lu_19;


	// layer64 batch_normalization_21 (None, 38, 38, 64)->(None, 38, 38, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer64 batch_normalization_21 (None, 38, 38, 64)->(None, 38, 38, 64)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_21_m=&buf_w[29314112];
	float* batch_normalization_21_v=&buf_w[29314176];
	float batch_normalization_21_e=0.001f;
	float* batch_normalization_21_b=&buf_w[29314240];
	float* batch_normalization_21_g=&buf_w[29314304];
	float* batch_normalization_21=bn_forward(conv2d_21,38,38,64,batch_normalization_21_m,batch_normalization_21_v,batch_normalization_21_e,batch_normalization_21_b,batch_normalization_21_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=64;
	if (stoplayer==64) return batch_normalization_21;


	// layer65 conv2d_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer65 conv2d_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_20_w=&buf_w[29314368];
	float* conv2d_20_b=&buf_w[38751552];
	float* conv2d_20 = conv_forward(leaky_re_lu_19,19,19,1024,conv2d_20_w,3,3,conv2d_20_b,1,1,buf_conv2d_20,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==65) return conv2d_20;


	// layer66 leaky_re_lu_21 (None, 38, 38, 64)->(None, 38, 38, 64)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer66 leaky_re_lu_21 (None, 38, 38, 64)->(None, 38, 38, 64)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_21=leaky_re_lu(batch_normalization_21,92416,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=38;*n_W=38;*n_C=64;
	if (stoplayer==66) return leaky_re_lu_21;


	// layer67 batch_normalization_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer67 batch_normalization_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_20_m=&buf_w[38752576];
	float* batch_normalization_20_v=&buf_w[38753600];
	float batch_normalization_20_e=0.001f;
	float* batch_normalization_20_b=&buf_w[38754624];
	float* batch_normalization_20_g=&buf_w[38755648];
	float* batch_normalization_20=bn_forward(conv2d_20,19,19,1024,batch_normalization_20_m,batch_normalization_20_v,batch_normalization_20_e,batch_normalization_20_b,batch_normalization_20_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==67) return batch_normalization_20;


	// layer68 space_to_depth_x2 (None, 38, 38, 64)->(None, 19, 19, 256)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer68 space_to_depth_x2 (None, 38, 38, 64)->(None, 19, 19, 256)\n");
	if (f_dbg) begin = clock();

	float* space_to_depth_x2 = space_to_depth(leaky_re_lu_21, 38,38,64, 2, buf_space_to_depth_x2);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=256;
	if (stoplayer==68) return space_to_depth_x2;


	// layer69 leaky_re_lu_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer69 leaky_re_lu_20 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_20=leaky_re_lu(batch_normalization_20,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==69) return leaky_re_lu_20;


	// layer70 concatenate_1 [(None, 19, 19, 256), (None, 19, 19, 1024)]->(None, 19, 19, 1280)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer70 concatenate_1 [(None, 19, 19, 256), (None, 19, 19, 1024)]->(None, 19, 19, 1280)\n");
	if (f_dbg) begin = clock();

	int nd1_70[]={ 19,19,256 },nd2_70[]={ 19,19,1024 };
	float *concatenate_1 = concatenate2(space_to_depth_x2,3,nd1_70, leaky_re_lu_20,3,nd2_70, -1,buf_concatenate_1);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1280;
	if (stoplayer==70) return concatenate_1;


	// layer71 conv2d_22 (None, 19, 19, 1280)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer71 conv2d_22 (None, 19, 19, 1280)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* conv2d_22_w=&buf_w[38756672];
	float* conv2d_22_b=&buf_w[50553152];
	float* conv2d_22 = conv_forward(concatenate_1,19,19,1280,conv2d_22_w,3,3,conv2d_22_b,1,1,buf_conv2d_22,19,19,1024);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==71) return conv2d_22;


	// layer72 batch_normalization_22 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer72 batch_normalization_22 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* batch_normalization_22_m=&buf_w[50554176];
	float* batch_normalization_22_v=&buf_w[50555200];
	float batch_normalization_22_e=0.001f;
	float* batch_normalization_22_b=&buf_w[50556224];
	float* batch_normalization_22_g=&buf_w[50557248];
	float* batch_normalization_22=bn_forward(conv2d_22,19,19,1024,batch_normalization_22_m,batch_normalization_22_v,batch_normalization_22_e,batch_normalization_22_b,batch_normalization_22_g);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==72) return batch_normalization_22;


	// layer73 leaky_re_lu_22 (None, 19, 19, 1024)->(None, 19, 19, 1024)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer73 leaky_re_lu_22 (None, 19, 19, 1024)->(None, 19, 19, 1024)\n");
	if (f_dbg) begin = clock();

	float* leaky_re_lu_22=leaky_re_lu(batch_normalization_22,369664,0.100000f);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=1024;
	if (stoplayer==73) return leaky_re_lu_22;


	// layer74 conv2d_23 (None, 19, 19, 1024)->(None, 19, 19, 425)
	// -------------------------------------------------------------------
	if (f_dbg) printf("--------- layer74 conv2d_23 (None, 19, 19, 1024)->(None, 19, 19, 425)\n");
	if (f_dbg) begin = clock();

	float* conv2d_23_w=&buf_w[50558272];
	float* conv2d_23_b=&buf_w[50993472];
	float* conv2d_23 = conv_forward(leaky_re_lu_22,19,19,1024,conv2d_23_w,1,1,conv2d_23_b,1,0,buf_conv2d_23,19,19,425);
	if (f_dbg) printf("Time elpased is %f seconds\n", (float)(clock() - begin) / CLOCKS_PER_SEC);
	*n_H=19;*n_W=19;*n_C=425;
	if (stoplayer==74) return conv2d_23;


	//--------------------

	printf("BUG unknown layer %d\n",stoplayer); return NULL;
}

