#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "yolo.h"
#include "conv_net.h"


// Convert model output to usable bounding box tensors that 
// needs to pass through filter over threshold and non-max suppression. 
// e.g yolo_model.layers[74] of shape (19, 19, 425) -> (19, 19, 5, 85)

float* yolo_head_1(float* feats,int h,int w,int c, float* anchors,int num_anchors, int num_classes)
{
//    Convert final layer features to bounding box parameters.
//
//    Parameters
//    ----------
//    feats : tensor (h=19, w=19, c=425) Final convolutional layer features. 
//    anchors : array (5,2) - Anchor box widths and heights.
//    num_anchors : e.g. 5
//    num_classes : int e.g. Number of target classes = 80
//
//    Returns
//    -------
//    box_xy : tensor (19, 19, 5, 2)
//        x, y box predictions adjusted by spatial location in conv layer. range (0..1)
//    box_wh : tensor (19, 19, 5, 2)
//        w, h box predictions adjusted by anchors and conv spatial resolution. scale_boxes
//    box_conf : tensor (19, 19, 5, 1)
//        Probability estimate for whether each box contains any object.
//    box_class_pred : tensor (19, 19, 5, 80)
//        Probability distribution estimate for each box over class labels.
    
    int a_stride = num_classes+5;
    int w_stride = num_anchors*a_stride;
    int h_stride = w*w_stride;
    float* arr_h=feats;
    for (int i=0;i<h;i++, arr_h+=h_stride) {
      float*  arr_w = arr_h;
      for (int j=0;j<w;j++,arr_w+=w_stride) {
        float* arr_a = arr_w;        
        for (int a=0;a<num_anchors;a++,arr_a+=a_stride) {
            //if (f_dbg) printf("yolo_head_1: i %d,j %d,a %d\n",i,j,a);
            float* box_confidence = &arr_a[4];
            float* box_xy = &arr_a[0];
            float* box_wh = &arr_a[2];
            float* box_class_probs = &arr_a[5];
            // box_confidence = sigmoid(feats[..., 4:5])
            sigmoid(box_confidence,1);
            // box_xy = sigmoid(feats[..., :2])
            sigmoid(box_xy,2);
            // box_wh = np.exp(feats[..., 2:4]) # feats[..., 2:4] : log(w,h) in unit of anchor-box size
            box_wh[0] = (float)exp(box_wh[0]);
            box_wh[1] = (float)exp(box_wh[1]);
            // box_class_probs = softmax(feats[..., 5:])
            softmax(box_class_probs,1,num_classes);
            
            // Adjust preditions to each spatial grid point and anchor size.
            // box_xy = (box_xy + (w,h) grid index 
            box_xy[0] = (box_xy[0] + j) / w; //horizontal width direction
            box_xy[1] = (box_xy[1] + i) / h; //vertical height direction
            // box_wh is now in linear unit of anchor-box size
            // anchors_tensor has the unit of grid cell size
            box_wh[0] = box_wh[0]*anchors[a+a+0]/w;
            box_wh[1] = box_wh[1]*anchors[a+a+1]/h;
        }
      }
    }

    return feats;
}

// Convert YOLO box predictions to bounding box corners (h1,w1, h2, w2)
// (box_xy[0], box_xy[0], box_wh[0], box_wh[0]) => (ymin,xmin) (ymax,xmax)
// return: replace input by output at the same memory location
// box_xy, box_wh are consecutive in mem
float* yolo_boxes_to_corners(float* box_xy, float* box_wh)
{    
    float box_mins[2] = {(box_xy[0]-box_wh[0]/2.f),(box_xy[1]-box_wh[1]/2.f)};
    float box_maxs[2] = {(box_xy[0]+box_wh[0]/2.f),(box_xy[1]+box_wh[1]/2.f)};
    box_xy[0] = box_mins[1]; // ymin
    box_xy[1] = box_mins[0]; // xmin
    box_wh[0] = box_maxs[1]; // ymax
    box_wh[1] = box_maxs[0]; // xax
    return box_xy;
}

// For each box, find:
//     - the index of the class with the maximum class prob 
//     - the corresponding box score = box_confidence x box_class_probs
// compare the box score with the threshold. 
// - Override box_confidence <= the calculated box score
// - If the box is retained, returns the index of the max class
// - If the box is rejected, returns -1

 
int yolo_filter_boxe(float* box_confidence, float* box_class_probs, int num_classes, float threshold)
{
    // find the index of the max class
    int max_class_index = -1;
    float max_class_prob = -1.f;
    for (int i=0;i<num_classes;i++) {
        if (box_class_probs[i] > max_class_prob) {
            max_class_prob = box_class_probs[i];
            max_class_index = i;
        }
    }    
    // Compute box scores
    *box_confidence = (*box_confidence) * max_class_prob;
    // return of discard the box
    if (*box_confidence > threshold) {
        return max_class_index;
    } else {
        return -1;
    }
}

//    Scales the predicted boxes in order to be drawable on the image
//    Arguments:
//        boxe -- of shape (4) = (ymin,xmin,ymax,xmax) in the rage of (0..1)
//                with '1' designates the full size of the image (either h or w)
//        image_shape -- (height,width) in pixels
//    return:
//        boxe -- square boxes stretched to match image_shape. Of shape (4)
void scale_box(float* box, int image_shape_h, int image_shape_w)
{
    box[0] *= (float)image_shape_h;
    box[2] *= (float)image_shape_h;
    box[1] *= (float)image_shape_w;
    box[3] *= (float)image_shape_w;
}

// Non-max suppression uses the very important function called **"Intersection over Union"**, or IoU.
// - we define a box using its two corners (upper left and lower right): (y1, x1, y2, x2).
// - To calculate the area of a rectangle, multiply its height (y2 - y1) by its width (x2 - x1)
// - the coordinates (yi1, xi1, yi2, xi2) of the intersection of two boxes:
//     - xi1 = maximum of the x1 coordinates of the two boxes
//     - yi1 = maximum of the y1 coordinates of the two boxes
//     - xi2 = minimum of the x2 coordinates of the two boxes
//     - yi2 = minimum of the y2 coordinates of the two boxes    
// Convention: (0,0) is the top-left corner of an image, (0,w) is the upper-right corner, and (h,w) the lower-right corner. 

float iou(float* box1, float* box2)
{
//    Arguments:
//    box1[4] -- first box,  list object with coordinates (y1, x1, y2, x2)
//    box2[4] -- second box, list object with coordinates (y1, x1, y2, x2)
//    Return:
//    area of the intersection (box1&box2) / area of the union (box1+box2)

    // Calculate the (y1, x1, y2, x2) coordinates of the intersection of box1 and box2. Calculate its Area.
    float yi1 = fmaxf(box1[0],box2[0]);
    float xi1 = fmaxf(box1[1],box2[1]);
    float yi2 = fminf(box1[2],box2[2]);
    float xi2 = fminf(box1[3],box2[3]);
    float inter_area;
    if ((xi2-xi1)<=0 || (yi2-yi1)<=0)
        inter_area = 0.;
    else
        inter_area = (xi2-xi1)*(yi2-yi1);

    // Calculate the Union area by using Formula: Union(A,B) = A + B - Inter(A,B)
    float box1_area = (box1[2]-box1[0])*(box1[3]-box1[1]);
    float box2_area = (box2[2]-box2[0])*(box2[3]-box2[1]);
    float union_area = box1_area+box2_area - inter_area;

    // compute the IoU
    float iou = inter_area/union_area;    
    return iou;
}

// Box indexes are sorted in the decreasing box scores.
// NMS_comp_boxes are the boxes that the sorted indexes are refering, they stay put; only the indexes are sorted.
float* NMS_comp_boxes; //  
int NMS_comp (const void * elem1, const void * elem2) 
{
    int f = *((int*)elem1);
    int s = *((int*)elem2);
    if (NMS_comp_boxes[f*6] > NMS_comp_boxes[s*6]) return  -1;
    if (NMS_comp_boxes[f*6] < NMS_comp_boxes[s*6]) return   1;
    return 0;
}

// Non-max suppression: 
// 1. Select the box that has the highest score.
// 2. Compute its overlap with all other boxes, and remove boxes that overlap it more than `iou_threshold`.
// 3. Go back to step 1 and iterate until there's no more boxes with a lower score than the current selected box.
// 
// This will remove all boxes that have a large overlap with the selected boxes. Only the "best" boxes remain.
// Recommanded values: max_boxes=10,float iou_threshold=0.5
int yolo_non_max_suppression(float inboxes[][6],int inboxes_cnt,float outboxes[][6],int max_boxes,float iou_threshold)
{
//    Applies Non-max suppression (NMS) to set of boxes
//    
//    Arguments:
//    inboxes[][0] -- box score
//    inboxes[][1] -- box coordinate y1
//    inboxes[][2] -- box coordinate x1
//    inboxes[][3] -- box coordinate y2
//    inboxes[][4] -- box coordinate x2
//    inboxes[][5] -- index of class that has the highest score
//    inboxes_cnt  -- valid entries in inboxes[]
//    max_boxes --  integer, maximum number of predicted boxes you'd like
//    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
//    
//    Returns:
//    outboxes -- same structure as inboxes
//    outboxes_cnt -- number of valid entries in outboxes
    // sort the boxes in decreasing score
    int idxs[MAX_NB_OF_CANDIDATE_BOXES];
    if (inboxes_cnt>MAX_NB_OF_CANDIDATE_BOXES) {
        printf("yolo_non_max_suppression not expecting inboxes_cnt %d >MAX_NB_OF_CANDIDATE_BOXES\n",inboxes_cnt);
    }
    for (int i=0;i<inboxes_cnt;i++) idxs[i] = i;
    NMS_comp_boxes = (float*)inboxes;
    qsort (idxs, inboxes_cnt, sizeof(*idxs), NMS_comp);
    int outboxes_cnt=0;
    for (int n1=0;n1<inboxes_cnt;n1++) {
        int i = idxs[n1];
        if (i == -1) continue;
        for (int k=0;k<6;k++) outboxes[outboxes_cnt][k] = inboxes[i][k];
        outboxes_cnt++;
        if (outboxes_cnt>=max_boxes) break;
        for (int n2=n1+1;n2<inboxes_cnt;n2++) {   
            int j = idxs[n2];
            if (j == -1) continue;
            if (iou(&inboxes[i][1], &inboxes[j][1])>iou_threshold) {// #discard j
                idxs[n2] = -1;
            }
        }
    }
    
    return outboxes_cnt;
}    

// recommanded default values: 
//                 image_shape_h = 720, image_shape_w = 1280, 
//                 max_boxes=10, 
//                 score_threshold=.6, iou_threshold=.5)
            
int yolo_eval(float* yo,int h,int w,int num_anchors, int num_classes,float** p_out_boxes,
                 int image_shape_h, int image_shape_w, 
                 int max_boxes, 
                 float score_threshold, float iou_threshold)
{
//    
//    Converts the output of YOLO encoding (a lot of boxes) to your predicted (a few) boxes 
//    along with their scores, box coordinates and classes.
//    
//    Arguments:
//    yo -- yolo_outputs, output of the encoding model (for image_shape of (608, 608, 3)), 
//          of shape (h,w,num_anchors,num_classes+5).
//          contains 4 tensors:
//                    box_confidence: yo[0:19, 0:19, 0:5, 4:5]
//                    box_xy: yo[0:19, 0:19, 0:5, 0:2] in the order of (w/x,h/y)
//                    box_wh: yo[0:19, 0:19, 0:5, 2:4] in the order of (w,h)
//                    box_class_probs: yo[0:19, 0:19, 0:5, 5:85)
//    image_shape -- tensor of shape (2,) containing the input shape, 
//                   we use (608., 608.) (has to be float32 dtype)
//    max_boxes -- integer, maximum number of predicted boxes you'd like
//    score_threshold -- real value, if [ highest class probability score < threshold], 
//                       then get rid of the corresponding box
//    iou_threshold -- real value, "intersection over union" threshold used for NMS filtering
//    
//    Returns:
//    (*p_out_boxes) float array of shape (out_boxes_cnt, 6):
//    (*p_out_boxes)[None, 0:1] -- scores, predicted score for each box
//    (*p_out_boxes)[None, 1:5] -- boxes, predicted box coordinates (y1,x1,y2,x2)
//    (*p_out_boxes)[None, 5:6] -- classes, predicted class for each box
//    
    static float cand_boxes[MAX_NB_OF_CANDIDATE_BOXES][6]; // (box score,box coordinates y1/x1/y2/x2,best class in box)
    if (h*w*num_anchors>MAX_NB_OF_CANDIDATE_BOXES) {
        printf("yolo_eval not expecting h*w*num_anchors %d >MAX_NB_OF_CANDIDATE_BOXES\n",h*w*num_anchors);        
    }
    if (max_boxes>MAX_NB_OF_OUTPUT_BOXES) max_boxes=MAX_NB_OF_OUTPUT_BOXES;
    
    int cand_boxes_cnt = 0; // valid entries in the array buf_out_boxes[]
    int a_stride = num_classes+5;
    int w_stride = num_anchors*a_stride;
    int h_stride = w*w_stride;
    float* arr_h=yo;
    for (int i=0;i<h;i++, arr_h+=h_stride) {
      float*  arr_w = arr_h;
      for (int j=0;j<w;j++,arr_w+=w_stride) {
        float* arr_a = arr_w;        
        for (int a=0;a<num_anchors;a++,arr_a+=a_stride) {
            float* box_confidence = (&arr_a[4]);
            float* box_xy         = (&arr_a[0]);
            float* box_wh         = (&arr_a[2]);
            float* box_class_probs= (&arr_a[5]);
            // based on threshold, tag the box as selected (box_confidence=class index) or discarded (box_confidence = -1.)
            int bestclass = yolo_filter_boxe(box_confidence, box_class_probs, num_classes, score_threshold);
            if (bestclass == -1) continue; // this box is discarded for too low score
            // Convert boxe to be ready for filtering functions 
            float* box = yolo_boxes_to_corners(box_xy, box_wh);
            // Scale boxes back to original image shape. Input range(0..1) => output (h,w) pixels
            scale_box(box, image_shape_h, image_shape_w);
            // Store the box
            cand_boxes[cand_boxes_cnt][0] = *box_confidence;  // score
            cand_boxes[cand_boxes_cnt][1] = box[0];           // y1
            cand_boxes[cand_boxes_cnt][2] = box[1];           // x1
            cand_boxes[cand_boxes_cnt][3] = box[2];           // y2
            cand_boxes[cand_boxes_cnt][4] = box[3];           // x2
            cand_boxes[cand_boxes_cnt][5] = (float)bestclass;        // index of class that has the highest score
            cand_boxes_cnt++;
        }
      }
    }
    if (f_dbg) printf("calling yolo_non_max_suppression with cand_boxes_cnt = %d, max_boxes=%d\n",cand_boxes_cnt,max_boxes);
    // perform Non-max suppression 
    // with a threshold of iou_threshold (˜1 line)
    static float out_boxes[MAX_NB_OF_OUTPUT_BOXES][6]; // (box score,box coordinates y1/x1/y2/x2,best class in box)
    int out_boxes_cnt=yolo_non_max_suppression(cand_boxes,cand_boxes_cnt,out_boxes,max_boxes,iou_threshold);
    *p_out_boxes = (float*)out_boxes;
    return out_boxes_cnt;
}
