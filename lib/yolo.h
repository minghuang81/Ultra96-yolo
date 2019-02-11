#if !defined(YOLO_H)
#define YOLO_H
                
// Convert final convNet layer features to bounding box parameters.
float* yolo_head_1(float* feats,int h,int w,int c, float* anchors,int num_anchors, int num_classes);

// Filter out boxes for the final selection. Recommanded default values: 
// image_shape_h = 720, image_shape_w = 1280, max_boxes=10, score_threshold=.6, iou_threshold=.5
int yolo_eval(float* yo,int h,int w,int num_anchors, int num_classes,float** p_out_boxes,
                 int image_shape_h, int image_shape_w, 
                 int max_boxes, float score_threshold, float iou_threshold);
#endif