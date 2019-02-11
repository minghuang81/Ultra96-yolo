import numpy as np
import cv2


def read_classes(classes_path):
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def read_anchors(anchors_path):
    with open(anchors_path) as f:
        anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        anchors = np.array(anchors).reshape(-1, 2)
    return anchors



# Argument:
#   img_path (h, w) = (720, 1280)
#   model_image_size -- (608, 608), (h, w) of the resized image
# Return:
#   image_data.shape =  (1, 608, 608, 3) -- numpy array of image pixels 
def preprocess_image_cv(img_path, model_image_size=(608, 608)):
    #print('yolo.preprocess_image img_path=',img_path)
    image_data = cv2.imread(img_path).astype('float32') / 255 # (h,w,ch)
    image_data = cv2.cvtColor(image_data,cv2.COLOR_RGB2BGR)
    #print('image_data type=',type(image_data),' shape=',image_data.shape,'dtype=',image_data.dtype)
    resized_image = cv2.resize(image_data,model_image_size[::-1], interpolation=cv2.INTER_CUBIC)
    #print('resized_image type=',type(resized_image),', shape=',resized_image.shape,'dtype=',resized_image.dtype)
    resized_image = np.expand_dims(resized_image, 0)  # Add batch dimension.
    return image_data,resized_image


def save_image(path, image):    
    # Un-normalize the image
    image2 = 256*cv2.cvtColor(image,cv2.COLOR_RGB2BGR)  
    # Clip and Save the image
    image2 = np.clip(image2, 0, 255).astype('uint8')
    cv2.imwrite(path, image2)    

def draw_boxes(image, out_scores, out_boxes, out_classes, class_names):
    """
    image -- numpy array (h,w) from cv2
    out_scores -- (None,)
    out_boxes -- (None,h,w)
    out_classes -- (None,) of 0..79
    class_names -- list of 'name1', 'name2', ..
    """
    thickness = (image.shape[0] + image.shape[1]) // 300
    #print('draw_boxes imge shape ',image.shape)

    for i, c in reversed(list(enumerate(out_classes))):
        predicted_class = class_names[c]
        box = out_boxes[i]
        score = out_scores[i]


        top, left, bottom, right = box
        top = max(0, np.floor(top + 0.5).astype('int32'))
        left = max(0, np.floor(left + 0.5).astype('int32'))
        bottom = min(image.shape[0], np.floor(bottom + 0.5).astype('int32'))
        right = min(image.shape[1], np.floor(right + 0.5).astype('int32'))
        
        label = '{} {:.2f}'.format(predicted_class, score)
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.5, 1)[0] 
        label_size = np.array(label_size) # [w,h]
        #print(label, (left, top), (right, bottom))
        text_origin = np.array([left, top])

        # coordinates convention: (x,y) corresponding to (height, width)
        cv2.rectangle(image, (left,top), (right,bottom), (255, 0, 0), thickness=thickness);
        cv2.rectangle(image, tuple(text_origin), tuple(text_origin+label_size), (0, 0, 0), cv2.FILLED);
        cv2.putText(image, label, tuple(text_origin+(0,label_size[1])), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255, 255, 255), 1)

