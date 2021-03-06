import numpy as np
import argparse
import cv2
import os
import time
import webcolors
#from func import *

def extract_boxes_confidences_classids(outputs, confidence, width, height):
    boxes = []
    confidences = []
    classIDs = []

    for output in outputs:
        for detection in output:            
            # Extract the scores, classid, and the confidence of the prediction
            scores = detection[5:]
            classID = np.argmax(scores)
            conf = scores[classID]
            
            # Consider only the predictions that are above the confidence threshold
            if conf > confidence:
                # Scale the bounding box back to the size of the image
                box = detection[0:4] * np.array([width, height, width, height])
                centerX, centerY, w, h = box.astype('int')

                # Use the center coordinates, width and height to get the coordinates of the top left corner
                x = int(centerX - (w / 2))
                y = int(centerY - (h / 2))

                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(conf))
                classIDs.append(classID)

    return boxes, confidences, classIDs

def draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]

            # draw the bounding box and label on the image
            color = [int(c) for c in colors[classIDs[i]]]
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
            text = "{}: {:.4f}".format(labels[classIDs[i]], confidences[i])
            cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image

def make_prediction(net, layer_names, labels, image, confidence, threshold):
    height, width = image.shape[:2]
    
    # Create a blob and pass it through the model
    blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(layer_names)

    # Extract bounding boxes, confidences and classIDs
    boxes, confidences, classIDs = extract_boxes_confidences_classids(outputs, confidence, width, height)

    # Apply Non-Max Suppression
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence, threshold)

    return boxes, confidences, classIDs, idxs

def classifyColor(img, boxes, idxs):
    numColor = 11
    l = [0]*numColor
    #            0        1        2       3         4         5       6       7        8        9       10 
    l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
    if len(idxs) > 0:
        for i in idxs.flatten():
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]      
            # (w, h, d) = img.shape

            # change into 70% w/h scale
            wn, hn = int(0.7*w), int(0.7*h)
            x, y = int(x+w/2-wn/2), int(y+h/2-hn/2)
            color =(255, 0, 0) 
            cv2.rectangle(img, (x, y), (x + wn, y + hn), color, 2)

            red, green, blue = 0, 0, 0
            for width in range(x, x+wn):
                for height in range(y, y+hn):
                      #rgb_pixel_value = img.getpixel((x+width ,y+height))
                      r, g, b = img[width][height][0], \
                      img[width][height][1], img[width][height][2]
                      red += r
                      green += g
                      blue += b
            #print(red,green,blue)
            area = wn*hn
            red = int(red/(area))
            green = int(green/(area))
            blue = int(blue/(area))
                        
            #print(red, green, blue)

            color = nearest_colour(red, green, blue)
            if l[l_color.index(color)] == 0:
                l[l_color.index(color)] = 1

            #cv2.putText(image, color, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
    return l
   
def nearest_colour(r, g, b):
  l = []
  #            0        1        2       3         4         5       6       7        8        9       10 
  l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
  l_scale = [(100,0,100),(100,75,80),(100,41,71),(100,8,58)\
             ,(50,0,50),(87,63,87),(73,33,83),(29,0,51)\
             ,(100,0,0),(100,63,45),(80,36,36),(55,0,0),\
             (100,27,0),(100,50,31),(100,27,0),(100,55,0),\
             (100,100,0),(100,100,88),(94,90,55),(100,89,55),\
             (0,50,0),(49,99,0),(13,55,13),(42,56,14),\
             (0,100,100),(88,100,100),(25,88,82),(0,55,55),\
             (0,0,100),(69,88,90),(0,75,100),(10,10,44),\
             (50,0,0),(87,72,53),(82,41,12),(55,27,7),\
             (100,100,100),(94,100,94),(96,96,86),(98,94,90),\
             (0,0,0),(41,41,41),(75,75,75),(50,50,50)]
  r, g, b = int(r*100/255), int(g*100/255), int(b*100/255)
  # calculate absolute distance
  for e in range(len(l_scale)):
    r0, g0, b0 = l_scale[e]
    l.append(int(0.299*0.299*(r-r0)*(r-r0)+0.587*0.587*(g-g0)*(g-g0)+0.114*0.114*(b-b0)*(b-b0)))
  print(l)
  color = l_color[l.index(min(l))//4]
  return color

def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.css3_hex_to_names.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]

def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name

def start_inference(selected_path,showvdo):
    #origin_path = 'G:/.shortcut-targets-by-id/1hQoqbcLdIKP1y1vE2waDf28rLEbA3yqR/CEProject63-21ObjectDetection/colab_work/darknet/forshirt/'
    inp = ['justshirt.names',
    'Copyofcustom-yolov4-detector.cfg',
    'custom-yolov4-detector_best.weights',
    ]
    video_path = selected_path[0]
    confidence = 0.5
    threshold = 0.3

    # Get the labels]
    global labels
    labels = open(inp[0]).read().strip().split('\n')

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet(inp[1],inp[2])

    # Get the ouput layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    if video_path != '':
        cap = cv2.VideoCapture(video_path)
    else:
        cap = cv2.VideoCapture(0)

    fps = cap.get(cv2.CAP_PROP_FPS)
    counter = 1

    countlist = []
    box_result = []
    imgs = []
    while cap.isOpened():
        ret, image = cap.read()
        if counter%fps==0:

            if not ret:
                print('Video file finished.')
                break
            #inprogress
            boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, confidence, threshold)
            
            if boxes:
                print(counter/60)
                box_result.append(boxes)
                countlist.append(int(counter/60))
                #imgs.append(image)
            if showvdo.get():
                image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
                cv2.imshow('YOLO Object Detection', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            counter += 1
        else:
            counter += 1
    print('end already')
    cap.release()
    cv2.destroyAllWindows()
    return box_result,imgs,countlist

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='model/yolov3.weights', help='Path to model weights')
    parser.add_argument('-cfg', '--config', type=str, default='model/yolov3.cfg', help='Path to configuration file')
    parser.add_argument('-l', '--labels', type=str, default='model/coco.names', help='Path to label file')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum confidence for a box to be detected.')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Threshold for Non-Max Suppression')
    parser.add_argument('-u', '--use_gpu', default=False, action='store_true', help='Use GPU (OpenCV must be compiled for GPU). For more info checkout: https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/')
    parser.add_argument('-s', '--save', default=False, action='store_true', help='Whether or not the output should be saved')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')
    input_group.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')

    args = parser.parse_args()

    # Get the labels
    labels = open(args.labels).read().strip().split('\n')

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

    # Load weights using OpenCV
    net = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    if args.use_gpu:
        print('Using GPU')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.save:
        print('Creating output directory if it doesn\'t already exist')
        os.makedirs('output', exist_ok=True)

    # Get the ouput layer names
    layer_names = net.getLayerNames()
    layer_names = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]


    if args.image_path != '':
        image = cv2.imread(args.image_path)

        boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)

        classifyColor(image,boxes,idxs)

        #image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)

        # show the output image
        if args.show:
            cv2.imshow('YOLO Object Detection', image)
            cv2.waitKey(0)
        
        if args.save:
            cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image)
    else:
        if args.video_path != '':
            cap = cv2.VideoCapture(args.video_path)
        else:
            cap = cv2.VideoCapture(0)

        if args.save:
            width = int(cap.get(3))
            height = int(cap.get(4))
            fps = cap.get(cv2.CAP_PROP_FPS)
            name = args.video_path.split("/")[-1] if args.video_path else 'camera.avi'
            out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

        fps = cap.get(cv2.CAP_PROP_FPS)
        counter = 1
        while cap.isOpened():
            ret, image = cap.read()
            if counter%fps==0:
                counter = 1

                if not ret:
                    print('Video file finished.')
                    break

                boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)

                #image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)

                if args.show:
                    cv2.imshow('YOLO Object Detection', image)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                if args.save:
                    out.write(image)
            else:
                counter += 1
        cap.release()
        if args.save:
            out.release()
    cv2.destroyAllWindows()


