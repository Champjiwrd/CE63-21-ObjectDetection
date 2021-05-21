import numpy as np
import argparse
import cv2
import os
import time
import colorsys
import time
import datetime

#for share var
import builtins
from functools import partial
#multiprocessing
import multiprocessing as mp

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

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
            if labels1[classIDs[i]] == 'shirt':
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and label on the image
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format("shirt", confidences[i])
                #text = "{}: {:.4f}".format(labels1[classIDs[i]], confidences[i])
                cv2.putText(image, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return image
def draw_bounding_boxes2(image, boxes, confidences, classIDs, idxs, colors):
    if len(idxs) > 0:
        for i in idxs.flatten():
            if labels2[classIDs[i]] == 'shirt':
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # draw the bounding box and label on the image
                color = [int(c) for c in colors[classIDs[i]]]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                text = "{}: {:.4f}".format("shirt", confidences[i])
                #text = "{}: {:.4f}".format(labels1[classIDs[i]], confidences[i])
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
  for ele in l_scale:
    r0, g0, b0 = ele
    l.append(int(0.299*0.299*(r-r0)*(r-r0)+0.587*0.587*(g-g0)*(g-g0)+0.114*0.114*(b-b0)*(b-b0)))
  color = l_color[l.index(min(l))//4]
  return color

def nearest_colour_hsv(r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r,g,b)
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
        # calculate absolute distance
        for ele in l_scale:
            r0, g0, b0 = ele
            h0, s0, v0 = colorsys.rgb_to_hsv(r0,g0,b0)
            dh = (min(abs(h-h0), 360-abs(h-h0)) / 180.0)*360
            ds = abs(s-s0)
            dv = abs(v-v0) / 255.0
            distance = dh*dh+ds*ds+dv*dv
            l.append(distance)
        #print(l)
        color = l.index(min(l))//4
        return color

def classifyColor(img, boxes, idxs):
    numColor = 11
    l = [0]*numColor
    # cvt BGR to RGB
    imgn = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    #            0        1        2       3         4         5       6       7        8        9       10 
    l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
    #l_color = ['pink','red','orange','yellow','brown','green','blue','white','black']
    if len(idxs) > 0:
        for i in idxs.flatten():\
            #print('number of bbox', i)
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]      
            # (w, h, d) = img.shape

            # change into 70% w/h scale
            wn, hn = int(0.7*w), int(0.7*h)
            xn, yn = int(x+w/2-wn/2), int(y+h/2-hn/2)
            w, h = int(0.7*w), int(0.7*h)
            x, y = int(x+w/2-wn/2), int(y+h/2-hn/2)

            #print('start-end x', xn, xn+wn)
            #print('start-end y', yn, yn+hn)
            #print(img.shape)

            red, green, blue = 0, 0, 0
            for width in range(xn, xn+wn):
                for height in range(yn, yn+hn):
                      #rgb_pixel_value = img.getpixel((x+width ,y+height))
                      r, g, b = imgn[height][width][0], \
                      imgn[height][width][1], imgn[height][width][2]
                      red += r
                      green += g
                      blue += b

                      #print(width, height)
            red = int(red/(w*h))
            green = int(green/(w*h))
            blue = int(blue/(w*h))
            color = nearest_colour_hsv(red, green, blue)
            if l[color] == 0:
                l[color] = 1
            #print(l)
    return l

def return_vidsNameAndTime(isInFrame):
    frames = []
    i = 0
    start = -1
    end = -1
    while i < len(isInFrame):
        if isInFrame[i] == '1':
            start = i
            while isInFrame[i] == '1':
                i += 1
                if (i >= len(isInFrame)):
                    break
            end = i-1
            if start == end:
                frames.append(str(datetime.timedelta(seconds=start)))
            else:
                frames.append(str(datetime.timedelta(seconds=start)) +'-'+ str(datetime.timedelta(seconds=end)))
        else:
            i += 1
    if start == -1:
        return None
    else:
        return " | ".join(frames)

def vdoprocess(vidpath,args):
    cap = cv2.VideoCapture(args.videoes_path+vidpath)
    f_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = cap.get(cv2.CAP_PROP_FPS)
    vdo_l = f_len//fps

    #print('start of process',os.getpid(),vidpath,'video length',datetime.timedelta(seconds=vdo_l))

    counter = 1
    fnum = 0
    isInFrame = ''
    bagInFrame = ''
    start_t = time.process_time()

    l_color = ['pink','red','orange','yellow','brown','green','blue','white','black']
    l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            end_t = time.process_time()
            disp = return_vidsNameAndTime(isInFrame)
            if disp != None:
                print(f"{bcolors.OKBLUE}# Video Name: {vidpath} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.OKGREEN}{disp}{bcolors.ENDC}")
        
                if args.bag:
                    print(f"Bag in frame: {bcolors.OKGREEN}{return_vidsNameAndTime(bagInFrame)}{bcolors.ENDC}")
                print('Elapsed time:', datetime.timedelta(seconds=end_t - start_t))

            else :
                print(f"{bcolors.OKBLUE}# Video Name: {vidpath} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.FAIL}Not Found{bcolors.ENDC} \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")
        
            break
        if counter%fps == 0:
                
                fnum+=1
                '''
                if args.bag and args.col == None :
                    boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)
                    if checkBag(boxes, idxs,classIDs):
                            bagInFrame += '1'
                    else:
                            bagInFrame += '0'
                else:'''
                #inference


                boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold)
                
            
                #image_bb = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
                l = classifyColor(image, boxes, idxs)
                if (l[l_color.index(args.color)]==1):
                    isInFrame += '1'
                    if args.bag:
                        boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)

                        if checkBag(boxes, idxs,classIDs):
                            bagInFrame += '1'
                        else:
                            bagInFrame += '0'

                else:
                    isInFrame += '0'
                    bagInFrame += '0'

            #print(vidpath+', frame number ', fnum, l)
            #print('elapsed time = ', end - start)


        counter+=1
    cap.release()

    # show the output image
    if args.show:
        #cv2.imshow('YOLO Object Detection', image_bb)
        #cv2.waitKey(0)
        pass

    if args.save:
        #cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image_bb)
        pass

    print('Done of process:',os.getpid())
    print()

# Detect Person first then detect shirt
def personDetection(net1,net2, layer_names1, layer_names2, labels1,labels2, image, confidence, threshold ):

    boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, confidence, threshold)
    
    bag = False
    if len(idxs) > 0:
        for i in idxs.flatten():
            if labels1[classIDs[i]] == 'handbag' or labels1[classIDs[i]] == 'backpack' :
                bag = True
            if labels1[classIDs[i]] == 'person':
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]
                img_person = image[abs(y):abs(y)+h, abs(x):abs(x)+w]

                boxes2, confidences2, classIDs2, idxs2 = make_prediction(net2, layer_names2, labels2, img_person, confidence, threshold)
        
                if len(idxs2) > 0:
                    for j in idxs2.flatten():
                        classIDs[i] = 80
                        boxes[i][0], boxes[i][1] = x + boxes2[j][0], y+boxes2[j][1]
                        boxes[i][2], boxes[i][3] = boxes2[j][2], boxes2[j][3]

    return boxes, confidences, classIDs, idxs,bag

def checkBag(boxes, idxs,classIDs):
    if len(idxs) > 0:
        for i in idxs.flatten():
            if labels1[classIDs[i]] == 'handbag' or labels1[classIDs[i]] == 'backpack' :
                return True
    return False

if __name__ == '__main__':
    #ctx = mp.set_start_method('spawn')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--weights', type=str, default='yolov4_shirt.weights', help='Path to model weights')
    parser.add_argument('-cfg', '--config', type=str, default='cfg/yolov4-custom-shirt.cfg', help='Path to configuration file')
    parser.add_argument('-l', '--labels', type=str, default='cfg/justshirt.names', help='Path to label file')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum confidence for a box to be detected.')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Threshold for Non-Max Suppression')
    parser.add_argument('-u', '--use_gpu', default=False, action='store_true', help='Use GPU (OpenCV must be compiled for GPU). For more info checkout: https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/')
    parser.add_argument('-s', '--save', default=False, action='store_true', help='Whether or not the output should be saved')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-col', '--color', type=str, help="color > 'pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black' ")
    parser.add_argument('-bag', '--bag', default=False, action="store_true", help='Filter Bag')
    parser.add_argument('-bi', '--bilayer', default=False, action="store_true", help='Detect in 2 layer model')

    #parser.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument('-i', '--image_path', type=str, default='', help='Path to the image file.')
    input_group.add_argument('-v', '--video_path', type=str, default='', help='Path to the video file.')
    input_group.add_argument('-vids', '--videoes_path', type=str, default='', help='Path to the videoes directory')

    args = parser.parse_args()
    # Get the labels
    labels1 = open("cfg/coco.names").read().strip().split('\n')
    labels2 = open(args.labels).read().strip().split('\n')

    # Create a list of colors for the labels
    colors = np.random.randint(0, 255, size=(len(labels1), 3), dtype='uint8')

    print('Load Network...')
    start_time = time.time()

    # Load weights using OpenCV
    config_person = "cfg/yolov4.cfg"
    weights_person = "cfg/yolov4.weights"
    net1 = cv2.dnn.readNetFromDarknet(config_person, weights_person)
    net2 = cv2.dnn.readNetFromDarknet(args.config, args.weights)

    if args.use_gpu:
        print('Using GPU')
        net1.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net1.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        net2.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net2.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)

    if args.save:
        print('Creating output directory if it doesn\'t already exist')
        os.makedirs('output', exist_ok=True)

    
    
    # Get the ouput layer names
    layer_names1 = net1.getLayerNames()
    layer_names1 = [layer_names1[i[0] - 1] for i in net1.getUnconnectedOutLayers()]

    layer_names2 = net2.getLayerNames()
    layer_names2 = [layer_names2[i[0] - 1] for i in net2.getUnconnectedOutLayers()]
    print(f"Finish load network, elapsed time: {(time.time()-start_time)} ")
    
    print('-------------------------------------------------------------------------')

    '''
    image = cv2.imread(args.image_path)
    boxes, confidences, classIDs, idxs = make_prediction(net, layer_names, labels, image, args.confidence, args.threshold)
    l = classifyColor(image, boxes, idxs)
    image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
    '''

    print('Detect \"{}\" shirt and bag'.format(args.color)) if args.bag else print('Detect \"{}\" shirt'.format(args.color))

    if args.video_path != '':
        cap = cv2.VideoCapture(args.video_path)
        f_len = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        vdo_l = f_len//fps

        #print('start of process',os.getpid(),vidpath,'video length',datetime.timedelta(seconds=vdo_l))

        counter = 1
        fnum = 0
        isInFrame = ''
        bagInFrame = ''
        start_t = time.process_time()

        l_color = ['pink','red','orange','yellow','brown','green','blue','white','black']
        l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                end_t = time.process_time()
                disp = return_vidsNameAndTime(isInFrame)
                if disp != None:
                    print(f"{bcolors.OKBLUE}# Video Name: {args.video_path} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.OKGREEN}{disp}{bcolors.ENDC}")
            
                    if args.bag:
                        print(f"Bag in frame: {bcolors.OKGREEN}{return_vidsNameAndTime(bagInFrame)}{bcolors.ENDC}")
                    print('Elapsed time:', datetime.timedelta(seconds=end_t - start_t))

                else :
                    print(f"{bcolors.OKBLUE}# Video Name: {args.video_path} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.FAIL}Not Found{bcolors.ENDC} \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")
            
                break
            if counter%fps == 0:
                    
                    fnum+=1
                    '''
                    if args.bag and args.col == None :
                        boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)
                        if checkBag(boxes, idxs,classIDs):
                                bagInFrame += '1'
                        else:
                                bagInFrame += '0'
                    else:'''
                    #inference


                    boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold)
                    
                
                    #image_bb = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors)
                    l = classifyColor(image, boxes, idxs)
                    if (l[l_color.index(args.color)]==1):
                        isInFrame += '1'
                        if args.bag:
                            boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)

                            if checkBag(boxes, idxs,classIDs):
                                bagInFrame += '1'
                            else:
                                bagInFrame += '0'

                    else:
                        isInFrame += '0'
                        bagInFrame += '0'

                #print(vidpath+', frame number ', fnum, l)
                #print('elapsed time = ', end - start)


            counter+=1
        cap.release()

        # show the output image
        if args.show:
            #cv2.imshow('YOLO Object Detection', image_bb)
            #cv2.waitKey(0)
            pass

        if args.save:
            #cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image_bb)
            pass

        print('Done of process:',os.getpid())
        print()

    if args.save:
        width = int(cap.get(3))
        height = int(cap.get(4))
        fps = cap.get(cv2.CAP_PROP_FPS)
        name = args.video_path.split("/")[-1] if args.video_path else 'camera.avi'
        out = cv2.VideoWriter(f'output/{name}', cv2.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))

    if args.videoes_path != '': # run for vids in dir

        #get all the file
        vdo_list = []
        

        for vidpath in [vid for vid in os.listdir(args.videoes_path)]:
            if vidpath.endswith(('.mp4',".MOV",'.avi')):
                vdo_list.append(vidpath)

        l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']

        print(vdo_list)
        print('Number of CPU :',mp.cpu_count())
        pool = mp.Pool(processes = mp.cpu_count())
        pool.map_async(partial(vdoprocess, args=args),vdo_list)

        pool.close()
        pool.join()

    if args.image_path != '':
        image = cv2.imread(args.image_path)

        if args.bag and args.color == None :
            boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)
            
        # Load weights using OpenCV
        start_time = time.time()
        #boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold) # Detect from yolov4.weight 
        if args.bilayer:
            boxes, confidences, classIDs, idxs, bag = personDetection(net1,net2,layer_names1,layer_names2,labels1,labels2,image,args.confidence,args.threshold) # Detect from yolo then detect shirt
            image = draw_bounding_boxes(image, boxes, confidences, classIDs, idxs, colors) # make_prediction, personDetection
        
        else:

            boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold) # Detect from shirt weight
            image = draw_bounding_boxes2(image, boxes, confidences, classIDs, idxs, colors)

        l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']

        l = classifyColor(image, boxes, idxs)
        if (l[l_color.index(args.color)]==1):
            print(f"{bcolors.OKGREEN}Found {args.color} shirt {bcolors.ENDC}")
        else:
            print(f"{bcolors.FAIL}Not Found {args.color} shirt {bcolors.ENDC}")

        #print('Bag: {}'.format(bag))
        #l = intregratedColor(l)
        print(f"Process time: {(time.time()-start_time)}")
        # show the output image
        if args.show:
            
            cv2.imshow('YOLO Object Detection', image)
            
            cv2.waitKey(0)
        
        if args.save:
            cv2.imwrite(f'output/{args.image_path.split("/")[-1]}', image)


    #cv2.destroyAllWindows()

