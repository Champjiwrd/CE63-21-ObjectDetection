import numpy as np
import argparse
import cv2
import os
import colorsys
import time
import datetime
from pathlib import Path

#for share var
import builtins
from functools import partial
#multiprocessing
import multiprocessing as mp
from multiprocessing import Manager
from line_notify import LineNotify

ACCESS_TOKEN = "zrs6CR00Pe8CO57XK16pl6ouJs1eqfGMqGCu4bN7byC"

notify = LineNotify(ACCESS_TOKEN)

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'

#l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
l_color = ['pink','red','orange','yellow','green','blue','brown','white','black']

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

def draw_bounding_boxes3(image, boxes, confidences, classIDs, idxs, colors):
    imgn = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
    if len(idxs) > 0:
        for i in idxs.flatten():
            if labels2[classIDs[i]] == 'shirt':
                # extract bounding box coordinates
                x, y = boxes[i][0], boxes[i][1]
                w, h = boxes[i][2], boxes[i][3]

                # change into 70% w/h scale
                wn, hn = int(0.7*w), int(0.7*h)
                xn, yn = int(x+w/2-wn/2), int(y+h/2-hn/2)
                w, h = int(0.7*w), int(0.7*h)
                x, y = int(x+w/2-wn/2), int(y+h/2-hn/2)

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

                if l_color[color] == args.color:

                    # draw the bounding box and label on the image
                    color = [int(c) for c in colors[classIDs[i]]]
                    x, y = boxes[i][0], boxes[i][1]
                    w, h = boxes[i][2], boxes[i][3]
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    text = "{} {}: {:.4f}".format(args.color,"shirt", confidences[i])
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

def nearest_colour_hsv(r, g, b):
        h, s, v = colorsys.rgb_to_hsv(r,g,b)

        pink = [(199,21,133),(255,20,147),(219,112,147),(255,105,180)]
        red = [(130,55,55),(123,51,57),(139,0,0),(255,0,0),(178,34,34),(220,20,60),(205,92,92),(170,57,65)]
        orange = [(217,103,62),(121,80,54),(122,81,55),(194,105,84),(255,69,0),(255,99,71),(255,140,0),(255,127,80),(255,165,0)]
        yellow = [(205,227,62),(255,215,0),(255,255,0),(240,230,140),(255,250,205),(189,183,107)]
        brown = [(136,90,72),(156,101,73),(128,0,0),(165,42,42),(139,69,19),(160,82,45),(210,105,30),(184,134,11),\
            (205,133,63),(188,143,143),(218,165,32),(244,164,96)] # not all
        green = [(93,131,101),(101,141,106),(173,255,47),(127,255,0),(124,252,0),(0,255,0),(50,205,50),(152,251,152),\
            (0,250,154),(0,255,127),(0,128,0),(0,100,0)]
        blue = [(52,91,94),(86,134,141),(60,70,113),(0,0,128),(0,0,139),(0,0,205),(0,0,255),\
            (65,105,225),(100,149,237),(30,144,255),(0,191,255),(135,206,250),(173,216,230),\
                (176,196,222),(70,130,180),(0,206,209),(0,255,255),(175,238,238),(64,224,208)]
        white = [(255,255,255),(255,250,250),(240,255,240),(245,255,250),(240,248,255),\
            (245,245,245),(255,240,245),(204,211,230)]
        black = [(0,0,0),(47,79,79),(105,105,105),(112,128,144),(128,128,128),(169,169,169),\
            (192,192,192)]
        listColor = [pink,red,orange,yellow,green,blue,brown,white,black]
        listDistance = []
        for color in listColor:
            listinColor = []
            for ele in color:
                r0, g0, b0 = ele
                h0, s0, v0 = colorsys.rgb_to_hsv(r0,g0,b0)
                dh = (min(abs(h-h0), 360-abs(h-h0)) / 180.0)*360
                ds = abs(s-s0)
                dv = abs(v-v0) / 255.0
                distance = dh*dh+ds*ds+dv*dv
                listinColor.append(distance)
            listDistance.append(min(listinColor))
            #listDistance.append(tmp)
        color = listDistance.index(min(listDistance))
        #print(r,g,b,l_color[color])
        
        return color

def classifyColor(img, boxes, idxs):
    # cvt BGR to RGB
    imgn = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
    #            0        1        2       3         4         5       6       7        8        9       10 
    #l_color = ['pink','red','orange','yellow','brown','green','blue','white','black']
    l = [0]*len(l_color)
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
    while cap.isOpened():
        ret, image = cap.read()
        if not ret:
            end_t = time.process_time()
            disp = return_vidsNameAndTime(isInFrame)
            if disp != None:
                found_video.append(vidpath)
                print(found_video)
                print(f"{bcolors.OKBLUE}# Video Name: {vidpath} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.OKGREEN}{disp}{bcolors.ENDC}")
                if args.bag:
                    notify.send(f"\nVideo Name: *{vidpath}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: {disp}\nBag in frame: {return_vidsNameAndTime(bagInFrame)}\nElapsed time:{datetime.timedelta(seconds=end_t - start_t)}\n")
                    print(f"Bag in frame: {bcolors.OKGREEN}{return_vidsNameAndTime(bagInFrame)}{bcolors.ENDC}")
                else:
                    notify.send(f"\n********************\nVideo Name: *{vidpath}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: `{disp}` \nElapsed time:{datetime.timedelta(seconds=end_t - start_t)}")
                print('Elapsed time:', datetime.timedelta(seconds=end_t - start_t))

            else :
                print(f"{bcolors.OKBLUE}# Video Name: {vidpath} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.FAIL}Not Found{bcolors.ENDC} \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")
                notify.send(f"\nVideo Name: *{vidpath}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: Not Found \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")
            break

        if counter%int(fps) == 0 :
                
                boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold)

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

        counter+=1
    cap.release()

    print('Done of process:',os.getpid())
    print()
    return found_video

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
    #parser.add_argument('-w', '--weights', type=str, default='cfg/custom-yolov4-detector_best.weights', help='Path to model weights')
    parser.add_argument('-w', '--weights', type=str, default='yolov4_shirt.weights', help='Path to model weights')
    parser.add_argument('-cfg', '--config', type=str, default='cfg/yolov4-custom-shirt.cfg', help='Path to configuration file')
    parser.add_argument('-l', '--labels', type=str, default='cfg/justshirt.names', help='Path to label file')
    parser.add_argument('-c', '--confidence', type=float, default=0.5, help='Minimum confidence for a box to be detected.')
    parser.add_argument('-t', '--threshold', type=float, default=0.3, help='Threshold for Non-Max Suppression')
    parser.add_argument('-u', '--use_gpu', default=True, action='store_true', help='Use GPU (OpenCV must be compiled for GPU). For more info checkout: https://www.pyimagesearch.com/2020/02/03/how-to-use-opencvs-dnn-module-with-nvidia-gpus-cuda-and-cudnn/')
    parser.add_argument('-s', '--save', default=False, action='store_true', help='Whether or not the output should be saved')
    parser.add_argument('-sh', '--show', default=True, action="store_false", help='Show output')
    parser.add_argument('-col', '--color', type=str, help="color > 'pink','red','orange','yellow','green','blue','brown','white','black' ")
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
    print(args.weights)

    # Load weights using OpenCV
    config_person = "cfg/yolov4.cfg"
    weights_person = "yolov4.weights"
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

    print('Detect \"{}\" shirt and bag'.format(args.color)) if args.bag else print('Detect \"{}\" shirt'.format(args.color))

    
    if args.video_path != '':
        notify.send(f"\n--Start Detect {args.color} shirt--\nIn video path: {args.video_path}")
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

        if args.save:
            width = int(cap.get(3))
            height = int(cap.get(4))
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            name = args.video_path.split("/")[-1].split('.')[0]+'.avi'
            out = cv2.VideoWriter(f'output/{name}', fourcc, 1, (width, height))
        
        while cap.isOpened():
            ret, image = cap.read()
            if not ret:
                end_t = time.process_time()
                disp = return_vidsNameAndTime(isInFrame)
                if disp != None:
                    print(f"{bcolors.OKBLUE}# Video Name: {args.video_path} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.OKGREEN}{disp}{bcolors.ENDC}")

                    if args.bag:
                        print(f"Bag in frame: {bcolors.OKGREEN}{return_vidsNameAndTime(bagInFrame)}{bcolors.ENDC}")
                        notify.send(f"\nVideo Name: *{args.video_path}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: {disp}\nBag in frame: {return_vidsNameAndTime(bagInFrame)}\nElapsed time:{datetime.timedelta(seconds=end_t - start_t)}")
                    else:
                        notify.send(f"\n********************\nVideo Name: *{args.video_path}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: `{disp}` \nElapsed time:{datetime.timedelta(seconds=end_t - start_t)}")
                
                    print('Elapsed time:', datetime.timedelta(seconds=end_t - start_t))

                else :
                    print(f"{bcolors.OKBLUE}# Video Name: {args.video_path} {bcolors.ENDC}\nVideo length: {datetime.timedelta(seconds=vdo_l)} \nIn frame: {bcolors.FAIL}Not Found{bcolors.ENDC} \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")
                    notify.send(f"\nVideo Name: *{args.video_path}* \nVideo length: {datetime.timedelta(seconds=vdo_l)} \n{args.color} shirt in frame: Not Found \nElapsed time: {datetime.timedelta(seconds=end_t - start_t)}")

                break
            if counter%int(fps) == 0:
                    
                    fnum+=1

                    boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold)
                    
                    l = classifyColor(image, boxes, idxs)
                    if (l[l_color.index(args.color)]==1):
                        isInFrame += '1'
                        if args.bag:
                            boxes2, confidences2, classIDs2, idxs2 = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)
                            
                            if checkBag(boxes2, idxs2,classIDs2):
                                bagInFrame += '1'
                            else:
                                bagInFrame += '0'

                    else:
                        isInFrame += '0'
                        bagInFrame += '0'

                    if args.save:
                        image_bb = draw_bounding_boxes3(image, boxes, confidences, classIDs, idxs, colors)
                        out.write(image)

            counter+=1
        cap.release()

        print('Done of process:',os.getpid())
        print()

    if args.videoes_path != '': # run for vids in dir
        notify.send(f"\n--Start Detect {args.color} shirt--\nDirectory Path: {args.videoes_path}")
        #get all the file
        vdo_list = []
        
        found_video = Manager().list()
        
        for vidpath in [vid for vid in os.listdir(args.videoes_path)]:
            if vidpath.endswith(('.mp4',".MOV",'.avi')):
                vdo_list.append(vidpath)

        
        print(vdo_list)
        print('Number of CPU :',mp.cpu_count())
        pool = mp.Pool(processes = mp.cpu_count())
        pool.map_async(partial(vdoprocess, args=args),vdo_list)
 
        pool.close()
        pool.join()
        print(f"Found {args.color} shirt in: {found_video}")
        notify.send(f'\nFound {args.color} shirt in: `{found_video}` ')

    if args.image_path != '':
        notify.send(f"\n--Start Detect {args.color} shirt--\nIn image path: {args.image_path}")
        image = cv2.imread(args.image_path)
            
        # Load weights using OpenCV
        start_time = time.time()
        #boxes, confidences, classIDs, idxs = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold) # Detect from yolov4.weight 

        boxes, confidences, classIDs, idxs = make_prediction(net2, layer_names2, labels2, image, args.confidence, args.threshold) # Detect from shirt weight
        #image = draw_bounding_boxes2(image, boxes, confidences, classIDs, idxs, colors)
        image = draw_bounding_boxes3(image, boxes, confidences, classIDs, idxs, colors)

        
        l = classifyColor(image, boxes, idxs)
        print(f"Process time: {(time.time()-start_time)}")
        #print(l)
        if (l[l_color.index(args.color)]==1):
            if args.bag:
                boxes2, confidences2, classIDs2, idxs2 = make_prediction(net1, layer_names1, labels1, image, args.confidence, args.threshold)  
                print(f"{bcolors.OKGREEN}{args.color} shirt: Found {bcolors.ENDC}")
                if checkBag(boxes2, idxs2,classIDs2):
                    notify.send(f"\nFrom {args.image_path}\n{args.color} `shirt: Found`\n`Bag: Found`")
                    print(f"{bcolors.OKGREEN}Bag: Found {bcolors.ENDC}")
                else:
                    notify.send(f"\nFrom {args.image_path}\n{args.color} `shirt: Found`\n`Bag: Not Found`")
                    print(f"{bcolors.FAIL}Bag: Not Found {bcolors.ENDC}")
            else:

                print(f"{bcolors.OKGREEN}Found {args.color} shirt {bcolors.ENDC}")
                notify.send(f"\nFrom {args.image_path}\n`Found {args.color} shirt` ")
        else:
            print(f"{bcolors.FAIL}Not Found {args.color} shirt {bcolors.ENDC}")
            notify.send(f"\nFrom {args.image_path}\n`Not Found {args.color} shirt` ")
            

        #print('Bag: {}'.format(bag))
        
        # show the output image
        path = str(Path(args.image_path).parent.absolute()) +'\prediction.jpg'
        cv2.imwrite(path, image)
        notify.send("image",image_path=path)
        
    notify.send(f"\n--Finish Detect {args.color} shirt--")
