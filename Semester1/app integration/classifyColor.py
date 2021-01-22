def nearest_colour(r,g,b):
        l = []
  #                  0        1        2       3         4         5       6       7        8        9       10 
        l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
        l_scale = [(100,0,100),(50,0,50),(100,0,0),(100,27,0),\
                    (100,100,0),(0,50,0),(0,100,100),(0,0,100),(50,0,0),(100,100,100),(0,0,0)]
        r, g, b = int(r*100/255), int(g*100/255), int(b*100/255)
        # calculate absolute distance
        for ele in l_scale:
            r0, g0, b0 = ele
            l.append(0.299*0.299*(r-r0)*(r-r0)+0.587*0.587*(g-g0)*(g-g0)+0.114*0.114*(b-b0)*(b-b0))
        print(l)
        color = l_color[l.index(min(l))]
        return color

def classifyColor(img, boxes, idxs):
    numColor = 11
    l = [0]*numColor
    #            0        1        2       3         4         5       6       7        8        9       10 
    l_color = ['pink', 'purple', 'red', 'orange', 'yellow', 'green', 'cyan', 'blue', 'brown', 'white', 'black']
    if len(idxs) > 0:
        for i in idxs.flatten():
            print('number of bbox', i)
            # extract bounding box coordinates
            x, y = boxes[i][0], boxes[i][1]
            w, h = boxes[i][2], boxes[i][3]      
            # (w, h, d) = img.shape

            # change into 70% w/h scale
            wn, hn = int(0.7*w), int(0.7*h)
            xn, yn = int(x+w/2-wn/2), int(y+h/2-hn/2)
            w, h = int(0.7*w), int(0.7*h)
            x, y = int(x+w/2-wn/2), int(y+h/2-hn/2)

            print('start-end x', x, x+wn)
            print('start-end y', y, y+hn)
            print(img.shape)

            red, green, blue = 0, 0, 0
            for width in range(x, x+wn):
                for height in range(y, y+hn):
                      #rgb_pixel_value = img.getpixel((x+width ,y+height))
                      r, g, b = img[height][width][0], \
                      img[height][width][1], img[height][width][2]
                      red += r
                      green += g
                      blue += b

                      #print(width, height)
            red = int(red/(w*h))
            green = int(green/(w*h))
            blue = int(blue/(w*h))
                        
            print(red, green, blue)

            color = nearest_colour(red, green, blue)
            print(color)
            if l[l_color.index(color)] == 0:
                l[l_color.index(color)] = 1
            print(l)
    return l
