import cv2
import numpy as np
import statistics as stat
import serial
import time


#Import cascade files for classifiers
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
can_cascade = cv2.CascadeClassifier('can_cascade.xml')
ser = serial.Serial('/dev/ttyACM0',9600, timeout = 5)
time.sleep(1)

#initialize cameras for video analysis and object detection
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 100);
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 100);

#cap2 = cv2.VideoCapture(2)
#cap2.set(cv2.CAP_PROP_FRAME_WIDTH, 100);
#cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, 100);

#constant declaration for rgb-green value
green = (0,255,0)

inicio = True

#functions declarations
def show(image):
    # Figure size in inches
    plt.figure(figsize=(10, 10))

    # Show image, with nearest neighbour interpolation
    plt.imshow(image, interpolation='nearest')

def overlay_mask(mask, image):
	#make the mask rgb
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    #calculates the weightes sum of two arrays. in our case image arrays
    #input, how much to weight each. 
    #optional depth value set to 0 no need
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Copy
    image = image.copy()
    #input, gives all the contours, contour approximation compresses horizontal, 
    #vertical, and diagonal segments and leaves only their end points. For example, 
    #an up-right rectangular contour is encoded with 4 points.
    #Optional output vector, containing information about the image topology. 
    #It has as many elements as the number of contours.
    #we dont need it

    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    # Isolate largest contour
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]

    if len(contour_sizes) > 0:
        biggest_contour = max(contour_sizes, key=lambda x: x[0])[1]

        mask = np.zeros(image.shape, np.uint8)
        cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
        return biggest_contour, mask

    else:
        return 0,0 

def find_contours(image):
    image = image.copy()

    image, contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def circle_contour(image, contour):
    # Bounding ellipse
    image_with_ellipse = image.copy()

    #ellipse = cv2.fitEllipse(contour)
    #add it
    #cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse

def find_object(image):
    #RGB stands for Red Green Blue. Most often, an RGB color is stored 
    #in a structure or unsigned integer with Blue occupying the least 
    #significant “area” (a byte in 32-bit and 24-bit formats), Green the 
    #second least, and Red the third least. BGR is the same, except the 
    #order of areas is reversed. Red occupies the least significant area,
    # Green the second (still), and Blue the third.
    # we'll be manipulating pixels directly
    #most compatible for the transofrmations we're about to do
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Make a consistent size
    #get largest dimension
    max_dimension = max(image.shape)
    #The maximum window size is 700 by 660 pixels. make it fit in that
    ##scale = 700/max_dimension
    #resize it. same width and hieght none since output is 'image'.
    ##image = cv2.resize(image, None, fx=scale, fy=scale)
    
    #we want to eliminate noise from our image. clean. smooth colors without
    #dots
    # Blurs an image using a Gaussian filter. input, kernel size, how much to filter, empty)
    image_blur = cv2.GaussianBlur(image, (7, 7), 0)
    #t unlike RGB, HSV separates luma, or the image intensity, from
    # chroma or the color information.
    #just want to focus on color, segmentation
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    # Filter by colour
    # 0-10 hue
    #minimum red amount, max red amount
    min_color = np.array([0,0,0])
    max_color = np.array([256,256,65])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_color, max_color)

    #birghtness of a color is hue
    # 170-180 hue
    min_color2 = np.array([0,0,0])
    max_color2 = np.array([256, 256, 65])

    mask2 = cv2.inRange(image_blur_hsv, min_color2, max_color2)

    #looking for what is in both ranges
    # Combine masks
    mask = mask1 + mask2

    # Clean up
    #we want to circle our strawberry so we'll circle it with an ellipse
    #with a shape of 15x15
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    #morph the image. closing operation Dilation followed by Erosion. 
    #It is useful in closing small holes inside the foreground objects, 
    #or small black points on the object.
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    #erosion followed by dilation. It is useful in removing noise
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    # Find biggest strawberry
    #get back list of segmented strawberries and an outline for the biggest one
    big_contour, mask_contour = find_biggest_contour(mask_clean)

    contours = find_contours(mask_clean)

    # Overlay cleaned mask on image
    # overlay mask on image, strawberry now segmented
    #overlay = image
    overlay = overlay_mask(mask_clean, image)


    # Circle biggest strawberry
    #circle the biggest one

    if (type(big_contour) == int):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    else:
        circled = circle_contour(overlay, big_contour)
        #show(circled)
        
        #we're done, convert back to original color scheme
        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        image_with_com = bgr.copy()

        for contour in contours:
            moments = cv2.moments(contour)
            centre_of_mass = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
            cv2.circle(image_with_com, centre_of_mass, 2, (0, 255, 0), -1, cv2.LINE_AA)
        
        bgr = image_with_com
    
    return bgr, contours

def find_red_object(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)

    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_color = np.array([0,100,80])
    max_color = np.array([10,256,256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_color, max_color)

    #birghtness of a color is hue
    # 170-180 hue
    min_color2 = np.array([175,100,80])
    max_color2 = np.array([180, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_color2, max_color2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_contour, mask_contour = find_biggest_contour(mask_clean)

    contours = find_contours(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    if (type(big_contour) == int):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    else:
        circled = circle_contour(overlay, big_contour)

        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        image_with_com = bgr.copy()

        for contour in contours:
            moments = cv2.moments(contour)
            centre_of_mass = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
            cv2.circle(image_with_com, centre_of_mass, 2, (0, 255, 0), -1, cv2.LINE_AA)
        
        bgr = image_with_com
    
    return bgr, contours

def find_blue_object(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    max_dimension = max(image.shape)

    image_blur = cv2.GaussianBlur(image, (7, 7), 0)

    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)

    min_color = np.array([110,60,80])
    max_color = np.array([130,256,256])
    #layer
    mask1 = cv2.inRange(image_blur_hsv, min_color, max_color)

    #birghtness of a color is hue
    # 170-180 hue
    min_color2 = np.array([110, 60,80])
    max_color2 = np.array([130, 256, 256])
    mask2 = cv2.inRange(image_blur_hsv, min_color2, max_color2)

    mask = mask1 + mask2

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))

    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)

    big_contour, mask_contour = find_biggest_contour(mask_clean)

    contours = find_contours(mask_clean)

    overlay = overlay_mask(mask_clean, image)

    if (type(big_contour) == int):
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    else:
        circled = circle_contour(overlay, big_contour)

        bgr = cv2.cvtColor(circled, cv2.COLOR_RGB2BGR)
        image_with_com = bgr.copy()

        for contour in contours:
            moments = cv2.moments(contour)
            centre_of_mass = int(moments['m10'] / moments['m00']), int(moments['m01'] / moments['m00'])
            cv2.circle(image_with_com, centre_of_mass, 2, (0, 255, 0), -1, cv2.LINE_AA)
        
        bgr = image_with_com
    
    return bgr, contours

#Defining Coordinate x buffer for data cleaning
coorX = []

for i in range(10):
	coorX.append(0)


while True:

	#Retrieving images for both cameras.
	ret, img = cap.read()
	#ret2, img2 = cap2.read()

	#Start of the image handling and the graphical presentation of the objects detected
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	logim = gray;
	powim = gray;


	#logaritmic and power transformations
	logim = np.log2(1 + logim.astype(np.float)).astype(np.uint8)
	eqim = cv2.equalizeHist(gray)

	powim = powim.astype(np.float)
	c = (powim.max()) / (powim.max()**(1.5))
	powim = (c*powim.astype(np.float)**(1.5)).astype(np.uint8)

	#definition of cascade classifiers in each transformation type
	cans = can_cascade.detectMultiScale(gray)
	canseq = can_cascade.detectMultiScale(eqim)


	#Searching for the nearest object in the vecinity, using only the y-cooordinate
	miny = 100;
	thebox = ( )
	for(x,y,w,h) in cans:
		cv2.rectangle(gray, (x,y), (x+w, y+h),(255,255,0), 2)

	#center to print
	center = (0, 0)

	for i in range(len(canseq)):
		(x,y,w,h) = canseq[i]
		if y < miny:
			miny = y
			thebox = canseq[i]
			center = (x+w/2,y+h/2)

	if len(thebox) > 0:
		cv2.rectangle(eqim, (thebox[0],thebox[1]), (thebox[0]+thebox[2], thebox[1]+thebox[3]),(255,255,0), 2)


	#Printing the X coordinate in the console, and writing it to arduinos serial.
	coorX.pop(0)
	coorX.append(center[0])

	#valX = stat.median(coorX) - 80
	valX = center[0] - 80

	if valX == -80:
		valX = 0

	print(valX)
    ser.write(valX)
	

	inicio == False

	imagecans, contours = find_object(img)
	imagered, redcontours = find_red_object(img)
	imageblue, bluecontours = find_blue_object(img)

	#creating display 
	display = np.zeros(img.shape, np.uint8)

	display = cv2.putText(display,str(valX), (40,65), cv2.FONT_HERSHEY_SCRIPT_SIMPLEX, 1, (green))

	img = cv2.resize(img,(500,500))
	imagecans = cv2.resize(imagecans, (500,500))
	eqim = cv2.resize(eqim,(500,500))
	display = cv2.resize(display,(500,500))
	imagered = cv2.resize(imagered, (500,500))
	imageblue = cv2.resize(imageblue, (500,500))

	eqim = cv2.cvtColor(eqim, cv2.COLOR_GRAY2BGR)

	col1 =  np.vstack([img, imagecans])
	col2 = 	np.vstack([display, eqim])
	col3 = np.vstack([imageblue, imagered])
	
	im = np.hstack([col1, col2, col3])

	cv2.imshow('gray',im)

	k = cv2.waitKey(30) & 0xff
	if k == 27:
		break

cap_release()
cv2.destroyAllWindows()