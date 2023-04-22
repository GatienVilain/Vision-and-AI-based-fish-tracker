import os
import cv2

coordinates = []
image = 0
i = 0

# using detect to predict on a pic or vid
def detect():
    os.system(
        'python yolov5_ws/yolov5/detect.py --weights yolov5_ws/yolov5/runs/train/exp4/weights/best.pt --img 640 --conf 0.4 --source Verification/1.mp4')

# Use camera to detect on camera output
def detect_cam():
    # activate camera
    cam = cv2.VideoCapture(0)
    # use model best.pt to detect on camera output
    os.system('python yolov5_ws/yolov5/detect.py --source 0 --weights yolov5_ws/data/best.pt --conf 0.25')
    # show camera output
    while True:
        ret, frame = cam.read()
        cv2.imshow("camera", frame)
        # we can use frame to control mouvement or speed maybe
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break



# function to display the coordinates of the points clicked on the image
def click_event(event, x, y, flags, param):
    # checking for left mouse clicks
    if event == cv2.EVENT_LBUTTONDOWN:
        global i
        i += 1
        if i > 2:
            return
        # normalizing the coordinates
        x = round(float(x / image.shape[0]), 3)
        y = round(float(y / image.shape[1]), 3)

        # displaying the coordinates on the image window
        x2 = int(x * image.shape[0])
        y2 = int(y * image.shape[1])
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image, str(x) + ',' +
                    str(y), (x2, y2), font,
                    1, (255, 0, 0), 2)
        cv2.imshow('image', image)

        coordinates.append([x, y])
    return 1

# function to get the coordinates of the points clicked on the image
def get_coordinates(img):
    global i
    i = 0
    cv2.imshow('image', img)
    cv2.setMouseCallback('image', click_event)
    while i < 2:
        cv2.waitKey(20)
    cv2.destroyAllWindows()
    return

# function to get the size of the rectangle from 2 clicked points
def rec_size():
    # get the size of the rectangle from 2 points
    # the 2 points are the 2 corners of the rectangle
    # the 2 points are the 2 coordinates of the 2 corners of the rectangle
    x = coordinates[0][0]
    y = coordinates[0][1]
    x2 = coordinates[1][0]
    y2 = coordinates[1][1]
    # width of the rectangle
    w = abs(x - x2)
    # height of the rectangle
    h = abs(y - y2)
    # center of the rectangle
    c = (x + w / 2, y + h / 2)
    return c, w, h

# function to display the rectangle on the image
def display_square(img, c, w, h):
    # class, x_center, y_center, width, height format of the bounding box normalized
    # upper left corner of the image is (0,0)
    # lower right corner of the image is (1,1)
    # upper right corner of the image is (1,0)
    # lower left corner of the image is (0,1)
    # get value of 2 points of coordinates

    p1 = (int(coordinates[0][0] * img.shape[0]), int(coordinates[0][1] * img.shape[1]))
    p2 = (int(coordinates[1][0] * img.shape[0]), int(coordinates[1][1] * img.shape[1]))
    img = cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
    # add a point in the center of the rectangle
    img = cv2.circle(img, (int(c[0] * img.shape[0]), int(c[1] * img.shape[1])), 5, (0, 0, 255), -1)

    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Use all the functions to create a dataset to train a yolo model
def create_dataset():
    # for every element in p that are .jpg
    for img in os.listdir('yolov5_ws/pic'):
        # print the name of img
        print(img)
        name = img
        # if name.txt exists, break
        if os.path.exists('yolov5_ws/pic/' + name[:-4] + '.txt'):
            print('already exists')
            continue

        path = 'yolov5_ws/pic/' + img
        path2 = 'yolov5_ws/pic/' + img[:-4] + '.txt'

        img = cv2.imread(path, 1)
        new_width = 720
        dsize = (new_width, img.shape[0])
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        dsize = (new_width, img.shape[1])
        # resize image
        img = cv2.resize(img, dsize, interpolation=cv2.INTER_AREA)
        global image
        image = img
        data = []
        # while enter is not pressed, the image is displayed
        loop_break = False
        while not loop_break:
            print("entering loop")
            get_coordinates(img)
            c, w, h = rec_size()
            # print(c, w, h)
            display_square(img, c, w, h)
            data.append([0, c[0], c[1], w, h])
            if cv2.waitKey(20) & 0xFF == 13:
                loop_break = True
        #print(data)


        # create a .txt file with the same name as the image
        with open(path2, 'w') as f:
            f.write(data)


        coordinates.clear()



# count the number of txt files in the folder and display the percentage of txt files compared to the number of jpg files
# It allows the user to know how many images have been processed
def count_txt():
    i = 0
    z = 0
    # For each tect file in the folder pic
    for txt in os.listdir('yolov5_ws/pic'):
        if txt[-3:] == 'txt':
            i += 1
        #else it's jpg file
        else:
            z += 1
    # print the percentage of txt files compared to the number of jpg files
    print(i,"txt done on",z,"jpg")
    print( i/z*100,"%")

if __name__ == "__main__":
    print('start')

    # use detect when you want to detect on camera to test...
    # detect()

    # Display the percentage of txt files compared to the number of jpg files
    count_txt()
    #
    create_dataset()
    print('end')
