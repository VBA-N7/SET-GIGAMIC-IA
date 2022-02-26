import cv2
import numpy as np
from keras.models import load_model
import os


class card(object):
    """docstring for card"""

    def __init__(self, contours, perimeter, frame):
        super(card, self).__init__()
        self.contours = contours
        self.perimeter = perimeter
        self.frame = frame
        
        x, y, w, h = cv2.boundingRect(self.contours)
        self.x = (x, x + w)
        self.y = (y, y + h) 
        
        self.prediction = None
        self.prediction_accuracy = None
        
        try:
            crop_ratio = 0.9
            
            minrect = cv2.minAreaRect(self.contours)
            self.box = np.int0(cv2.boxPoints(minrect))
            
            W = minrect[1][0]
            H = minrect[1][1]

            Xs = [i[0] for i in self.box]
            Ys = [i[1] for i in self.box]
            y2 = max(Ys)
            x1 = min(Xs)
            x2 = max(Xs)
            y1 = min(Ys)

            rotated = False
            angle = minrect[2]

            if angle < -45:
                angle+=90
                rotated = True

            center = (int((x1+x2)/2), int((y1+y2)/2))
            size = (int(crop_ratio*(x2-x1)),int(crop_ratio*(y2-y1)))

            M = cv2.getRotationMatrix2D((size[0]/2, size[1]/2), angle, 1.0)

            # cropped = cv2.getRectSubPix(raw_pic, size, center)
            cropped = cv2.getRectSubPix(self.frame, size, center)
            cropped = cv2.warpAffine(cropped, M, size)

            croppedW = W if not rotated else H 
            croppedH = H if not rotated else W

            self.img = cv2.getRectSubPix(cropped, (int(croppedW*crop_ratio), int(croppedH*crop_ratio)), (size[0]/2, size[1]/2))

            # preparing img for IA - resize
            self.img_IA_rdy = cv2.resize(self.img, (224, 224), cv2.INTER_AREA)
            # cast into npa array
            self.img_IA_rdy = np.asarray(self.img_IA_rdy)
            # normalize
            self.img_IA_rdy = (self.img_IA_rdy.astype(np.float32) / 127.0) - 1
        except Exception:
            self.img = None
            self.img_IA_rdy = None
        


class set_finder(object):
    """docstring for set_finder."""
    def __init__(self, keras_model_path, model_labels, fixed_perimeter=None,):
        super(set_finder, self).__init__()
        self.labels = model_labels
        self.model = load_model(keras_model_path, compile=False)
        self.fixed_perimeter = fixed_perimeter
        
    def feed(self, frame):
        # resize pics
        frame = cv2.resize(frame, (int(frame.shape[1] / 2),
                                   int(frame.shape[0] / 2)))
    
        # darkening pic
        gray_pic = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blured_pic = cv2.GaussianBlur(gray_pic, (15, 15), 0)
        
        # thresholding
        _, threshold_pic = cv2.threshold(blured_pic,
                                         0,
                                         255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # getting contours
        contours, hierarchy = cv2.findContours(threshold_pic,
                                               cv2.RETR_TREE,
                                               cv2.CHAIN_APPROX_SIMPLE)     

        # getting perimeters of contours
        perimeters = [cv2.arcLength(cnt, True) for cnt in contours]

        if self.fixed_perimeter is None:
            #calculating perimeter median
            median_perimeters = np.median(perimeters)
            print('Median perimeters: {}'.format(median_perimeters))
        else: 
            median_perimeters = self.fixed_perimeter

        # filtering perimeter depending on their value
        cards_list = [card(contours=cnt, perimeter=per, frame=frame) for (cnt, per) in zip(contours, perimeters) if (per > (median_perimeters - 100) and (per < (median_perimeters + 100)))]
        cards_list = [crd for crd in cards_list if (crd.img is not None and crd.img_IA_rdy is not None)]
        print('{} cards detected'.format(len(cards_list)))
        
        if len(cards_list) != 0:
            # dataset for model
            data = np.array([card.img_IA_rdy for card in cards_list], dtype=np.float32)

            prediction_list = self.model.predict(data)

            for (crd, prediction) in zip(cards_list, prediction_list): 
                crd.prediction = self.labels[prediction.argmax()]
                crd.prediction_accuracy = prediction.max() * 100
                
                # draw contours
                frame = cv2.drawContours(frame, [crd.box], 0, (0, 0, 255), 1)
                # write prediction accuracy
                position = (crd.x[0],
                            int((crd.y[0] + crd.y[1]) / 2))
                cv2.putText(img=frame,
                            text='{} {:.1f}%'.format(crd.prediction,
                                                    crd.prediction_accuracy),
                            org=position,
                            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                            fontScale=0.4,
                            color=(255, 0, 0),
                            thickness=1)
        return frame

model_path = os.getcwd() + '\\keras_model.h5'
with open('labels.txt') as f:
    labels = tuple(f.readline().split(' '))

camera = cv2.VideoCapture(0)
finder = set_finder(keras_model_path=model_path, 
                    model_labels=labels)

while((cv2.waitKey(5) & 0xFF) != 27):
    camera_pic = camera.read()[1]
    # cv2.imshow('raw frame from camera', camera_pic)
    frame = finder.feed(frame=camera_pic)
    cv2.imshow('Final', frame)
