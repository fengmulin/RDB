import cv2
import numpy as np
from shapely.geometry import Polygon,LineString
import pyclipper
from concern.config import Configurable, State
import numba


@numba.jit(nopython=True)
def cal_boxs( points:np.array, dis_x:np.array ,dis_y:np.array,dest_width:np.array, 
            dest_height:np.array,width_ratio:np.array,height_ratio:np.array):
    lens = points.shape[0]
    
    box = []
    
    for iii in range(lens):  
        xx,yy = points[iii]
        if round(yy)<dis_x.shape[0]:
            int_y = round(yy)
        else:
            int_y = dis_x.shape[0]-1
        if round(xx)<dis_x.shape[1]:
            int_x = round(xx)
        else:
            int_x = dis_x.shape[1]-1
        offset_x = dis_x[int_y][int_x]
        offset_y = dis_y[int_y][int_x]
        box.append([round(xx+offset_x),round(yy+offset_y)])
        #box.append([xx, yy])
    box = np.array(box).astype(np.int16)
    box[:, 0] = box[:, 0]* width_ratio
    box[:, 0][box[:, 0]>dest_width] = dest_width
    box[:, 1] = box[:, 1]* height_ratio
    box[:, 1][box[:, 1]>dest_height] = dest_height
    return box
def cal_box(points:np.array, dis_x:np.array ,dis_y:np.array,dest_width:np.array, 
            dest_height:np.array,width_ratio:np.array,height_ratio:np.array):
    lens = points.shape[0]
    # print('qzzzz')
    # raise
    # print(points)
    # points = np.round(points)
    box = []
    for iii in range(lens):  
        xx,yy = points[iii]
        # dis1 = (dis_x[yy-1:yy+2,xx-1:xx+2]).sum()/9
        # dis2 = (dis_y[yy-1:yy+2,xx-1:xx+2]).sum()/9
        # print(dis1, dis_x[yy][xx])
        # box.append([round(points[iii][0]+dis1), round(points[iii][1]+dis2)])
        box.append([points[iii][0]+dis_x[yy][xx], points[iii][1]+dis_y[yy][xx]])
        #box.append([xx, yy])
    box = np.array(box)
    box[:, 0] = box[:, 0]* width_ratio
    # box[:, 0][box[:, 0]>dest_width] = dest_width
    box[:, 1] = box[:, 1]* height_ratio
    # box[:, 1][box[:, 1]>dest_height] = dest_height
    return box
class SegDetectorRepresenter(Configurable):
    thresh = State(default=0.3)
    box_thresh = State(default=0.7)
    max_candidates = State(default=100)
    dest = State(default='binary')

    def __init__(self, cmd={}, **kwargs):
        self.load_all(**kwargs)
        self.min_size = 3
        #self.scale_ratio = 0.4
        if 'debug' in cmd:
            self.debug = cmd['debug']
        if 'thresh' in cmd:
            self.thresh = cmd['thresh']
        if 'box_thresh' in cmd:
            self.box_thresh = cmd['box_thresh']
        if 'dest' in cmd:
            self.dest = cmd['dest']
        #self.box_thresh =0.5
    def represent(self, height, width,binary,dis_x,dis_y,  is_output_polygon=False):
        '''
        batch: (image, polygons, ignore_tags
        batch: a dict produced by dataloaders.
            image: tensor of shape (N, C, H, W).
            polygons: tensor of shape (N, K, 4, 2), the polygons of objective regions.
            ignore_tags: tensor of shape (N, K), indicates whether a region is ignorable or not.
            shape: the original shape of images.
            filename: the original filenames of images.
        pred:
            binary: text region segmentation map, with shape (N, 1, H, W)
            thresh: [if exists] thresh hold prediction with shape (N, 1, H, W)
            thresh_binary: [if exists] binarized with threshhold, (N, 1, H, W)
        '''

        segmentation = self.binarize(binary)

        boxes_batch = []
        scores_batch = []
        
        if is_output_polygon:
            boxes, scores = self.polygons_from_bitmap(
                binary,dis_x,dis_y,
                segmentation, width, height)
        else:
            boxes, scores = self.boxes_from_bitmap(
                binary,dis_x,dis_y,
                segmentation, width, height)
    
        boxes_batch.append(boxes)
        scores_batch.append(scores)
       
        return boxes_batch, scores_batch
    
    def binarize(self, pred):
        return pred > self.thresh

    def polygons_from_bitmap(self,binary,dis_x,dis_y,  bitmap,dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        bitmap = bitmap  # The first channel
        height, width = bitmap.shape
        boxes = []
        scores = []

        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
       # print(pred.shape)
        for contour in contours[:self.max_candidates]:

            epsilon = 0.002 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape((-1, 2))

            score = self.box_score_fast(binary, points.reshape(-1, 2))
            if self.box_thresh > score:
                continue
            #lens = points.shape[0]
            box=[]
            width_ratio = dest_width/width
            height_ratio = dest_height/height
            # pointss = np.round(points)
            box =cal_box(points,dis_x ,dis_y,dest_width, dest_height,width_ratio,height_ratio)
            # box = box.astype(np.int16)
            
            boxes.append(box.tolist())
            scores.append(score)
        return boxes, scores

    def boxes_from_bitmap(self, binary,dis_x,dis_y,  bitmap,dest_width, dest_height):
        '''
        _bitmap: single map with shape (1, H, W),
            whose values are binarized as {0, 1}
        '''
        # print(_bitmap.size)
        height, width = bitmap.shape
        contours, _ = cv2.findContours(
            (bitmap).astype(np.uint8),
            cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        num_contours = min(len(contours), self.max_candidates)
        boxes = np.zeros((num_contours, 4, 2), dtype=np.int16)
        scores = np.zeros((num_contours,), dtype=np.float32)
        for index in range(num_contours):
            
            contour = contours[index]
            # print(contour.shape)
            bounding_box = cv2.minAreaRect(contour)
            sside = min(bounding_box[1])
            if sside < self.min_size:
                continue
            bbox = cv2.boxPoints(bounding_box)
            score = self.box_score_fast(binary, bbox)
            if self.box_thresh > score:
                continue
            points = np.array(bbox)

            width_ratio = dest_width/width
            height_ratio = dest_height/height

            box = cal_box(points,dis_x,dis_y,dest_width, dest_height,width_ratio,height_ratio)
            boxes[index, :, :] = box.astype(np.int16)
            scores[index] = score
            #t7 = time.time()
           # print("last:",(t2-t1)/50," db:",(t3-t2)/50," numba:",(t4-t3)/50," string:",(t5-t4)/50," ori:",(t6-t5)/50,(t7-t6)/50)
        return boxes, scores

    def get_mini_boxes(self, contour):
        contour = np.float32(contour)
        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2],
               points[index_3], points[index_4]]
        return box, min(bounding_box[1])

    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax+1, xmin:xmax+1], mask)[0]
