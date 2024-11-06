import numpy as np
import cv2
from shapely.geometry import Polygon
import pyclipper
import math
from concern.config import State
from .data_process import DataProcess2
from collections import OrderedDict

class MakeSizeGaussData(DataProcess2):
    r'''
    Making binary mask from detection data with ICDAR format.
    Typically following the process of class `MakeICDARData`.
    '''
    min_text_size = State(default=5)
    shrink_ratio = State(default=0.4)
    gauss_size = State(default=15)
    gauss_scale = State(default=1000)
    def __init__(self, **kwargs):
        self.load_all(**kwargs)

        self.gauss_size = int(self.gauss_size)
        self.guass_kernel = self.GaussianKernel(shape=(self.gauss_size, self.gauss_size), sigma=4)
    def Addkernel(self, center_points,guass_kernel,center_den_IPM,value):
        center_x = int(center_points[1])
        center_y = int(center_points[0])
        h, w = center_den_IPM.shape
        # for z in range(len(center_x)) :
        offset_l = (self.gauss_size-1)/2
        offset_r = (self.gauss_size+1)/2
        cut_x1, cut_x2,cut_y1,cut_y2 = 0, 0, 0, 0
        x, y = center_x,center_y
        x1, y1, x2, y2 = x-offset_l,y-offset_l, x+offset_r, y+offset_r
        if x1<0:
            cut_x1 = 0-x1            
            x1 = 0
        if y1<0:
            cut_y1 = 0-y1            
            y1 = 0
        if x2 > w-1:
            cut_x2 = x2-w+1            
            x2 = int(w-1)
        if y2> h-1:
            cut_y2 = y2-h+1            
            y2 = int(h-1)
        xxx = int(self.gauss_size-cut_y2)
        xxxx = int(self.gauss_size-cut_x2)
        # print(type(xxx), xxxx,type(y1),type(cut_x1))
        # print(self.gauss_size,cut_y2)
        # a = center_den_IPM[y1:y2, x1:x2]
        # b = guass_kernel[cut_y1:25-cut_y2,cut_x1:25-cut_x2]
        center_den_IPM[int(y1):int(y2), int(x1):int(x2)]+=guass_kernel[int(cut_y1):xxx,int(cut_x1):xxxx]*value
        return center_den_IPM

    def GaussianKernel(self, shape=(15, 15), sigma=0.5):
        """
        2D gaussian kernel which is equal to MATLAB's fspecial('gaussian',[shape],[sigma])
        """
        radius_x, radius_y = [(radius-1.)/2. for radius in shape]
        y_range, x_range = np.ogrid[-radius_y:radius_y+1, -radius_x:radius_x+1]
        h = np.exp(- (x_range*x_range + y_range*y_range) / (2.*sigma*sigma))  # (25,25),max()=1~h[12][12]
        h[h < (np.finfo(h.dtype).eps*h.max())] = 0
        max = h.max()
        min = h.min()
        h = (h-min)/(max-min)
        sumh = h.sum()
        if sumh != 0:
            h /= sumh
        # a=h.sum()
        return h
    def process(self, data):
        '''
        requied keys:
            image, polygons, ignore_tags, filename
        adding keys:
            mask
        '''
        # mid_area = 0
        # ints = 0
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        image = data['image']
        filename = data['filename']
        text = data['text']
        h, w = image.shape[:2]
        if data['is_training']:
            polygons, ignore_tags = self.validate_polygons(
                polygons, ignore_tags, h, w)

        mask = np.ones((h, w), dtype=np.float32)
        gauss = np.zeros(( h, w), dtype=np.float32)
        size_balance= np.ones((1,h, w), dtype=np.float32)
        for i in range(len(polygons)):
            temp = np.zeros(( h, w), dtype=np.float32)
            polygon = polygons[i]

            height = max(polygon[:, 1]) - min(polygon[:, 1])
            width = max(polygon[:, 0]) - min(polygon[:, 0])

            if ignore_tags[i] or min(height, width) < self.min_text_size:
                cv2.fillPoly(mask, polygon.astype(
                    np.int32)[np.newaxis, :, :], 0)
                ignore_tags[i] = True
            else:
                polygon_shape = Polygon(polygon)
                distance = polygon_shape.area * \
                    (1 - np.power(self.shrink_ratio, 2)) / polygon_shape.length
                subject = [tuple(l) for l in polygons[i]]
                padding = pyclipper.PyclipperOffset()
                padding.AddPath(subject, pyclipper.JT_ROUND,
                                pyclipper.ET_CLOSEDPOLYGON)
                distance = np.round(distance)
                shrinked = padding.Execute(-distance)
                if shrinked == []:
                    cv2.fillPoly(mask, polygon.astype(
                        np.int32)[np.newaxis, :, :], 0)
                    ignore_tags[i] = True
                    continue
                shrinked = np.array(shrinked[0]).reshape(-1, 2)    
                # area = Polygon(shrinked).area 
                # if area <1000:
                #    cv2.fillPoly(tiny_mask, [polygon.astype(np.int32)], 1)
                cv2.fillPoly(temp, [shrinked.astype(np.int32)], 1)         
                cv2.fillPoly(size_balance[0], [shrinked.astype(np.int32)], len(text[i]['text'].replace(' ', '').replace('"', '')))
                points = np.array(np.where(temp==1))
                mid_x = (points[1].min() + points[1].max())//2
                indices = np.where([points[1]==mid_x])[1]
                points_f = points[:, indices]
                points_f = np.array(points_f).mean(axis=1)
                # print(points_f,mid_x,points)
                # print(text[i]['text'])
                gauss = self.Addkernel((points_f), self.guass_kernel, gauss, len(text[i]['text'].replace(' ', '').replace('"', '')))


        # print(gauss.max())
        # cv2.imwrite('temp.png',(gauss*20000).astype(np.uint8))
        # cv2.imwrite('temp2.png',(gt[0]*255).astype(np.uint8))
        # print(size.shape, gt.shape,mask.shape)
        #print(kwmask.max(),kwmask.min())
        if filename is None:
            filename = ''
        # return OrderedDict(image=data['image'],
        #                 polygons=polygons,
        #                 shape=data['shape'],
        #                 ignore_tags=ignore_tags,
        #                 gauss = gauss*1000,
        #                 filename=filename,
        #                 is_training=data['is_training'],
        #                 )
        data.update(size = gauss*self.gauss_scale,
                    size_balance = size_balance
                    # size=size
                   )
        data.pop('text')
        return data

    def validate_polygons(self, polygons, ignore_tags, h, w):
        '''
        polygons (numpy.array, required): of shape (num_instances, num_points, 2)
        '''
        if len(polygons) == 0:
            return polygons, ignore_tags
        assert len(polygons) == len(ignore_tags)
        for polygon in polygons:
            polygon[:, 0] = np.clip(polygon[:, 0], 0, w - 1)
            polygon[:, 1] = np.clip(polygon[:, 1], 0, h - 1)

        for i in range(len(polygons)):
            area = self.polygon_area(polygons[i])
            if abs(area) < 1:
                ignore_tags[i] = True
            if area > 0:
                polygons[i] = polygons[i][::-1, :]
        return polygons, ignore_tags

    def polygon_area(self, polygon):
        edge = 0
        for i in range(polygon.shape[0]):
            next_index = (i + 1) % polygon.shape[0]
            edge += (polygon[next_index, 0] - polygon[i, 0]) * (polygon[next_index, 1] + polygon[i, 1])

        return edge / 2.

