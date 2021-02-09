import os
import glob
import shutil
import cv2
import torch
import copy
import numpy as np

ROOTPATH = '/home/ma-user/work/'

class CutImage():
    """
    finish the final task:
    create folderÃ¯Â¼Å¡input_video, output
    create labels folder: person, car etc
    output cut image
    """

    def __init__(self, boxes=None, scores=None, labels=None, metadata=None, panoptic_seg=None, segments_info=None, image=None):
        self.image = image
        self.boxes = boxes
        self.scores = scores
        self.labels = labels
        self.notThing_labels = None
        self.metadata = metadata
        self.panoptic_seg = panoptic_seg
        self.segments_info = segments_info
        self.input_video_folder = ROOTPATH + 'sourcecode/input_video'
        self.output_folder = ROOTPATH + 'sourcecode/output'
        self.classes = {}

    def main(self):
        self.updataClasses(self.labels)
        self.createFolders()
        self.cutIsThing()
        text = self.cutNotThing(self.panoptic_seg, self.segments_info)
        self.updataClasses(text)

    def createFolders(self):
        """
        create the two root folders as well as labels folders
        """
        if not os.path.exists(self.input_video_folder):
            os.mkdir(self.input_video_folder)
        if not os.path.exists(self.output_folder):
            os.mkdir(self.output_folder)
        if not os.path.exists(self.output_folder + '/imageSeg'):
            os.mkdir(self.output_folder + '/imageSeg')
        if not os.path.exists(self.output_folder + '/outputimageRec'):
            os.mkdir(self.output_folder + '/outputimageRec')
        for key in self.classes.keys():
            if not os.path.exists(self.output_folder + '/imageSeg/' + key):
                os.mkdir(self.output_folder + '/imageSeg/' + key)

    def cutIsThing(self):
        """
        cut the little images and save them into the object folders
        """
        for id, box in enumerate(self.boxes):
            x0, y0, x1, y1 = box[: 4]
            little_img = self.image[int(y0): int(y1), int(x0): int(x1)]
            obj = self.labels[id].split(' ')[0]
            # print(obj)
            if os.path.exists(self.output_folder + '/imageSeg/' + obj + '/0.jpg'):
                max_id = self.getMaxPictureId(self.output_folder + '/imageSeg/' + obj)
                # print(max_id, type(max_id))
                cv2.imwrite(self.output_folder + '/imageSeg/' + obj + '/' + str(max_id + 1) + '.jpg', little_img)
            else:
                cv2.imwrite(self.output_folder + '/imageSeg/' + obj + '/' + '0.jpg', little_img)

    def cutNotThing(self, panoptic_seg, segments_info):
        """:arg
        cut is not thing
        """
        pred = _PanopticPrediction(panoptic_seg, segments_info)
        # false_thing_segments_info = list(filter(lambda x: x['isthing'] == False, segments_info))
        text = []
        for mask, sinfo in pred.semantic_masks():
            img = copy.deepcopy(self.image)
            category_idx = sinfo["category_id"]
            if self.metadata.stuff_classes[category_idx] == None:
                continue
            # for k in range(int(mask.shape[0])):
            #     for j in range(int(mask.shape[1])):
            #         if mask[k][j] == False:
            #             img[k][j] = 0
            mask = np.repeat(mask[:, :, np.newaxis], 3, axis=2) 
            img = img * mask
            # img = img[:,:,0] * mask
            # img = img[:,:,1] * mask
            # img = img[:,:,2] * mask
            # label
            label = self.metadata.stuff_classes[category_idx]
            # create folder
            if not os.path.exists(self.output_folder + '/imageSeg/' + label):
                os.mkdir(self.output_folder + '/imageSeg/' + label)
                cv2.imwrite(self.output_folder + '/imageSeg/' + label + '/' + '0.jpg', img)
            else:
                max_index = self.getMaxPictureId(self.output_folder + '/imageSeg/' + label)
                cv2.imwrite(self.output_folder + '/imageSeg/' + label + '/' + str(max_index + 1) + '.jpg' , img)

            text.append(label)
        return text

    def updataClasses(self, labels):
        for label in labels:
            if label in self.classes:
                self.classes[label] += 1
            else:
                self.classes[label] = 1

    def getMaxPictureId(self, path):
        """
        to get the max number index of the picture in certain folder
        return: max number
        """
        picture_id_list = glob.glob(path + '/*.jpg')
        picture_id_list = list(map(lambda x: x.split('/')[-1], picture_id_list))
        max_id = max(picture_id_list, key=lambda x: int(x.split('.')[0]))
        return int(max_id.split('.')[0])

    def clearAllImgs(self, ifTotal=True):
        """
        clear all images in folder imageSeg as well as in folder outputimageRec
        """
        if os.path.exists(self.output_folder + '/imageSeg'):
            shutil.rmtree(self.output_folder + '/imageSeg')
        if ifTotal:
            if os.path.exists(self.output_folder + '/outputimageRec'):
                shutil.rmtree(self.output_folder + '/outputimageRec')

    def getClasses(self):
        """
        return total classes as well as its number in one image
        """
        return self.classes


class _PanopticPrediction:
    def __init__(self, panoptic_seg, segments_info):
        self._seg = panoptic_seg

        self._sinfo = {s["id"]: s for s in segments_info}  # seg id -> seg info
        segment_ids, areas = torch.unique(panoptic_seg, sorted=True, return_counts=True)
        areas = areas.cpu().numpy()
        sorted_idxs = np.argsort(-areas)
        self._seg_ids, self._seg_areas = segment_ids[sorted_idxs], areas[sorted_idxs]
        self._seg_ids = self._seg_ids.tolist()
        for sid, area in zip(self._seg_ids, self._seg_areas):
            if sid in self._sinfo:
                self._sinfo[sid]["area"] = float(area)

    def non_empty_mask(self):
        """
        Returns:
            (H, W) array, a mask for all pixels that have a prediction
        """
        empty_ids = []
        for id in self._seg_ids:
            if id not in self._sinfo:
                empty_ids.append(id)
        if len(empty_ids) == 0:
            return np.zeros(self._seg.shape, dtype=np.uint8)
        assert (
                len(empty_ids) == 1
        ), ">1 ids corresponds to no labels. This is currently not supported"
        return (self._seg != empty_ids[0]).numpy().astype(np.bool)

    def semantic_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or sinfo["isthing"]:
                # Some pixels (e.g. id 0 in PanopticFPN) have no instance or semantic predictions.
                continue
            yield (self._seg == sid).cpu().numpy().astype(np.bool), sinfo

    def instance_masks(self):
        for sid in self._seg_ids:
            sinfo = self._sinfo.get(sid)
            if sinfo is None or not sinfo["isthing"]:
                continue
            mask = (self._seg == sid).cpu().numpy().astype(np.bool)
            if mask.sum() > 0:
                yield mask, sinfo
