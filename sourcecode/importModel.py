# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger

setup_logger()

# import some common libraries
import os, sys
import numpy as np
import cv2
import argparse
import shutil
from tqdm import tqdm

# import some common detectron2 utilities
ROOTPATH = '/content/drive/MyDrive/python综合实验/'
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer, ColorMode, GenericMask
from detectron2.data import MetadataCatalog
# os.chdir(ROOTPATH + 'sourcecode/tools')
# list_dir = os.listdir(ROOTPATH + 'sourcecode/tools')
# print(list_dir)
# from detectron2.utils.video_visualizer import _DetectedInstance
import pycocotools.mask as mask_util
from detectron2.utils.colormap import random_color
from tools.cutImg import CutImage, _PanopticPrediction

_SMALL_OBJECT_AREA_THRESH = 1000
_LARGE_MASK_AREA_THRESH = 120000
_OFF_WHITE = (1.0, 1.0, 240.0 / 255)
_BLACK = (0, 0, 0)
_RED = (1.0, 0, 0)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobType',
                        help='segment image or video')
    parser.add_argument('--path_inputVideo',
                        default=os.path.join('/home/ma-user/work/', 'sourcecode/input_video/', 'inputvideo.mp4'),
                        help='the path of input video')
    parser.add_argument('--path_outputVideo',
                        default=os.path.join('/home/ma-user/work/', 'sourcecode/output/', 'outputvideo.mp4'),
                        help='the path of output video')
    parser.add_argument('--path_imageSeg',
                        default=os.path.join(ROOTPATH, 'sourcecode/output/imageSeg'),
                        help='the path of imageSeg')
    parser.add_argument('--path_totalCount',
                        default=os.path.join(ROOTPATH, 'sourcecode/output/totalCount.txt'),
                        help='the path of totalCount')
    parser.add_argument('-t', '--time_interval', type=int, default=10)
    args = parser.parse_args()
    return args

def _create_text_labels(classes, scores, class_names):
    labels = None
    if classes is not None and class_names is not None and len(class_names) > 1:
        labels = [class_names[i] for i in classes]
    if scores is not None:
        if labels is None:
            labels = ["{:.0f}%".format(s * 100) for s in scores]
        else:
            labels = ["{}:{:.0f}%".format(l, s * 100) for l, s in zip(labels, scores)]
    return labels


class _DetectedInstance:
    """
    Used to store data about detected objects in video frame,
    in order to transfer color to objects in the future frames.

    Attributes:
        label (int):
        bbox (tuple[float]):
        mask_rle (dict):
        color (tuple[float]): RGB colors in range (0, 1)
        ttl (int): time-to-live for the instance. For example, if ttl=2,
            the instance color can be transferred to objects in the next two frames.
    """

    __slots__ = ["label", "bbox", "score", "mask_rle", "color", "ttl"]

    def __init__(self, label, bbox, score, mask_rle, color, ttl):
        self.label = label
        self.bbox = bbox
        self.score = score
        self.mask_rle = mask_rle
        self.color = color
        self.ttl = ttl


class Visualizer_rewrite(Visualizer):
    def __init__(self, img_rgb, _old_instances, obj_classes, boxes, metadata=None, scale=1.0,
                 instance_mode=ColorMode.IMAGE):
        super().__init__(img_rgb, metadata, scale, instance_mode)
        self._old_instances = _old_instances
        self.obj_classes = obj_classes
        self.new_instances = None
        self.boxes = boxes

    def draw_instance_predictions(self, predictions):
        """
        Draw instance-level prediction results on an image.

        Args:
            predictions (Instances): the output of an instance detection/segmentation
                model. Following fields will be used to draw:
                "pred_boxes", "pred_classes", "scores", "pred_masks" (or "pred_masks_rle").

        Returns:
            output (VisImage): image object with visualizations.
        """
        boxes = predictions.pred_boxes if predictions.has("pred_boxes") else None
        scores = predictions.scores if predictions.has("scores") else None
        classes = predictions.pred_classes if predictions.has("pred_classes") else None
        labels = _create_text_labels(classes, scores, self.metadata.get("thing_classes", None))
        keypoints = predictions.pred_keypoints if predictions.has("pred_keypoints") else None

        if predictions.has("pred_masks"):
            masks = np.asarray(predictions.pred_masks)
            masks = [GenericMask(x, self.output.height, self.output.width) for x in masks]
        else:
            masks = None

        if self._instance_mode == ColorMode.SEGMENTATION and self.metadata.get("thing_colors"):
            # colors = [
            #     self._jitter([x / 255 for x in self.metadata.thing_colors[c]]) for c in classes
            # ]
            num_instances = len(predictions)
            detected = [
                _DetectedInstance(classes[i], boxes[i], score=scores[i], mask_rle=None, color=None, ttl=8)
                for i in range(num_instances)
            ]
            colors = self._assign_colors(detected)
            alpha = 0.8
        else:
            colors = None
            alpha = 0.5

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(
                (predictions.pred_masks.any(dim=0) > 0).numpy()
            )
            alpha = 0.3

        self.overlay_instances(
            masks=masks,
            boxes=boxes,
            labels=labels,
            keypoints=keypoints,
            assigned_colors=colors,
            alpha=alpha,
        )
        return self.output

    def draw_panoptic_seg_predictions(
            self, panoptic_seg, segments_info, area_threshold=None, alpha=0.7
    ):
        """
        Draw panoptic prediction results on an image.
        Args:
            panoptic_seg (Tensor): of shape (height, width) where the values are ids for each
                segment.
            segments_info (list[dict]): Describe each segment in `panoptic_seg`.
                Each dict contains keys "id", "category_id", "isthing".
            area_threshold (int): stuff segments with less than `area_threshold` are not drawn.
        Returns:
            output (VisImage): image object with visualizations.
        """

        pred = _PanopticPrediction(panoptic_seg, segments_info)

        if self._instance_mode == ColorMode.IMAGE_BW:
            self.output.img = self._create_grayscale_image(pred.non_empty_mask())

        # draw mask for all semantic segments first i.e. "stuff"
        # 这里的pred.semantic_masks()仅仅是notThing的信息
        for mask, sinfo in pred.semantic_masks():
            category_idx = sinfo["category_id"]
            try:
                mask_color = [x / 255 for x in self.metadata.stuff_colors[category_idx]]
            except AttributeError:
                mask_color = None

            text = self.metadata.stuff_classes[category_idx]
            self.draw_binary_mask(
                mask,
                color=mask_color,
                edge_color=_OFF_WHITE,
                text=text,
                alpha=alpha,
                area_threshold=area_threshold,
            )

        # draw mask for all instances second
        all_instances = list(pred.instance_masks())
        if len(all_instances) == 0:
            return self.output
        masks, sinfo = list(zip(*all_instances))
        category_ids = [x["category_id"] for x in sinfo]

        try:
            scores = [x["score"] for x in sinfo]
        except KeyError:
            scores = None
        # print(self.metadata.thing_classes)
        labels = _create_text_labels(category_ids, scores, self.metadata.thing_classes)
        # print(labels)
        # count the current frame obj info
        current_frame_labels = None
        current_frame_info = {}
        if category_ids is not None and self.metadata.thing_classes is not None and len(self.metadata.thing_classes) > 1:
            current_frame_labels = [self.metadata.thing_classes[i] for i in category_ids]
        for obj in current_frame_labels:
            if obj in current_frame_info:
                current_frame_info[obj] += 1
            else:
                current_frame_info[obj] = 1

        try:
            # colors = [random_color(rgb=True, maximum=1) for k in category_ids]
            num_instances = len(masks)
            masks_rles = mask_util.encode(
                np.asarray(np.asarray(masks).transpose(1, 2, 0), dtype=np.uint8, order="F")
            )
            assert len(masks_rles) == num_instances
            detected = [
                _DetectedInstance(category_ids[i], bbox=self.boxes[i], score=scores[i], mask_rle=masks_rles[i],
                                  color=None, ttl=90)
                for i in range(num_instances)
            ]
            colors = self._assign_colors(detected)

        except AttributeError:
            colors = None
        self.overlay_instances(masks=masks, labels=labels, assigned_colors=colors, alpha=alpha)

        labels = [label.split(':')[0] for label in labels]
        warning_content = []
        shape = masks[0].shape
        full_area = shape[0] * shape[1]
        # 计算person的掩码面积占比
        person_id = np.where(np.array(labels) == 'person')
        person_mask = np.array(masks)[person_id]
        areas = np.array([np.sum(mask) / full_area for mask in person_mask])
        accept = np.where(areas >= 0.005)[0]  # 这是个经验值
        if len(accept) >= 1:
            warning = 'Watch out for people, please slow down or stop!'
            warning_content.append(warning)
        else:
            warning_content.append('')

        # 计算car与bus等的掩码面积占比
        # vehicle_id = np.where(np.array(labels) == 'car' or np.array(labels) == 'truck' or np.array(labels) == 'bus')
        # car
        car_id = np.where(np.array(labels) == 'car')
        car_mask = np.array(masks)[car_id]
        areas = np.array([np.sum(mask) / full_area for mask in car_mask])
        accept_car = np.where(areas >= 0.04)[0] # 这是个经验值
        # bus
        bus_id = np.where(np.array(labels) == 'bus')
        bus_mask = np.array(masks)[bus_id]
        areas = np.array([np.sum(mask) / full_area for mask in bus_mask])
        accept_bus = np.where(areas >= 0.04)[0]  # 这是个经验值
        # truck
        truck_id = np.where(np.array(labels) == 'truck')
        truck_mask = np.array(masks)[truck_id]
        areas = np.array([np.sum(mask) / full_area for mask in truck_mask])
        accept_truck = np.where(areas >= 0.04)[0]  # 这是个经验值
        if len(accept_car) + len(accept_bus) + len(accept_truck) >= 1:
            warning = 'Please pay attention to the traffic ahead!'
            warning_content.append(warning)
        else:
            warning_content.append('')
        # print(warning_content)
        return self.output, self.new_instances, self.obj_classes, \
               self._old_instances, current_frame_info, warning_content, pred

    def _assign_colors(self, instances):
        """
        Naive tracking heuristics to assign same color to the same instance,
        will update the internal state of tracked instances.

        Returns:
            list[tuple[float]]: list of colors.
        """

        # Compute iou with either boxes or masks:
        is_crowd = np.zeros((len(instances),), dtype=np.bool)
        # if instances[0].bbox is None:
        assert instances[0].mask_rle is not None
        # use mask iou only when box iou is None because box seems good enough
        rles_old = [x.mask_rle for x in self._old_instances]
        rles_new = [x.mask_rle for x in instances]
        ious = mask_util.iou(rles_old, rles_new, is_crowd)
        threshold = 0.3
        # else:
        #     boxes_old = [x.bbox for x in self._old_instances]
        #     boxes_new = [x.bbox for x in instances]
        #     ious = mask_util.iou(boxes_old, boxes_new, is_crowd)
        #     threshold = 0.6
        if len(ious) == 0:
            ious = np.zeros((len(self._old_instances), len(instances)), dtype="float32")

        # Only allow matching instances of the same label:
        for old_idx, old in enumerate(self._old_instances):
            for new_idx, new in enumerate(instances):
                if old.label != new.label:
                    ious[old_idx, new_idx] = 0

        matched_new_per_old = np.asarray(ious).argmax(axis=1)
        max_iou_per_old = np.asarray(ious).max(axis=1)

        # Try to find match for each old instance:
        extra_instances = []
        for idx, inst in enumerate(self._old_instances):
            if max_iou_per_old[idx] > threshold:
                newidx = matched_new_per_old[idx]
                if instances[newidx].color is None:
                    instances[newidx].color = inst.color
                    continue
            # If an old instance does not match any new instances,
            # keep it for the next frame in case it is just missed by the detector
            inst.ttl -= 1
            if inst.ttl > 0:
                extra_instances.append(inst)

        # new_instances:
        self.new_instances = [instance for instance in instances if instance.color is None]
        # Assign random color to newly-detected instances:
        for inst in instances:
            if inst.color is None:
                inst.color = random_color(rgb=True, maximum=1)

        # update the obj_classes
        # print(len(new_instances))
        for obj in self.new_instances:
            if obj.label in self.obj_classes:
                self.obj_classes[obj.label] += 1
            else:
                self.obj_classes[obj.label] = 1
        self._old_instances = instances[:] + extra_instances

        return [d.color for d in instances]


class ProcessVideo():
    def __init__(self, input_path, output_path, time_interval=10, fps=10):
        """
        input_path:视频的输入路径
        output_path:视频的输出路径
        time_interval:处理过的视频相邻两帧的间隔帧数
        test_need_interval:需要切小图的帧间隔
        fps:帧率，每秒钟帧数越多，所显示的动作就会越流畅
        """
        self.time_interval = time_interval
        self.new_instances = []
        self.old_instances = []
        self.old_boxes = []
        self.old_labels = []
        self.obj_classes = {}

        if args.jobType == 'video':
            self.videoCapture = cv2.VideoCapture(input_path)
            self.size = (int(self.videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                         int(self.videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.frame_num = self.videoCapture.get(cv2.CAP_PROP_FRAME_COUNT)
            print(self.frame_num)
            self.fps = self.videoCapture.get(cv2.CAP_PROP_FPS)  # 视频的流畅度
            self.videoWrite = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), self.fps / self.time_interval, self.size)
        if args.jobType == 'image':
            image = cv2.imread(input_path, 1)
            self.size = (int(image.shape[1]), int(image.shape[0]))
        self.predictor = None
        self.cfg_cut = None

    def _process_video(self):
        """
        将需要切小图的帧存放在outputimageRec文件夹下
        对视频进行全景分割
        切小图
        return：整个视频物体的种类与数目
        """
        self.loadModel()
        success, frame = self.videoCapture.read()
        # 将选定帧图片存放在这个文件夹目录下：
        # frames_save_path = ROOTPATH + 'sourcecode/output/outputimageRec'
        # if not os.path.exists(frames_save_path):
        #   os.mkdir(frames_save_path)
        #         while success:  # 在到达视屏末尾之前
        #             count += 1
        #             if count % self.time_interval == 0:
        #                 image = self.processPerImage(frame)
        # #                 print(image.shape)
        # #                 print(self.size)
        #                 # self.updateClasses(obj_classes)
        #                 self.videoWrite.write(image)  # 写视屏
        # #                 print(self.obj_classes)
        #             if count % self.cut_need_interval == 0:
        #                 print(str(count) + '...')
        #                 # im_encode = cv2.imencode('.jpg', image)[1]
        #                 # im_encode.tofile(frames_save_path + "/%d.jpg" % (count))
        #                 self.proecssCutNeedImage(frame)
        #             success, frame = self.videoCapture.read()  # 读取下一帧
        #         self.videoWrite.release()
        for count in tqdm(range(1, int(self.frame_num + 1))):
            success, frame = self.videoCapture.read()
            if success:
                if count % self.time_interval == 0:
                    image = self.processPerImage(frame, count)
                    # self.updateClasses(obj_classes)
                    self.videoWrite.write(image)  # 写视屏
                # if count % 500 == 0:
                #     self.proecssCutNeedImage(frame)

        self.videoWrite.release()
        class_names = MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).get("thing_classes", None)
        # 将实例和背景分离
        instance = [item for item in self.obj_classes.items() if isinstance(item[0], int)]
        background = [item for item in self.obj_classes.items() if isinstance(item[0], str)]
        # 将各自转换为字典
        instance = dict(zip([item[0] for item in instance], [item[1] for item in instance]))
        background = dict(zip([item[0] for item in background], [item[1] for item in background]))
        classes = [item[0] for item in instance.items()]
        obj_labels = None
        if classes is not None and class_names is not None and len(class_names) > 1:
            obj_labels = [class_names[i] for i in classes]
        instance = dict(zip(obj_labels, instance.values()))
        # 将背景的数量更正：
        background = dict(zip(background.keys(), [1] * len(background)))
        # 合并实例字典和背景字典
        self.obj_classes = dict(instance, **background)

        print(self.obj_classes)
        return self.obj_classes

    def loadModel(self):
        """
        load model
        """
        self.cfg_cut = get_cfg()
        # add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
        self.cfg_cut.merge_from_file(model_zoo.get_config_file("COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml"))
        self.cfg_cut.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model
        # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
        self.cfg_cut.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
            "COCO-PanopticSegmentation/panoptic_fpn_R_101_3x.yaml")
        self.predictor = DefaultPredictor(self.cfg_cut)

    def processPerImage(self, im, count):
        """
        function:panoramic segmentation of incoming image, update totalCount
        return:processed image
        """
        # print("["+"="*int(100*(i+1)/length)+'>'+'-'*(100-int(100*(i+1)/length))+']'+"["+str(100*(i+1)/length)+"%]")
        # write image
        # if i == 0:
        #   # print(outputs)
        #   break
        # print(panoptic_seg, segments_info)
        if args.jobType == 'image':
            self.loadModel()
        outputs = self.predictor(im)
        results = outputs["instances"].to("cpu")
        # outputimageRec_path = ROOTPATH + 'sourcecode/output/outputimageRec/'
        # list_dir = os.listdir(outputimageRec_path)
        panoptic_seg, segments_info = outputs["panoptic_seg"]
        # the obj information of current frame:
        v = Visualizer_rewrite(im[:, :, ::-1], self.old_instances,
                               self.obj_classes, results.pred_boxes.tensor.numpy(),
                               metadata=MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]), scale=1.0)
        v, self.new_instances, self.obj_classes, self.old_instances, current_frame_info, warning_content, pred = \
            v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        masks = results.pred_masks

        img = v.get_image()[:, :, ::-1]
        img = img.copy()
        # update obj_classes:
        # 更新totalCount
        # self.updateClasses(labels)

        """这里需要将实例跟踪"""
        scores = [instance.score for instance in self.new_instances]
        labels = [instance.label for instance in self.new_instances]
        # scores = results.scores if results.has("scores") else None
        # classes = results.pred_classes if results.has("pred_classes") else None
        boxes = [instance.bbox for instance in self.new_instances]
        # boxes = results.pred_boxes if results.has("pred_boxes") else None
        labels = _create_text_labels(labels, scores,
                                     MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).get("thing_classes",
                                                                                             None))
        labels = [label.split(':')[0] for label in labels]
        """
        new_center = [((box[0] + box[2]) / 2., (box[1] + box[3]) / 2.) for box in boxes]
        old_center = [((box[0] + box[2]) / 2., (box[1] + box[3]) / 2.) for box in self.old_boxes]
        range = [(cen[0] - 1. * self.time_interval, cen[1] - 1. * self.time_interval,
                  cen[0] + 1. * self.time_interval, cen[1] + 1. * self.time_interval) for cen in old_center]
        # 当前图片包含的对象的id
        now_id = [i for i, clss in enumerate(classes)]
        # 需要切割以及计数出来的id
        cut_id = []
        for i, cen in enumerate(new_center):
            notcut = False
            # 同类label
            temp_old_label = [(j, label) for j, label in enumerate(self.old_labels) if label == labels[i]]
            # 同类的id
            id_lyst = [tl[0] for tl in temp_old_label]
            # 同类中心生成的范围
            range = np.array(range)
            temp_range = list(range[id_lyst])
            # temp_range = [(k, r) for k, r in enumerate(range) if k in id_lyst]
            for _range in temp_range:
                # cen在某一个范围内：
                if _range[0] <= cen[0] <= _range[2] and _range[1] <= cen[1] <= _range[3]:
                    notcut = True
                    break
            if notcut:
                continue
            cut_id.append(i)
        self.old_labels = labels
        labels = np.array(labels)
        labels = list(labels[cut_id])
        # print(type(boxes))
        self.old_boxes = boxes
        boxes = boxes[cut_id]
        scores = np.array(scores)
        scores = list(scores[cut_id])
        """

        """为图片添加检测出来的物体类别以及数量文本"""
        # 为处理过后的图片添加统计数量
        #
        # 首先将背景物体检测并计数：
        text = [MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).stuff_classes[sinfo["category_id"]]
                for mask, sinfo in pred.semantic_masks()]

        back_ground = {}
        for item in text:
            if item in back_ground:
                back_ground[item] += 1
            else:
                back_ground[item] = 1

        # put not-instances text:
        totalClasses = len(back_ground) + len(current_frame_info)
        text_start_place = 50

        # put totalClasses:
        content = "Total count classes: %d" % totalClasses
        img = cv2.putText(img, content, (50, text_start_place),
                          cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # put classes and num:
        for key, value in back_ground.items():
            text_start_place += 40
            content = '%s: %d' % (key, value)
            if text_start_place >= (self.size[1] - 50):
                break
            img = cv2.putText(img, content, (50, text_start_place),
                              cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # put instances text:
        #
        # put classes and num:
        for key, value in current_frame_info.items():
            text_start_place += 40
            content = '%s: %d' % (key, value)
            if text_start_place >= (self.size[1] - 50):
                break
            img = cv2.putText(img, content, (50, text_start_place),
                              cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        """fix bugs
        img_labels = category_ids
        box_labels = results.pred_classes.tolist()
        if img_labels != box_labels:

            print(count)
            print(img_labels)
            print(box_labels)
            
            box和current_img的类别列表不同"""

        # 将背景放入总体的实例中：
        # (注意如果self.obj_classes已有和back_ground一样的key的item，
        # 将会用back_ground中的item的value覆盖self.obj_classes)
        #
        # 因为背景的数量不重要，所以无影响
        self.obj_classes = dict(self.obj_classes, **back_ground)

        """为图片添加道路情况文本"""
        img = cv2.putText(img, warning_content[0], (int(img.shape[0] / 2), 90),
                              cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        img = cv2.putText(img, warning_content[1], (int(img.shape[0] / 2), 130),
                          cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
        """添加底部文本"""
        bottom = 'You can see the tracking effect from the color rather than ID!'
        img = cv2.putText(img, bottom, (50, self.size[1] - 50),
                              cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

        # 切小图
        """这里切割小图，只切割notThing"""
#         if count % 1000 == 0:
#             obj = CutImage(metadata=MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]),
#                            panoptic_seg=panoptic_seg, segments_info=segments_info,
#                            image=im, isThing=False, isNotThing=True)
#             obj.main()
        """这里切割小图，只切Thing"""
#         box_labels = results.pred_classes.tolist()
#         obj = CutImage(boxes=boxes, scores=scores, labels=labels,
#                        metadata=MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]),
#                        image=im, isThing=True, isNotThing=False, box_labels=box_labels)
#         obj.main()
        # update notThing classes:
        # obj.image = im
        """这里需要改，不能将树直接加过来.NotThing数量如何处理之后加进来"""
        # pred = _PanopticPrediction(panoptic_seg, segments_info)
        # text = [MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).stuff_classes[sinfo["category_id"]] for mask, sinfo in pred.semantic_masks()]
        # self.updateClasses(text)
        return img

    def proecssCutNeedImage(self, im):
        """
        function:cut notThing obj only
        return:None
        """
        # outputimageRec_path = ROOTPATH + 'sourcecode/output/outputimageRec/'
        # list_dir = os.listdir(outputimageRec_path)
        # print(list_dir)

        # im = cv2.imread(outputimageRec_path + img)
        # print('processing ' + img + '...')
        outputs = self.predictor(im)
        panoptic_seg, segments_info = self.predictor(im)["panoptic_seg"]
        # print(panoptic_seg, segments_info)

        # results = outputs["instances"].to("cpu")
        # boxes = results.pred_boxes if results.has("pred_boxes") else None
        # scores = results.scores if results.has("scores") else None
        # classes = results.pred_classes if results.has("pred_classes") else None
        # labels = self._create_text_labels(classes, scores, MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).get("thing_classes", None))
        # labels = list(map(lambda x: x.split(' ')[0], labels))

        # ColorMode.IMAGE, instance_mode=ColorMode.SEGMENTATION, ColorMode.IMAGE_BW
        # v = Visualizer(im[:, :, ::-1], MetadataCatalog.get(cfg_cut.DATASETS.TRAIN[0]), scale=1.0)
        # v = v.draw_panoptic_seg_predictions(panoptic_seg.to("cpu"), segments_info)
        # cv2_imshow(v.get_image()[:, :, ::-1])
        pred = _PanopticPrediction(panoptic_seg, segments_info)
        text = [MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]).stuff_classes[sinfo["category_id"]] for mask, sinfo
                in pred.semantic_masks()]
        self.updateClasses(text)
        """这里切割小图，只切割notThing"""
        obj = CutImage(metadata=MetadataCatalog.get(self.cfg_cut.DATASETS.TRAIN[0]),
                       panoptic_seg=panoptic_seg, segments_info=segments_info,
                       image=im, isThing=False, isNotThing=True)

        obj.main()

    def updateClasses(self, labels):
        """更新obj_classes"""
        for label in labels:
            if label in self.obj_classes:
                self.obj_classes[label] += 1
            else:
                self.obj_classes[label] = 1


if __name__ == '__main__':
    # frames_save_path = ROOTPATH + 'sourcecode/input_video/processed_image_path'
    # if os.path.exists(frames_save_path):
    #    shutil.rmtree(frames_save_path)
    import time
    args = get_args()
    time1 = time.time()
    if os.path.exists(args.path_imageSeg):
        shutil.rmtree(args.path_imageSeg)
    if os.path.exists(args.path_outputVideo):
        os.remove(args.path_outputVideo)
    if os.path.exists(args.path_totalCount):
        os.remove(args.path_totalCount)
    input_path = args.path_inputVideo
    print(input_path)
    output_path = args.path_outputVideo
    print(output_path)
    if args.jobType == 'image':
        img = cv2.imread(input_path, 1)
        obj = ProcessVideo(input_path, output_path, time_interval=args.time_interval)
        result = obj.processPerImage(img, 0)
        cv2.imwrite(args.path_outputVideo, result)
    elif args.jobType == 'video':
        obj = ProcessVideo(input_path, output_path, time_interval=args.time_interval)
        obj_class = obj._process_video()
        time2 = time.time()
        print('time2 - time1 = ', str(time2 - time1))
        # 将字典的数据写入txt文件保存：
        for key, value in obj_class.items():
            with open(args.path_totalCount, 'a') as f:
                if isinstance(value, str):
                    f.write(key + ': ' + value)
                    f.write('\n')
                else:
                    f.write(key + ': ' + str(value))
                    f.write('\n')
    else:
        print('wrong type!')
        exit()
    print('done!')


# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('-v', '--verbose', action='store_true', choices = [0, 1, 2], help='the usage of verbose')
# parser.add_argument('-q', action='count', help='the usage of count')
# parser.add_argument('square', type=int, help='the usage of square')
#
# args = parser.parse_args()
# answer = args.square ** 2
# if args.verbose:
#     print("the square of {} equals {}".format(args.square, answer))
# else:
#     print(answer)