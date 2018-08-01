# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import shutil
from glob import glob

PURGE_SAFE = '/home/kits-adm/Datasets/flickr_epa/faces/'


# borrowed functions, erase when merge
def _log_one_line(file_loc, line):
    """Log one line to file.
    Note: the operation is expensive but more safe, always close file right after.

    :param file_loc: location of file write to.
    :param line: a line need to write to log.
    :return: None.
    """
    with open(file_loc, 'a+') as log_file:
        log_file.write(line + '\n')


def _purge(path, check_type, rebuild=True):
    """Purge a path and ensure safety.

    :param path: path intend to purge.
    :param check_type: one of 'file' or 'path'. Ensure path type.
    :param rebuild: bool, default True. Rebuild an empty file or folder.
    :return:
    """
    # https://stackoverflow.com/questions/3812849
    if path is None:
        return  # nothing to purge
    if os.path.commonpath([os.path.abspath(PURGE_SAFE)]) != \
            os.path.commonpath([os.path.abspath(PURGE_SAFE), os.path.abspath(path)]):
        raise Exception('Path is not safe to purge: {}'.format(path))
    if os.path.exists(path):
        if check_type == 'file':
            assert(os.path.isfile(path))
            os.remove(path)
        elif check_type == 'path':  # path
            os.path.isdir(path)
            shutil.rmtree(path)
        else:
            raise Exception('check_type has to be file or path, got: {}'.format(check_type))
    if rebuild:
        if check_type == 'file':
            open(path, 'a').close()
        if check_type == 'path':
            os.mkdir(path)
    return  # no need to purge


def big_ass_warning(line):
    print('------------------------------------------------------------------------------')
    print('-- {} --'.format(line))
    print('______________________________________________________________________________')


class Filter:
    def __init__(self, img_path, log_path, meta_path, purges=True):
        self.detector = MtcnnDetector(model_folder='model', ctx=mx.gpu(0), num_worker=4, accurate_landmark=False)
        self.img_path = img_path
        self.success_logs_path = os.path.join(log_path, 'successes/')
        self.folders = [x for x in glob(img_path + "*")]
        self.log_path = log_path
        self.file_log = os.path.join(log_path, 'lists/')
        data_log = os.path.join(log_path, 'bboxes/')
        if purges:
            _purge(data_log, 'path')
        self.b_log = os.path.join(data_log, 'b/')  # bbox
        self.p_log = os.path.join(data_log, 'p/')  # facial points
        self.save_path = os.path.join(log_path, 'imgs/')
        # meta
        self.meta_path = meta_path
        self.meta_records = os.path.join(meta_path, 'records/')
        self.meta_rois = os.path.join(meta_path, 'rois/')
        if os.path.isdir(self.meta_records) and os.path.isdir(self.meta_rois): # validate file structure
            pass
        else:
            raise Exception('meta record broken.')
        # reset logs
        if purges:
            _purge(self.save_path, 'path')
            _purge(self.file_log, 'path')
            _purge(self.b_log, 'path')
            _purge(self.p_log, 'path')

    def _intersect_ratio(self, box, rois):
        """Calculate max intersect ratio between a box and list of candidate, ratio relative to the box.
        E.g: area(overlap) / area(box)
        Note: https://stackoverflow.com/questions/27152904

        :param box: tuple, format: x, y, w, h.
        :param rois: list, a list of rois. Same format as box.
        :return: float, a number between 0 and 1.
        """
        max_ = 0
        box_area = (box[2] - box[0]) * (box[3] - box[1])
        if box_area == 0:
            return 0
        for roi in rois:
            dx = min(roi[2], box[2]) - max(roi[0], box[0])
            dy = min(roi[3], box[3]) - max(roi[1], box[1])
            if (dx >= 0) and (dy >= 0):
                max_ = max(max_, ((dx*dy) / box_area))
        if max_ < 0 or max_ > 1:
            raise Exception('Implementation or data error, max out of bound: {}'.format(max_))
        return max_

    def _detect_img(self, img_path, img_id, save_paths,
                    file_log, b_log, p_log, rejection_log,
                    rois,
                    min_confidence=0.9, min_resolution=24):
        print('Detecting:', img_path)
        img = cv2.imread(img_path)
        # run detector
        results = self.detector.detect_face(img)
        # filter out low resolution, increase next step CNN accuracy and decrease this step false positive
        total_boxes = []
        points = []
        if results is not None:
            for i, b in enumerate(results[0]):
                # check resolution, confidence, and intersect ratio to rcnn meta.
                if (b[2]-b[0] >= min_resolution) and (b[3]-b[1] >= min_resolution) and \
                        (b[4] >= min_confidence) and \
                        (self._intersect_ratio(b, rois) >= 0.7):
                    total_boxes.append(b)
                    points.append(results[1][i])
                else:  # rejected by filter
                    # write format: img_id, x1, y1, x2, y2, confidence.
                    _log_one_line(rejection_log, '{} {} {} {} {} {}'.format(img_id,
                                                                            b[0], b[1], b[2], b[3],
                                                                            b[4]))
        if (results is None) or (len(total_boxes) == 0):
            _log_one_line(file_log, '{} 0'.format(img_id))
            return

        # extract aligned face chips
        chips = self.detector.extract_image_chips(img, points, 144, 0.37)
        face_num = len(chips)
        assert(len(save_paths) == 3)
        if face_num == 1:
            save_path = save_paths[0]
        elif face_num == 2:
            save_path = save_paths[1]
        else:
            save_path = save_paths[2]
        for ind, chip in enumerate(chips):
            cv2.imwrite(os.path.join(save_path, '{}_{}.jpg'.format(img_id, ind)), chip)
        _log_one_line(file_log, '{} {}'.format(img_id, len(chips)))
        # write boxes
        for ind, b in enumerate(total_boxes):
            # write format: img_id, x1, y1, x2, y2, confidence.
            _log_one_line(b_log, '{} {} {} {} {} {} {}'.format(img_id, ind,
                                                               b[0], b[1], b[2], b[3],
                                                               b[4]))
        for ind, p in enumerate(points):
            for i in range(5):
                # write format: img_id
                _log_one_line(p_log, '{} {} {} {} {}'.format(img_id, ind, i, p[i], p[i+5]))

    def write_all(self):
        # some extra logging info
        _total = len(self.folders)
        _count = 0
        for folder in self.folders:
            category_id = folder.split('/')[-1]
            big_ass_warning('CATEGORY: {}, PROGRESS: {:.2f}%'.format(category_id, float(_count)*100/_total))
            # load meta
            _meta_path = os.path.join(self.meta_rois, '{}.txt'.format(category_id))
            if not os.path.isfile(_meta_path):  # meta may not be there due to no person in category.
                continue
            with open(_meta_path) as f:
                _lines = f.readlines()
            records = {}  # meta key: image_id, value: roi
            for line in _lines:
                _raw = line.strip().split()
                assert(len(_raw) == 6)
                _img_id = _raw[0]
                roi = [int(_raw[2]),
                       int(_raw[3]),
                       int(_raw[4]) + int(_raw[2]),
                       int(_raw[5]) + int(_raw[3]), ]  # x1, y1, x2, y2
                if _img_id in records:
                    records[_img_id].append(roi)
                else:
                    records[_img_id] = [roi]
            # build log system
            file_log = os.path.join(self.file_log, '{}.txt'.format(category_id))
            _purge(file_log, 'file')
            c_b_log = os.path.join(self.b_log, '{}.txt'.format(category_id))
            _purge(c_b_log, 'file')
            c_p_log = os.path.join(self.p_log, '{}.txt'.format(category_id))
            _purge(c_p_log, 'file')
            c_rj_log = os.path.join(self.b_log, 'r{}.txt'.format(category_id))
            _purge(c_rj_log, 'file')
            save_path = os.path.join(self.save_path, '{}/'.format(category_id))
            _purge(save_path, 'path')
            save_paths = []
            for i in range(3):  # three tiers for ppl number.
                p = os.path.join(save_path, '{}/'.format((i+1),))
                _purge(p, 'path')
                save_paths.append(p)
            # write data
            for file in os.listdir(folder):
                if file.endswith(".jpg"):
                    _path = os.path.join(folder, file)
                    img_id = str(int(file.split('/')[-1].split('.')[0])).zfill(6)
                    if not (img_id in records):  # zero person in image, based on meta data
                        continue
                    rois = records[img_id]
                    self._detect_img(img_path=_path,
                                     img_id=img_id,
                                     save_paths=save_paths,
                                     file_log=file_log,
                                     b_log=c_b_log,
                                     p_log=c_p_log,
                                     rejection_log=c_rj_log,
                                     rois=rois)
            _count += 1


if __name__ == "__main__":
    f = Filter(img_path='/home/kits-adm/Datasets/flickr_epa/pics/',
               meta_path='/home/kits-adm/Datasets/flickr_epa/filters/',
               log_path=PURGE_SAFE)
    f.write_all()
