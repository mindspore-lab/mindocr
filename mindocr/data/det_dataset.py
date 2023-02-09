
class DetAug():

class DetDataset():
    """Dataset iterator for detection task, including ICDAR15. 
    
    Annotation format:
        Image file name\tImage annotation information encoded by json.dumps
        ch4_test_images/img_61.jpg\t[{"transcription": "MASA", "points": [[310, 104], [416, 141], [418, 216], [312, 179]]}, {...}]   


    """
    def __init__(self, dataset_config: dict, transform_config :dict=None, is_train=True):
        


        self.RGB_MEAN = np.array([122.67891434, 116.66876762, 104.00698793])
        self.config = config
        self.isTrain = isTrain

        self.ra = RandomAugment(max_tries=config.dataset.random_crop.max_tries,
                                min_crop_side_ratio=config.dataset.random_crop.min_crop_side_ratio)
        self.ms = MakeSegDetectionData(config.train.min_text_size,
                                       config.train.shrink_ratio)
        self.mb = MakeBorderMap(config.train.shrink_ratio,
                                config.train.thresh_min, config.train.thresh_max)

        if isTrain:
            img_paths = glob.glob(os.path.join(config.train.img_dir,
                                               '*' + config.train.img_format))
        else:
            img_paths = glob.glob(os.path.join(config.eval.img_dir,
                                               '*' + config.eval.img_format))

        if self.isTrain:
            img_dir = config.train.gt_dir
            if config.dataset.is_icdar2015:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.jpg.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]
        else:
            img_dir = config.eval.gt_dir
            if config.dataset.is_icdar2015:
                gt_paths = [os.path.join(img_dir, 'gt_' + img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]
            else:
                gt_paths = [os.path.join(img_dir, img_path.split('/')[-1].split('.')[0] + '.txt')
                            for img_path in img_paths]

        self.img_paths = img_paths
        self.gt_paths = gt_paths

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        gt_path = self.gt_paths[index]

        # Getting
        img = get_img(img_path)
        original = resize(img)
        polys, dontcare = get_bboxes(gt_path, self.config)

        # Random Augment
        if self.isTrain and self.config.train.is_transform:
            img, polys = self.ra.random_scale(img, polys, self.config.dataset.short_side)
            img, polys = self.ra.random_rotate(img, polys, self.config.dataset.random_angle)
            img, polys = self.ra.random_flip(img, polys)
            img, polys, dontcare = self.ra.random_crop(img, polys, dontcare)
        else:
            polys = polys.reshape((polys.shape[0], polys.shape[1] // 2, 2))
        img, polys = resize(img, polys, isTrain=self.isTrain)

        # Post Process
        if self.isTrain:
            img, gt, gt_mask = self.ms.process(img, polys, dontcare)
            img, thresh_map, thresh_mask = self.mb.process(img, polys, dontcare)
        else:
            polys = np.array(polys)
            dontcare = np.array(dontcare, dtype=np.bool8)
            img, polys = scale_pad(img, polys, self.config.eval.eval_size)

        # Show Images
        if self.config.dataset.is_show:
            cv2.imwrite('./images/img.jpg', img)
            cv2.imwrite('./images/gt.jpg', gt[0]*255)
            cv2.imwrite('./images/gt_mask.jpg', gt_mask*255)
            cv2.imwrite('./images/thresh_map.jpg', thresh_map*255)
            cv2.imwrite('./images/thresh_mask.jpg', thresh_mask*255)

        # Random Colorize
        if self.isTrain and self.config.train.is_transform:
            colorjitter = RandomColorAdjust(brightness=32.0 / 255, saturation=0.5)
            img = colorjitter(ToPIL()(img.astype(np.uint8)))

        # Normalize
        img -= self.RGB_MEAN
        img = ToTensor()(img)

        if self.isTrain:
            return img, gt, gt_mask, thresh_map, thresh_mask
        return original, img, polys, dontcare


