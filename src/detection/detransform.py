import albumentations as alb

class DetrTransform:

    def __init__(self, use_transform: str = 'transform_1'):
        self.use_transform = use_transform

    def train_transform(self):
        if self.use_transform == 'transform_1':
            transform = self.train_transform_1()
        else:
            raise ValueError('Invalid transform name')
        return transform
    def val_transform(self):
        return self.val_transform_1()


    def train_transform_1(self):
        transform = alb.Compose([
            alb.Affine(scale=(0.8, 1.2),
                       rotate=1.0),
            alb.Perspective(p=0.1),
            alb.RandomBrightnessContrast(),
            alb.HueSaturationValue(p=0.1),
            alb.Blur(),
            alb.RandomGamma(),
            alb.Sharpen(),
            alb.CoarseDropout(num_holes_range=(1, 32),
                              hole_height_range=(4, 32),
                              hole_width_range=(4, 32)),
            alb.CLAHE()],
            bbox_params=alb.BboxParams(format='coco',
                                       label_fields=['category'],
                                       clip=True,
                                       min_area=1))
        return transform


    def val_transform_1(self):
        transform = alb.Compose([alb.NoOp()],
            bbox_params=alb.BboxParams(format='coco',
                                       label_fields=['category'],
                                       clip=True,
                                       min_area=1))
        return transform