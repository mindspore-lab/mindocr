import cv2

__all__ = ["DecodeImage"]


class DecodeImage:
    """
    adapted to DecodeImage
    real DecodeImage has been moved to decode_node in parallel mode
    """

    def __init__(self, img_mode="BGR", channel_first=False, to_float32=False, **kwargs):
        self.img_mode = img_mode
        self.to_float32 = to_float32
        self.channel_first = channel_first

    def __call__(self, data):
        data["image"] = self._decode(data["image"])
        return data

    def _decode(self, img):
        if self.img_mode == "RGB":
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.channel_first:
            img = img.transpose((2, 0, 1))

        if self.to_float32:
            img = img.astype("float32")

        return img
