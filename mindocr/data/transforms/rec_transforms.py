'''
transform for text recognition tasks.
'''
from typing import Any, Optional, Dict, List
import cv2
import math
import numpy as np

__all__ = ['RecCTCLabelEncode', 'RecAttnLabelEncode', 'RecResizeImg', 'RecResizeNormForInfer', 'SVTRRecResizeImg']


class RecCTCLabelEncode(object):
    ''' Convert text label (str) to a sequence of character indices according to the char dictionary

    Args:
        max_text_len: to pad the label text to a fixed length (max_text_len) of text for ctc loss computate.
        character_dict_path: path to dictionary, if None, a dictionary containing 36 chars (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
        use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
        blank_at_last(bool): padding with blank index (not the space index). If True, a blank/padding token will be appended to the end of the dictionary, so that blank_index = num_chars, where num_chars is the number of character in the dictionary including space char if used. If False, blank token will be inserted in the beginning of the dictionary, so blank_index=0.
        lower (bool): if True, all upper-case chars in the label text will be converted to lower case. Set to be True if dictionary only contains lower-case chars. Set to be False if not and want to recognition both upper-case and lower-case.

    Attributes:
        blank_idx: the index of the blank token for padding
        num_valid_chars: the number of valid characters (including space char if used) in the dictionary
        num_classes: the number of classes (which valid characters char and the speical token for blank padding). so num_classes = num_valid_chars + 1


    '''
    def __init__(self,
                max_text_len=23,
                character_dict_path=None,
                use_space_char=False,
                blank_at_last=True,
                lower=False,
                **kwargs,
                #start_token='<BOS>',
                #end_token='<EOS>',
                #unkown_token='',
                ):
        self.max_text_len = max_text_len
        self.space_idx = None
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = [c for c in  "0123456789abcdefghijklmnopqrstuvwxyz"]

            self.lower = True
            #print("INFO: The character_dict_path is None, model can only recognize number and lower letters")
        else:
            # TODO: this is commonly used in other modules, wrap into a func or class.
            # parse char dictionary
            char_list = []
            with open(character_dict_path, 'r') as f:
                for line in f:
                    c = line.rstrip('\n\r')
                    char_list.append(c)
        # add space char if set
        if use_space_char:
            if ' ' not in char_list:
                char_list.append(' ')
            self.space_idx = len(char_list) - 1
        else:
            if ' ' in char_list:
                print("WARNING: The dict still contains space char in dict although use_space_char is set to be False, because the space char is coded in the dictionary file ", character_dict_path)

        self.num_valid_chars = len(char_list) # the number of valid chars (including space char if used)

        # add blank token for padding
        if blank_at_last:
            # the index of a char in dict is [0, num_chars-1], blank index is set to num_chars
            char_list.append('<PAD>')
            self.blank_idx = self.num_valid_chars
        else:
            char_list = ['<PAD>'] + char_list
            self.blank_idx = 0

        self.dict = {c:idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: dict):
        '''
        required keys:
            label -> (str) text string
        added keys:
            text_seq-> (np.ndarray, int32), sequence of character indices after  padding to max_text_len in shape (sequence_len), where ood characters are skipped
        added keys:
            length -> (np.int32) the number of valid chars in the encoded char index sequence,  where valid means the char is in dictionary.
            text_padded -> (str) text label padded to fixed length, to solved the dynamic shape issue in dataloader.
            text_length -> int, the length of original text string label

        '''
        char_indices = str2idx(data['label'], self.dict, max_text_len=self.max_text_len, lower=self.lower)

        if char_indices is None:
            char_indices = []
            #return None
        data['length'] = np.array(len(char_indices), dtype=np.int32)
        # padding with blank index
        char_indices = char_indices + [self.blank_idx] * (self.max_text_len - len(char_indices))
        # TODO: raname to char_indices
        data['text_seq'] = np.array(char_indices, dtype=np.int32)
        #
        data['text_length'] = len(data['label'])
        data['text_padded'] = data['label'] + ' ' * (self.max_text_len - len(data['label']))

        return data


class RecAttnLabelEncode:
    def __init__(self,
                 max_text_len: int = 25,
                 character_dict_path: Optional[str] = None,
                 use_space_char: bool = False,
                 lower: bool = False,
                 **kwargs,
    ) -> None:
        """
        Convert text label (str) to a sequence of character indices according to the char dictionary

        Args:
            max_text_len: to pad the label text to a fixed length (max_text_len) of text for attn loss computate.
            character_dict_path: path to dictionary, if None, a dictionary containing 36 chars (i.e., "0123456789abcdefghijklmnopqrstuvwxyz") will be used.
            use_space_char(bool): if True, add space char to the dict to recognize the space in between two words
            lower (bool): if True, all upper-case chars in the label text will be converted to lower case. Set to be True if dictionary only contains lower-case chars. Set to be False if not and want to recognition both upper-case and lower-case.

        Attributes:
            go_idx: the index of the GO token
            stop_idx: the index of the STOP token
            num_valid_chars: the number of valid characters (including space char if used) in the dictionary
            num_classes: the number of classes (which valid characters char and the speical token for blank padding). so num_classes = num_valid_chars + 1
        """
        self.max_text_len = max_text_len
        self.lower = lower

        # read dict
        if character_dict_path is None:
            char_list = list("0123456789abcdefghijklmnopqrstuvwxyz")

            self.lower = True
            print("INFO: The character_dict_path is None, model can only recognize number and lower letters")
        else:
            # parse char dictionary
            char_list = []
            with open(character_dict_path, 'r') as f:
                for line in f:
                    c = line.rstrip('\n\r')
                    char_list.append(c)

        # add space char if set
        if use_space_char:
            if ' ' not in char_list:
                char_list.append(' ')
            self.space_idx = len(char_list) + 1
        else:
            if ' ' in char_list:
                print("WARNING: The dict still contains space char in dict although use_space_char is set to be False, because the space char is coded in the dictionary file ", character_dict_path)

        self.num_valid_chars = len(char_list) # the number of valid chars (including space char if used)

        special_token = ['<GO>', '<STOP>']
        char_list = special_token + char_list

        self.go_idx = 0
        self.stop_idx = 1

        self.dict = {c:idx for idx, c in enumerate(char_list)}

        self.num_classes = len(self.dict)

    def __call__(self, data: Dict[str, Any] ) -> str:
        char_indices = str2idx(data['label'], self.dict, max_text_len=self.max_text_len, lower=self.lower)

        if char_indices is None:
            char_indices = []
        data['length'] = np.array(len(char_indices), dtype=np.int32)

        char_indices = [self.go_idx] + char_indices + [self.stop_idx] + [self.go_idx] * (self.max_text_len - len(char_indices))
        data['text_seq'] = np.array(char_indices, dtype=np.int32)

        data['text_length'] = len(data['label'])
        data['text_padded'] = data['label'] + ' ' * (self.max_text_len - len(data['label']))
        return data


def str2idx(text: str, label_dict: Dict[str, int], max_text_len: int = 23, lower: bool = False) -> List[int]:
    '''
    Encode text (string) to a squence of char indices
    Args:
        text (str): text string
    Returns:
        char_indices (List[int]): char index seq
    '''
    if len(text) == 0 or len(text) > max_text_len:
        return None
    if lower:
        text = text.lower()

    char_indices = []
    # TODO: for char not in the dictionary, skipping may lead to None data. Use a char replacement? refer to mmocr
    for char in text:
        if char not in label_dict:
            #print('WARNING: {} is not in dict'.format(char))
            continue
        char_indices.append(label_dict[char])
    if len(char_indices) == 0:
        print('WARNING: {} doesnot contain any valid char in the dict'.format(text))
        return None

    return char_indices

# TODO: reorganize the code for different resize transformation in rec task
def resize_norm_img(img,
                    image_shape,
                    padding=True,
                    interpolation=cv2.INTER_LINEAR):
    '''
    resize image
    Args:
        img: shape (H, W, C)
        image_shape: image shape after resize, in (C, H, W)
        padding: if Ture, resize while preserving the H/W ratio, then pad the blank.

    '''
    imgH, imgW = image_shape
    h = img.shape[0]
    w = img.shape[1]
    c = img.shape[2]
    if not padding:
        resized_image = cv2.resize(
            img, (imgW, imgH), interpolation=interpolation)
        resized_w = imgW
    else:
        ratio = w / float(h)
        if math.ceil(imgH * ratio) > imgW:
            resized_w = imgW
        else:
            resized_w = int(math.ceil(imgH * ratio))
        resized_image = cv2.resize(img, (resized_w, imgH))
    '''
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    '''
    padding_im = np.zeros((imgH, imgW, c), dtype=np.uint8)
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio

# TODO: check diff from resize_norm_img
def resize_norm_img_chinese(img, image_shape):
    ''' adopted from paddle
    '''
    imgH, imgW = image_shape
    # todo: change to 0 and modified image shape
    max_wh_ratio = imgW * 1.0 / imgH
    h, w = img.shape[0], img.shape[1]
    c = img.shape[2]
    ratio = w * 1.0 / h
    max_wh_ratio = min(max(max_wh_ratio, ratio), max_wh_ratio)
    imgW = int(imgH * max_wh_ratio)
    if math.ceil(imgH * ratio) > imgW:
        resized_w = imgW
    else:
        resized_w = int(math.ceil(imgH * ratio))
    resized_image = cv2.resize(img, (resized_w, imgH))
    '''
    resized_image = resized_image.astype('float32')
    if image_shape[0] == 1:
        resized_image = resized_image / 255
        resized_image = resized_image[np.newaxis, :]
    else:
        resized_image = resized_image.transpose((2, 0, 1)) / 255
    resized_image -= 0.5
    resized_image /= 0.5
    '''
    #padding_im = np.zeros((imgC, imgH, imgW), dtype=np.float32)
    padding_im = np.zeros((imgH, imgW, c), dtype=np.uint8)
    #padding_im[:, :, 0:resized_w] = resized_image
    padding_im[:, 0:resized_w, :] = resized_image
    valid_ratio = min(1.0, float(resized_w / imgW))
    return padding_im, valid_ratio

# TODO: remove infer_mode and character_dict_path if they are not necesary
class RecResizeImg(object):
    ''' adopted from paddle
    resize, convert from hwc to chw, rescale pixel value to -1 to 1
    '''
    def __init__(self,
                 image_shape,
                 infer_mode=False,
                 character_dict_path=None,
                 padding=True,
                 **kwargs):
        self.image_shape = image_shape
        self.infer_mode = infer_mode
        self.character_dict_path = character_dict_path
        self.padding = padding

    def __call__(self, data):
        img = data['image']
        if self.infer_mode and self.character_dict_path is not None:
            norm_img, valid_ratio = resize_norm_img_chinese(img,
                                                            self.image_shape)
        else:
            norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                    self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        # TODO: data['shape_list'] = ?
        return data


class SVTRRecResizeImg(object):
    def __init__(self, image_shape, padding=True, **kwargs):
        self.image_shape = image_shape
        self.padding = padding

    def __call__(self, data):
        img = data['image']

        norm_img, valid_ratio = resize_norm_img(img, self.image_shape,
                                                self.padding)
        data['image'] = norm_img
        data['valid_ratio'] = valid_ratio
        return data


class RecResizeNormForInfer(object):
    '''
    Resize image for text recognition

    Args:
        target_height: target height after resize. Commonly, 32 for crnn, 48 for svtr. default is 32.
        target_width: target width. Default is 320. If None, image width is scaled to make aspect ratio unchanged.
        keep_ratio: keep aspect ratio. If True, resize the image with ratio=target_height / input_height (certain image height is required by recognition network).
                    If False, simply resize to targte size (`target_height`, `target_width`)
        padding: If True, pad the resized image to the targte size with zero RGB values. only used when `keep_ratio` is True.

    Notes:
        1. The default choice (keep_ratio, not padding) is suitable for inference for better accuracy.
    '''
    def __init__(self, target_height=32, target_width=320, keep_ratio=True, padding=False, interpolation=cv2.INTER_LINEAR, norm_before_pad=False,  mean=[127.0, 127.0, 127.0], std=[127.0, 127.0, 127.0], **kwargs):
        self.keep_ratio = keep_ratio
        self.padding = padding
        #self.targt_shape = target_shape
        self.tar_h = target_height
        self.tar_w = target_width
        self.interpolation = interpolation
        self.norm_before_pad = norm_before_pad
        self.mean = np.array(mean, dtype="float32") 
        self.std = np.array(std, dtype="float32")

    def norm(self, img):
        return (img - self.mean) / self.std

    def __call__(self, data):
        '''
        data: image in shape [h, w, c]
        '''
        img = data['image']
        h, w = img.shape[:2]
        #tar_h, tar_w = self.targt_shape
        resize_h = self.tar_h
        
        max_wh_ratio = self.tar_w / float(self.tar_h)      

        if self.keep_ratio==False:
            assert self.tar_w is not None, 'Must specify target_width if keep_ratio is False'
            resize_w = self.tar_w #if self.tar_w is not None else resized_h * self.max_wh_ratio
        else:
            src_wh_ratio = w / float(h)
            resize_w = int(min(src_wh_ratio, max_wh_ratio) * resize_h)
        #print('Rec resize: ', h, w, "->", resize_h, resize_w)

        resized_img = cv2.resize(img, (resize_w, resize_h), interpolation=self.interpolation)

        # TODO: norm before padding

        data['shape_list'] = [h, w, resize_h / h, resize_w / w] # TODO: reformat, currently align to det
        if self.norm_before_pad:
            resized_img = self.norm(resized_img) 

        if self.padding and self.keep_ratio:
            padded_img = np.zeros((self.tar_h, self.tar_w, 3), dtype=resized_img.dtype)
            padded_img[:, :resize_w, :] = resized_img
            data['image'] = padded_img
        else:
            data['image'] = resized_img

        if not self.norm_before_pad:
            data['image'] = self.norm(data['image']) 

        return data


if __name__ == '__main__':
    text = '012 ab%c'

    # test dict and ctc label encode
    trans = RecCTCLabelEncode(10, use_space_char=False)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    gt = np.array([0, 1, 2, 10, 11 ,12] + [trans.blank_idx]*4)
    assert trans.num_valid_chars==36
    assert trans.num_classes==37
    assert out['length'] ==len(text)-2, 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=Flase, space and OOV char excluded
    assert np.array_equal(seq, gt)

    # test dict and attn label encode
    trans = RecAttnLabelEncode(10, use_space_char=False)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    gt = np.array([trans.go_idx] + [2, 3, 4, 12, 13, 14] + [trans.stop_idx] + [trans.go_idx]*2)
    assert trans.num_valid_chars==36
    assert trans.num_classes==38
    assert out['length'] == len(text) - 2, 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=Flase, space and OOV char excluded
    assert np.array_equal(seq, gt)

    trans = RecCTCLabelEncode(max_text_len=10, use_space_char=True)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    gt = np.array([0, 1, 2, trans.space_idx, 10, 11 ,12] + [trans.blank_idx]*3)
    assert trans.num_valid_chars==36+1, 'num_valid_chars is {}'.format(trans.num_valid_chars)
    assert trans.num_classes==37+1
    assert np.array_equal(seq, gt)
    assert out['length'] ==len(text)-1, 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=True, length

    trans = RecAttnLabelEncode(max_text_len=10, use_space_char=True)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    gt = np.array([trans.go_idx] + [2, 3, 4, trans.space_idx, 12, 13, 14] + [trans.stop_idx] + [trans.go_idx]*1)
    assert trans.num_valid_chars==36+1, 'num_valid_chars is {}'.format(trans.num_valid_chars)
    assert trans.num_classes == 38 + 1
    assert np.array_equal(seq, gt)
    assert out['length'] ==len(text)-1, 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=True, length

    trans = RecCTCLabelEncode(max_text_len=10, character_dict_path='mindocr/utils/dict/en_dict.txt', use_space_char=False)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    gt = np.array([ 0, 1, 2, 94, 49, 50, 51, 95, 95, 95])
    assert trans.num_valid_chars==95
    assert trans.num_classes==96
    assert out['length'] ==len(text), 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=False, but the dict contains space, % is also in dict

    trans = RecAttnLabelEncode(max_text_len=10, character_dict_path='mindocr/utils/dict/en_dict.txt', use_space_char=False)
    inp = {'label': text}
    out = trans(inp)
    seq = out['text_seq']
    assert trans.num_valid_chars==95
    assert trans.num_classes==97
    assert out['length'] == len(text), 'Not equal: {}, {}'.format(out['length'], text) # use_space_char=False, but the dict contains space, % is also in dict
