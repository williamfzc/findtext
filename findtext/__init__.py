import numpy as np
import cv2
import typing
import locale
from tesserocr import PyTessBaseAPI, RIL
import tesserocr
from PIL import Image

locale.setlocale(locale.LC_ALL, "C")


class WordBlock(object):
    def __init__(self, box: dict, content: str):
        # reused by other calculations
        self.box = box
        # human readable location
        self.location = self._get_border_point_from_box(box)
        self.left_top_point, self.right_bottom_point = self.location
        # size
        self.width, self.height = self._get_size_from_box(box)
        # ocr content (text)
        self.content = self._content_filter(content)

    @staticmethod
    def _get_border_point_from_box(box: dict) -> typing.List[tuple]:
        return [
            # left top
            (box['x'], box['y']),
            # right bottom
            (box['x'] + box['w'], box['y'] + box['h'])
        ]

    @staticmethod
    def _get_size_from_box(box: dict) -> list:
        """ return (width, height) """
        return [box['w'], box['h']]

    @staticmethod
    def _content_filter(content: str) -> str:
        return content.replace(' ', '').replace('\n', '')

    def get_y_interval(self, offset: int = None) -> list:
        if not offset:
            offset = 0
        return [self.left_top_point[1] - offset,
                self.right_bottom_point[1] + offset]

    def update_box(self, box: dict):
        self.location = self._get_border_point_from_box(box)
        self.left_top_point, self.right_bottom_point = self.location
        self.width, self.height = self._get_size_from_box(box)


class FindText(object):
    def __init__(self, lang: str = None):
        if not lang:
            lang = 'eng'
        self.lang = lang

    def __str__(self):
        return f'<FindText Object lang={self.lang}>'

    __repr__ = __str__

    @staticmethod
    def get_data_home() -> str:
        return tesserocr.get_languages()[0]

    @staticmethod
    def get_available_lang() -> list:
        return tesserocr.get_languages()[1]

    def _get_word_block_list_from_image(self,
                                        image: Image,
                                        find_type: int,
                                        spec_box: dict = None) -> typing.List[WordBlock]:
        word_list = list()
        with PyTessBaseAPI(lang=self.lang) as api:
            api.SetImage(image)
            if spec_box:
                ocr_result = api.GetUTF8Text()
                return [WordBlock(box=spec_box, content=ocr_result)]

            boxes = api.GetComponentImages(find_type, True)
            for _, box, *_ in boxes:
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocr_result = api.GetUTF8Text()
                word_list.append(
                    WordBlock(
                        box=box,
                        content=ocr_result
                    )
                )
        return word_list

    @staticmethod
    def _get_img_object(image_path: str = None, image_object: np.ndarray = None) -> np.ndarray:
        if image_path:
            # read as gray
            image_object = cv2.imread(image_path, 0)
        if image_object is None:
            raise AttributeError('need image_path or image_object')
        return image_object

    def _find(self,
              image_path: str = None,
              image_object: np.ndarray = None,
              find_type: str = None,
              spec_box: dict = None) -> typing.List[WordBlock]:
        """ SHOULD NOT be used directly """
        image_object = self._get_img_object(image_path, image_object)

        find_type_dict = {
            'textline': RIL.TEXTLINE,
            'word': RIL.WORD,
        }

        assert find_type, 'find type must be filled'
        find_type_code: int = find_type_dict[find_type]

        # do not change the raw image
        image = Image.fromarray(image_object)
        return self._get_word_block_list_from_image(image, find_type_code, spec_box)

    @staticmethod
    def crop_object(image_object: np.ndarray,
                    x_start: int = None,
                    x_end: int = None,
                    y_start: int = None,
                    y_end: int = None) -> np.ndarray:
        return image_object[y_start:y_end, x_start:x_end]

    def find_text_line(self,
                       image_path: str = None,
                       image_object: np.ndarray = None) -> typing.List[WordBlock]:
        return self._find(image_path, image_object, 'textline')

    def find_word(self,
                  image_path: str = None,
                  image_object: np.ndarray = None,
                  deep: bool = None,
                  offset: int = None,
                  *args, **kwargs) -> typing.List[WordBlock]:
        if not deep:
            return self._find(image_path, image_object, 'word', *args, **kwargs)
        if not offset:
            offset = 0

        image_object = self._get_img_object(image_path, image_object)
        height, width = image_object.shape

        # pre find (text line)
        textline_block_list = self.find_text_line(image_path, image_object)

        # find words in rescaled image
        final_word_list = list()
        for each_line in textline_block_list:
            sub_image = self.crop_object(
                image_object,
                0,
                width,

                # TODO out of range?
                each_line.left_top_point[1] - offset,
                each_line.right_bottom_point[1] + offset)
            word_list = self.find_word(image_object=sub_image)

            # location fix
            for each_word in word_list:
                new_box = {
                    'x': each_word.left_top_point[0],
                    'y': each_line.left_top_point[1] + each_word.left_top_point[1] - offset,
                    'w': each_word.box['w'],
                    'h': each_word.box['h'],
                }
                each_word.update_box(new_box)
            final_word_list.extend(word_list)
        return final_word_list
