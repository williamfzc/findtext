import numpy as np
import cv2
import typing
from tesserocr import PyTessBaseAPI, RIL
from PIL import Image


class WordBlock(object):
    def __init__(self, box: dict, location: typing.Union[list, tuple], content: str):
        # reused by other calculations
        self.box = box
        # human readable location
        self.location = location
        # ocr content (text)
        self.content = content


class FindText(object):
    def __init__(self, lang: str = None):
        if not lang:
            lang = 'eng'
        self.lang = lang

    @staticmethod
    def _get_border_point_from_box(box: dict) -> typing.List[tuple]:
        return [
            # left top
            (box['x'], box['y']),
            # right bottom
            (box['x'] + box['w'], box['y'] + box['h'])
        ]

    def _get_word_block_list_from_image(self, image: Image, find_type: int) -> typing.List[WordBlock]:
        word_list = list()
        with PyTessBaseAPI(lang=self.lang) as api:
            api.SetImage(image)
            boxes = api.GetComponentImages(find_type, True)
            for _, box, *_ in boxes:
                api.SetRectangle(box['x'], box['y'], box['w'], box['h'])
                ocr_result = api.GetUTF8Text()
                word_list.append(
                    WordBlock(
                        box=box,
                        location=self._get_border_point_from_box(box),
                        content=ocr_result
                    )
                )
        return word_list

    def _find(self,
              image_path: str = None,
              image_object: np.ndarray = None,
              find_type: str = None) -> typing.List[WordBlock]:
        """ SHOULD NOT be used directly """
        if image_path:
            image_object = cv2.imread(image_path)
        if image_object is None:
            raise AttributeError('need image_path or image_object')

        find_type_dict = {
            'textline': RIL.TEXTLINE,
            'word': RIL.WORD,
        }

        assert find_type, 'find type must be filled'
        find_type_code: int = find_type_dict[find_type]

        # do not change the raw image
        image = Image.fromarray(image_object)
        return self._get_word_block_list_from_image(image, find_type_code)

    def find_text_line(self,
                       image_path: str = None,
                       image_object: np.ndarray = None) -> typing.List[WordBlock]:
        return self._find(image_path, image_object, 'textline')

    def find_word(self,
                  image_path: str = None,
                  image_object: np.ndarray = None) -> typing.List[WordBlock]:
        return self._find(image_path, image_object, 'word')
