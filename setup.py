from setuptools import setup, find_packages


install_requirement_list = [
    'opencv-python',
    'pillow',
    'tesserocr',
]

setup(
    name='findtext',
    version='0.1.0',
    description='OCR, based on tesseract (tesserocr), with accuracy improvement',
    author='williamfzc',
    author_email='fengzc@vip.qq.com',
    url='https://github.com/williamfzc/findtext',
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=install_requirement_list
)
