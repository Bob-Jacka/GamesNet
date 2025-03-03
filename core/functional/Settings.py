"""
File contains some important value definitions that using in neuro network.
RULES:
    Do not use variables, that starts and end with double underscores (__) outside of this package.
    Use this variables in other file to correct work of the program.
"""
from typing import Literal

import torch

input_img_size: tuple[int, int] = (1705, 780)
"""
Tuple value. Value of the input image to proceed.
First value - width, Second value - height.
Width x Height of the image.
"""

learning_rate: float = 0.001
"""
Value that specifies float number of optimizer step.
"""

device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
"""
Value that specifies where will be loaded model with data.
"""

leaky_relu_value: float = 0.01
"""
Used for leaky_relu components of neuro net.
Cut that value in input signals.
"""

__static_pic_ext__ = '.png'
"""
Image extension.
Because train and test images captured by "windows print screen" function static picture extension is .png and not .jpeg instead.
"""

user_input_cursor: str = '>> '
"""
User input cursor used in input() function and select_terminal().
"""

model_name = 'network'
"""
Global name of the model.
"""

model_ext: Literal['.pth', '.pt', '.pwf'] = '.pth'
"""
Global model extension of the file.
"""

success_img_indicator: str = '_startPage' + __static_pic_ext__
"""
Representing success image result, which will be used in update labels function.
"""

failure_img_indicator: str = '_failure' + __static_pic_ext__
"""
Representing failure image result, which will be used in update labels function.
"""

test_labels: dict[str, int] = {
    'Success': 0,
    'Failed': 1
}
"""
Map of test labels.
"""

optimizers: tuple = ('MSELoss', 'L1Loss', 'BCELoss', 'BCEWithLogitsLoss', 'CrossEntropyLoss', 'NLLLoss')
"""
Optimizer names.
"""
