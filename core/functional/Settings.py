"""
File contains some important value definitions that using in neuro network.
RULES:
    Do not use variables, that starts and end with double underscores (__) outside of this package.
Variables:
    save_model_dir,
    train_data_dir,
    validate_data_dir,
    device,
    input_img_size,
    learning_rate.
"""

import torch

input_img_size: tuple[int, int] = (1705, 780)
"""
Tuple value. Value of the input image to proceed.
First value - width, Second value - height.
Width x Height of the monitor.
"""

train_img_input_size: tuple[int, int, int, int] = (1, 1, 1705, 780)
"""
Tuple value. Value of the input image to proceed.
Third value - width, Fourth value - height.
batch_size x image x Width x Height of the monitor.
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

user_input_cursor: str = '>> '
"""
User input cursor used in input() function and select_terminal().
"""

success_img_indicator: str = '_startPage' + __static_pic_ext__
failure_img_indicator: str = '_Failure' + __static_pic_ext__
