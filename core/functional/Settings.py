"""
File contains some important value definitions that using in neuro network.
RULES:
    Do not use variables, that starts with double underscores (__) outside of this package.
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

dropout_rate: float = 0.001
"""
Dropout rate of the model.
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

success_img_indicator: str = '_startPage' + __static_pic_ext__
failure_img_indicator: str = '_Failure' + __static_pic_ext__
