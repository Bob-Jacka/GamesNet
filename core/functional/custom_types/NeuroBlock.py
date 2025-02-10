from collections import OrderedDict
from types import UnionType
from typing import Iterator

from torch.nn import (
    Conv2d,
    Linear,
    Sequential,
    LeakyReLU,
    MaxPool2d,
    Dropout2d,
    Parameter,
    Module
)

from core.functional.Settings import (
    leaky_relu_value,
    dropout_rate
)


class Neuro_block(Module):
    """
    Class for one neuro block.
    Implements nn.Module.
    Contains methods for constructing Linear or Conv neural networks.
    """

    img_input_size: tuple[int, int]
    """
    Size of the input image.
    Width x Height.
    """

    nb_kernel_size: int
    """
    Kernel size of the neuro block.
    """

    create_class: UnionType
    """
    Which class to be created.
    """

    inner_structure: Sequential

    def __init__(self, input_size: tuple[int, int], kernel_size: int = 3, class_to_create=Conv2d | Linear,
                 *args, **kwargs):
        """
        Constructor for neuro block with different layers.
        :param input_size 2 value tuple, where first element is width of image and second value is height.
        :param kernel_size: size of window to compute. By default, equals 3.
        :param class_to_create torch classes of models.
        :param args: None.
        :param kwargs: None.
        """
        super().__init__(*args, **kwargs)
        self.img_input_size = input_size
        self.nb_kernel_size = kernel_size
        self.create_class = class_to_create
        if check_type(class_to_create, Conv2d):
            self.__create_conv_net__(input_count=input_size[0], output_count=input_size[1], kernel_size=kernel_size)
        elif check_type(class_to_create, Linear):
            self.__create_linear_net__(input_count=input_size[0], output_count=input_size[1])
        else:
            print(f'Error occurred, expecting types Conv2d or Linear, got {type(class_to_create)} instead.')

    def __create_conv_net__(self, input_count: int, output_count: int, kernel_size: int):
        """
        *Private method of class*.
        Creates convolutional network with given parameters.
        :param input_count: input count of connections.
        :param output_count: output count of connections.
        :param kernel_size: size of window to compute.
        :return: nothing.
        """
        self.inner_structure = Sequential(
            OrderedDict([
                ('input_conv',
                 Conv2d(in_channels=input_count, out_channels=output_count, kernel_size=kernel_size, padding=1)),
                ('l_relu', LeakyReLU(leaky_relu_value)),
                ('maxPool', MaxPool2d(kernel_size=kernel_size - 2, stride=2)),
                ('drop', Dropout2d(p=dropout_rate))
            ])
        )
        print('Conv network created.')

    def __create_linear_net__(self, input_count: int, output_count: int):
        """
        *Private method of class*.
        Creates convolutional network with given parameters.
        :param input_count: input count of connections.
        :param output_count: output count of connections.
        :return: nothing.
        """
        self.inner_structure = Sequential(
            OrderedDict([
                ('linear1', Linear(input_count, output_count)),
                ('linear2', Linear(input_count, output_count)),
                ('drop', Dropout2d(p=dropout_rate)),
            ]),
        )
        print('Linear network created.')

    def get_parameters(self) -> Iterator[Parameter]:
        """
        Method for receiving NeuroBlock parameters.
        :return: Iterator.
        """
        return self.parameters()

    def forward(self, X):
        return self.inner_structure(X)


def check_type(to_check, expected_type) -> bool:
    """
    Static Function that checks equality of types.
    :param to_check: which object need to check.
    :param expected_type: type, that expected to be.
    :return: True \ or False.
    """
    return type(to_check) == type(expected_type)
