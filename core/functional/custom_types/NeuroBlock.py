from collections import OrderedDict
from types import UnionType
from typing import Iterator

from termcolor import colored
from torch import Tensor
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
from typing_extensions import deprecated

from core.functional.Settings import (
    leaky_relu_value,
)
from core.functional.custom_types.Types import Neuro_type


class Neuro_block(Module):
    """
    Class for one neuro block, can be convolutional or linear.
    Implements nn.Module.
    Contains methods for constructing Linear or Conv neural networks.
    """

    nb_kernel_size: int
    """
    Kernel size of the neuro block.
    """

    dropout_rate: float
    """
    Dropout rate for dropout layer of neuro block.
    """

    create_class: str
    """
    Which class to be created.
    Equals only Conv2d or Linear.
    """

    inner_structure: Sequential

    def __init__(self, sizes: tuple[int, int], input_dropout_rate: float, class_to_create: Neuro_type, kernel_size: int = 3, padding: int = 1, stride: int = 1, bias: bool = True,
                 *args, **kwargs):
        """
        Constructor for neuro block with different layers.
        :param sizes 2 value tuple, where first element is width of image and second value is height.
        :param kernel_size: size of window to compute. By default, equals 3.
        :param class_to_create torch classes of models.
        :param args: None.
        :param kwargs: None.
        """
        super().__init__(*args, **kwargs)
        self.dropout_rate = input_dropout_rate
        self.nb_kernel_size = kernel_size
        self.create_class = class_to_create
        if class_to_create == 'Conv2d':
            self.__create_conv_layer__(input_count=sizes[0], output_count=sizes[1], kernel_size=kernel_size, padding=padding, stride=stride, bias=bias,
                                       dropout_rate=input_dropout_rate)
        elif class_to_create == 'Linear':
            self.__create_linear_layer__(input_count=sizes[0], output_count=sizes[1], dropout_rate=input_dropout_rate, bias=bias, percentage_to_reduce=20)
        else:
            print(colored(f'Error occurred, expecting types Conv2d or Linear, got {type(class_to_create)} instead.', 'red'))

    def __create_conv_layer__(self, input_count: int, output_count: int, kernel_size: int, stride: int, padding: int, bias: bool, dropout_rate: float):
        """
        *Private method of class*.
        Creates convolutional network with given parameters.
        :param input_count: input count of connections.
        :param output_count: output count of connections.
        :param dropout_rate: drop out rate for conv net, cut off this value from input flow.
        :param kernel_size: size of window to compute.
        :return: nothing.
        """
        self.inner_structure = Sequential(
            OrderedDict([
                ('input_conv', Conv2d(in_channels=input_count, out_channels=output_count, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)),
                ('l_relu', LeakyReLU(leaky_relu_value)),
                ('maxPool', MaxPool2d(kernel_size=kernel_size + 1, stride=stride)),
                ('drop', Dropout2d(p=dropout_rate))
            ])
        )
        print(colored('Conv network created.', 'green'))

    def __create_linear_layer__(self, input_count: int, output_count: int, bias: bool, dropout_rate: float, percentage_to_reduce: int):
        """
        *Private method of class*.
        Creates Linear network with given parameters.
        :param input_count: input count of connections.
        :param output_count: output count of connections.
        :param bias: bool value of structure bias.
        :param dropout_rate: drop out rate for conv net, cut off this value from input flow.
        :return: nothing.
        """

        self.inner_structure = Sequential(
            OrderedDict([
                ('linear1', Linear(input_count, output_count, bias=bias)),
                ('linear2', Linear(input_count - int(input_count * percentage_to_reduce / 100), output_count - int(output_count * percentage_to_reduce / 100), bias=bias)),
                ('drop', Dropout2d(p=dropout_rate)),
            ]),
        )
        print(colored('Linear network created.', 'green'))

    def get_parameters(self) -> Iterator[Parameter]:
        """
        Method for receiving NeuroBlock parameters.
        :return: Iterator.
        """
        return self.parameters()

    def forward(self, x: Tensor):
        """
        Inner torch function for computing.
        :param x: input tensor
        :return: nothing.
        """
        return self.inner_structure(x)


@deprecated('Deprecated, because type were replaced by Literals.')
def check_type(to_check: UnionType, expected_type: type) -> bool:
    """
    Static Function that checks equality of types.
    :param to_check: which object need to check.
    :param expected_type: type, that expected to be.
    :return: True or False.
    """
    try:
        return to_check == expected_type
    except Exception as e:
        print(e.__cause__)
        print(colored(f'Error in checking type - {e.with_traceback(None)}.', 'red'))
