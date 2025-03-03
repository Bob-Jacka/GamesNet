from collections import OrderedDict
from typing import Iterator, Literal

from torch import Tensor
from torch.nn import (
    Conv2d,
    Linear,
    Sequential,
    LeakyReLU,
    MaxPool2d,
    Dropout2d,
    Parameter,
    Module, Sigmoid, Flatten
)

from core.functional.Settings import (
    leaky_relu_value,
)
from core.functional.Utils import print_error, print_success


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
    """
    Inner structure of neuro block.
    """

    def __init__(self, sizes: tuple[int, int] | list[int, int], input_dropout_rate: float, class_to_create: Literal['Conv2d', 'Linear', 'Flatten'], kernel_size: int = 3,
                 padding: int = 1, stride: int = 1, bias: bool = True, *args, **kwargs):
        """
        Constructor for neuro block with different layers.
        :param sizes: 2 value tuple, where first element is width of image and second value is height.
        :param input_dropout_rate: values in tensors with this float number will be cut off.
        :param kernel_size: size of window to compute. By default, equals 3.
        :param: padding:
        :param class_to_create torch classes of models to create, can be one of ... .
        :param args: parameters for super constructor.
        :param kwargs: parameters for super constructor.
        """
        super().__init__(*args, **kwargs)
        self.dropout_rate = input_dropout_rate
        self.nb_kernel_size = kernel_size
        self.padding = padding
        self.create_class = class_to_create
        match class_to_create:
            case 'Conv2d':
                self.__create_conv_layer__(
                    input_count=sizes[0],
                    output_count=sizes[1],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=stride,
                    bias=bias,
                    dropout_rate=input_dropout_rate
                )
            case 'Linear':
                self.__create_linear_layer__(
                    input_count=sizes[0],
                    output_count=sizes[1],
                    dropout_rate=input_dropout_rate,
                    bias=bias,
                )
            case 'Flatten':
                self.__create_flatten_layer__(
                    dropout_rate=input_dropout_rate,
                    start_dim=sizes[0],
                    end_dim=sizes[1]
                )
            case _:
                print_error(f'Error occurred, expecting types Conv2d, Linear or Flatten, got {type(class_to_create)} instead.')

    def __create_conv_layer__(self, input_count: int, output_count: int, kernel_size: int, stride: int, padding: int, bias: bool, dropout_rate: float):
        """
        *Private method of class*.
        Creates convolutional network with given parameters.
        :param input_count: input count of connections to network.
        :param output_count: output count of connections to network.
        :param dropout_rate: drop out rate for Dropout layer, cut off this value from input flow.
        :param kernel_size: size of window to compute.
        :return: convolutional layer wrapped in inner structure.
        """
        self.inner_structure = Sequential(
            OrderedDict([
                ('input_conv', Conv2d(in_channels=input_count, out_channels=output_count, kernel_size=kernel_size, padding=padding, stride=stride, bias=bias)),
                ('l_relu', LeakyReLU(leaky_relu_value)),
                ('maxPool_layer', MaxPool2d(kernel_size=kernel_size, stride=stride)),
                ('drop_layer', Dropout2d(p=dropout_rate))
            ])
        )
        print_success('Conv network created.')

    def __create_linear_layer__(self, input_count: int, output_count: int, bias: bool, dropout_rate: float):
        """
        *Private method of class*.
        Creates Linear network with given parameters.
        :param input_count: input count of connections.
        :param output_count: output count of connections.
        :param bias: bool value of structure bias.
        :param dropout_rate: drop out rate for Dropout layer, cut off this value from input flow.
        :return: linear layer in inner structure.
        """
        self.inner_structure = Sequential(
            OrderedDict([
                ('linear', Linear(input_count, output_count, bias=bias)),
                ('activation_func', Sigmoid()),
                ('drop_layer', Dropout2d(p=dropout_rate)),
            ]),
        )
        print_success('Linear network created.')

    def __create_flatten_layer__(self, dropout_rate: float, start_dim: int, end_dim: int):
        self.inner_structure = Sequential(
            OrderedDict([
                ('flatten', Flatten(start_dim, end_dim)),
                ('activation_func', Sigmoid()),
                ('drop_layer', Dropout2d(p=dropout_rate))
            ])
        )

    def get_parameters(self) -> Iterator[Parameter]:
        """
        Method for receiving NeuroBlock parameters.
        :return: Iterator with parameters.
        """
        return self.parameters()

    def forward(self, x: Tensor):
        """
        Inner torch function for computing.
        :param x: input tensor
        :return: nothing.
        """
        return self.inner_structure(x)
