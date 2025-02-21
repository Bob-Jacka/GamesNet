from collections import OrderedDict
from inspect import Parameter
from typing import Iterator, Literal

from torch import nn
from torch.nn import (
    BCELoss,
    Module,
    Sequential,
    Softmax
)
from torch.optim import Optimizer

from core.functional import Utils
from core.functional.Settings import *
from core.functional.Utils import *
from core.functional.custom_types.NeuroBlock import Neuro_block


class GameNet(nn.Module):
    """
    Class that represents torch model. Implementing nn module of torch library.
    """

    __model__: Module
    """
    Instance of neuro model.
    """
    __optimizer__: Optimizer
    """
    Instance of optimizer, ex.Adam, Nadam.
    """
    __loss_fn__: BCELoss
    """
    Instance of loss function.
    """

    __model_ext: str
    """
        Value represents file extensions.
        In torch library by default use '.pth'.
    """

    __model_name: str
    """
    Value represents model name to save and load.
    """

    __save_model_path__: str | PathLike
    """
    Path for model saving or loading.
    """

    def __init__(self, is_already_trained: bool = False, *args, **kwargs):
        """
        Contractor for creating GameNet with defined count of layers.
        :param is_already_trained: bool value that define - get trained model or train new model.
        :param args: args parameters for super constractor.
        :param kwargs: kwargs parameters for super constractor.
        """
        try:
            super().__init__(*args, **kwargs)
            if not is_already_trained:
                self.__model__ = Sequential(
                    OrderedDict(
                        [
                            ('Input_layer', Neuro_block((780, 780), input_dropout_rate=0.05, kernel_size=1, class_to_create='Linear')),

                            ('First_layer', Neuro_block((1, 256), input_dropout_rate=0.1, kernel_size=2, padding=1, class_to_create='Conv2d')),
                            ('Second_layer', Neuro_block((256, 64), input_dropout_rate=0.11, kernel_size=3, padding=1, class_to_create='Conv2d')),
                            ('Third_layer', Neuro_block((64, 4), input_dropout_rate=0.12, kernel_size=4, padding=1, class_to_create='Conv2d')),
                            ('Fourth_layer', Neuro_block((4, 1), input_dropout_rate=0.14, kernel_size=5, padding=1, class_to_create='Conv2d')),  # 1689 x 764

                            ('Fifth_layer', Neuro_block((764, 764), input_dropout_rate=0.15, class_to_create='Linear')),
                            ('Sixth_layer', Neuro_block((764, 100), input_dropout_rate=0.16, class_to_create='Linear')),
                            ('Seventh_layer', Neuro_block((100, 10), input_dropout_rate=0.17, class_to_create='Linear')),

                            ('Result_layer', Softmax(2))
                        ]
                    )
                ).to(device)
                self.create_optim()
            elif is_already_trained and len(os.listdir('../save_model')) != 0:
                self.__model__ = GameNet.load_model(load_path='', model_name='network', ext='.pth')
        except Exception as e:
            print_error(f'Error occurred during model creating - {e.with_traceback(None)}.')

    @staticmethod
    def load_model(load_path: str | PathLike, model_name: str, ext: Literal['.pth', '.pt', '.pwf']) -> Module:
        """
        Loads model, using load path as a path where model saved.
        :param model_name: name of the creating model.
        :param ext: extension of the model file.
        :param load_path: path string from load model.
        :return: loaded model object from file.
        """
        try:
            if exists(load_path) and len(os.listdir(load_path)) != 0:
                if ext.startswith('.'):
                    state_dict = torch.load(load_path + model_name + ext, weights_only=True)
                    loaded_model = torch.nn.Module()
                    if 'module.' in next(iter(state_dict.keys())):
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    loaded_model.load_state_dict(state_dict, strict=False)
                    loaded_model.eval()
                    print_error("Model loaded and ready.")
                    return loaded_model
            else:
                print_error("Model is not exist.")
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred during model loading. - {e.with_traceback(None)}.')

    @staticmethod
    def save_model(save_path: str | PathLike, __model_name: str = 'network', ext: Literal['.pth', '.pt', '.pwf'] = '.pth', __model__=None) -> None:
        """
        Method for saving model on saving full_path.
        :param ext: extension of the save file.
        :param __model_name: name of the model.
        :param __model__: model to save.
        :param save_path: where to save (directory) by default equals save_model_dir in Utils.py.
        :return: nothing.
        """
        try:
            if save_path is not None and __model__ is not None:
                full_path = save_path + __model_name + ext
                if exists(full_path):
                    torch.save(__model__.state_dict(), full_path)
                    print_success(f"Model save on full_path - '{full_path}'")
                else:
                    print_error('Full path to file does not exists.')
            else:
                print_error('Given save full_path or model is None.')
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred during model saving. - {e.with_traceback(None)}.')

    def forward(self, X):
        """
        Inner method of neuro model.
        :param X: ?
        :return: ?
        """
        return self.model(X)

    def get_result(self, input_img: str | PathLike) -> str:
        """
        Method for get result of the test classify.
        :param input_img path to image to proceed by model.
        :return: 'success' for successful test, 'failed' for failed test, 'skip' for skipped test.
        """
        try:
            tensor_img = Utils.proceed_image(input_img)
            model_output = self.__model__(tensor_img)  # TODO сделать возврат значения из метода
            return Test_results.SUCCESS.get_value()
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred during model result prediction. - {e.with_traceback(None)}.')

    def train_model(self, train_data_loader: DataLoader, train_epochs_count: int = 40, after_train_save: bool = False):
        """
        Method for training model on images during epoch_count.
        :param train_data_loader: object for storing data.
        :param train_epochs_count: count of epoch to train the model.
        :param after_train_save: bool value represent needs of saving the model after train.
        :return: nothing.
        """
        try:
            self.__model__.train()
            print('Training start:')
            epoch_count = range(train_epochs_count)
            for epoch in epoch_count:
                running_loss: float = 0.0
                for images, labels in train_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    for image in images:
                        self.__optimizer__.zero_grad()
                        outputs = self.__model__(image)
                        loss = self.__loss_fn__(outputs, labels)
                        loss.backward()
                        self.__optimizer__.step()
                        running_loss += loss.item()
                    print(f'Current train epoch - {epoch + 1} of {epoch_count}, Loss: {running_loss / len(train_data_loader):.4f}.')
            if after_train_save:
                print_success('After train save occurred.')
                self.save_model(self.__save_model_path__, __model__=self.__model__)
        except Exception as e:
            e.add_note('Game net class - train_model method.')
            print(e.__cause__)
            print_error(f'Error occurred during model training. - {e.with_traceback(None)}.')

    def test_model(self, test_data_loader: DataLoader, test_epochs_count: int = 20, after_test_save: bool = False):
        """
        Method for testing model on unseen images.
        :param test_data_loader: object for storing data.
        :param test_epochs_count: count of epoch to test the model.
        :param after_test_save: bool value represent needs of saving the model after test.
        :return: nothing.
        """
        try:
            self.__model__.eval()
            print('Testing start:')
            for epoch in range(test_epochs_count):
                running_loss: float = 0.0
                for images, labels in test_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    for image in images:
                        self.__optimizer__.zero_grad()
                        outputs = self.__model__(image)
                        loss = self.__loss_fn__(outputs, labels)
                        loss.backward()
                        self.__optimizer__.step()
                        running_loss += loss.item()
                    print(f'Current test epoch - {epoch + 1} of {epoch}, Loss: {running_loss / len(test_data_loader):.4f}.')
            if after_test_save:
                print_error('After test save occurred.')
                self.save_model(self.__save_model_path__)
        except Exception as e:
            e.add_note('Game net class - test_model method.')
            print(e.__cause__)
            print_error(f'Error occurred during model testing. - {e.with_traceback(None)}.')

    def create_optim(self):
        """
        None safety function to create optimizer and loss function.
        :return: nothing.
        """
        try:
            if self.__model__ is not None:
                parameters: Iterator[Parameter] = self.__model__.parameters(recurse=True)
                if parameters is not None:
                    self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=learning_rate)  # TODO в будущем заменить на Nadam.
                    self.__loss_fn__ = BCELoss()
                    print_success('Optimizer created.')
                else:
                    print_error('Optimizer cannot created. Because parameters of model is None.')
            else:
                print_error('Error in creating optimizer, because model is None.')
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred during creating optimizer. - {e.with_traceback(None)}.')
