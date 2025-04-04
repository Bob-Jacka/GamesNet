from collections import OrderedDict
from datetime import datetime
from inspect import Parameter
from typing import Iterator

from torch import nn
from torch.nn import (
    BCELoss,
    Module,
    Sequential,
    Softmax,
    MSELoss,
    L1Loss,
    NLLLoss,
    CrossEntropyLoss,
    BCEWithLogitsLoss
)
from torch.nn.modules.loss import _Loss
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
    Private instance of neuro model.
    """

    __optimizer__: Optimizer
    """
    Private instance of optimizer, ex.Adam, Nadam.
    """

    __loss_fn__: _Loss
    """
    Private instance of loss function.
    """

    def __init__(self, is_already_trained: bool = False, *args, **kwargs):
        """
        Constractor for creating GameNet with defined count of layers.
        :param is_already_trained: bool value that define - get trained model or train new model.
        :param args: args parameters for super constractor.
        :param kwargs: kwargs parameters for super constractor.
        """
        try:
            super(GameNet, self).__init__(*args, **kwargs)
            if not is_already_trained:
                self.__model__ = Sequential(
                    OrderedDict(
                        [
                            ('Input_layer', Neuro_block((1, 32), input_dropout_rate=0.05, kernel_size=3, padding=1, class_to_create='Conv2d')),
                            ('First_layer', Neuro_block((32, 64), input_dropout_rate=0.06, kernel_size=6, padding=1, class_to_create='Conv2d')),
                            ('Second_layer', Neuro_block((64, 128), input_dropout_rate=0.07, kernel_size=9, padding=1, class_to_create='Conv2d')),

                            ('Third_layer', Neuro_block((756, 256), input_dropout_rate=0.08, padding=1, class_to_create='Linear')),
                            ('Fourth_layer', Neuro_block((256, 2), input_dropout_rate=0.09, padding=1, class_to_create='Linear')),
                            ('Result_layer', Softmax(2))
                        ]
                    )
                ).to(device)
            elif is_already_trained and len(os.listdir('../save_model')) != 0:
                self.__model__ = GameNet.load_model(load_path='', model_name_to_load=model_name, ext=model_ext)
            else:
                raise Exception('Error in creating model.')
        except Exception as e:
            print_error(f'Error occurred during model creating - {e.with_traceback(None)}.')
        finally:
            if self.__model__ is not None:
                self.create_optim()

    @staticmethod
    def load_model(load_path: str | PathLike, model_name_to_load: str, ext: Literal['.pth', '.pt', '.pwf']) -> Module | None:
        """
        Loads model, using load path as a path where model saved.
        :param model_name_to_load: name of the creating model.
        :param ext: extension of the model file.
        :param load_path: path string from load model.
        :return: loaded model object from file or in case of exception return None.
        """
        try:
            if exists(load_path) and len(os.listdir(load_path)) != 0:
                if ext.startswith('.'):
                    state_dict = torch.load(load_path + model_name_to_load + ext, weights_only=False)  # Было True
                    loaded_model = GameNet()
                    if 'module.' in next(iter(state_dict.keys())):
                        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
                    loaded_model.load_state_dict(state_dict, strict=False)
                    print_error('Model loaded and ready.')
                    return loaded_model
            else:
                print_error('Model is not exist.')
        except Exception as e:
            print_error(f'Error occurred during model loading. - {e.with_traceback(None)}.')

    @staticmethod
    def save_model(save_path: str | PathLike, __model__, __model_name: str = model_name, ext: Literal['.pth', '.pt', '.pwf'] = model_ext) -> None:
        """
        Method for saving model on saving full_path.
        :param ext: extension of the save file.
        :param __model_name: name of the model.
        :param __model__: model to save.
        :param save_path: where to save (directory)
        """
        try:
            if save_path is not None and __model__ is not None:
                full_path = save_path + __model_name + ext
                if exists(full_path):
                    torch.save(__model__.state_dict(), full_path)
                    print_success(f'Model save on full_path - "{full_path}".')
                else:
                    print_error(f'Full path to file does not exists. Given full path - {full_path}.')
            else:
                print_error('Given save full_path or model is None.')
        except Exception as e:
            print_error(f'Error occurred during model saving. - {e.with_traceback(None)}.')

    def forward(self, x):
        """
        Inner method of neuro model.
        :param x: input information tensor.
        :return: result of the model action.
        """
        return self.model(x)

    def get_result(self, input_img: str | PathLike, is_str_output: bool = False) -> str | None:
        """
        Method for get result of the test classify process.
        :param is_str_output: value that representing need in output value.
        :param input_img path to image to proceed by model.
        :return: 'success' for successful test, 'failed' for failed test, 'skip' for skipped test.
        """
        try:
            self.__model__.eval()
            with torch.no_grad():
                tensor_img = Utils.proceed_image(input_img)
                model_output = self.__model__(tensor_img)
                predicted = torch.max(model_output.data, 1).values
                result = __get_min_and_max__(predicted)
                if is_str_output:
                    return f'Probability of success is {result[0]}%, probability of failure is {result[1]}%'
                else:
                    print_success(f'Probability of success is {result[0]}%, probability of failure is {result[1]}%')
        except Exception as e:
            print_error(f'Error occurred during model result prediction - {e.with_traceback(None)}.')

    def train_model(self, train_data_loader: DataLoader, train_epochs_count: int = 10, after_train_save: bool = False, path_on_after_train: str | PathLike = ''):
        """
        Method for training model on images during epoch_count.
        Include execution timer.
        :param train_data_loader: object for storing data.
        :param train_epochs_count: count of epoch to train the model.
        :param after_train_save: bool value represent needs of saving the model after train.
        :param path_on_after_train: path on which will model saved.
        :return: nothing.
        """
        try:
            squeezed_labels: Tensor
            self.__model__.train()
            start_train = datetime.now()
            print_info(f'Training start at - {start_train}.')
            epoch_count = range(0, train_epochs_count)
            for epoch in epoch_count:
                running_loss: float = 0.0
                for images, labels in train_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    increased_labels = labels
                    for image in images:
                        running_loss = self.__neuro_learn__(image, increased_labels, running_loss, True)
                    print_info(f'Current train epoch - №{epoch + 1} of {epoch_count.stop}\'s, Loss: {running_loss / len(train_data_loader):.4f}.')
            if after_train_save:
                print_success('After train save occurred.')
                GameNet.save_model(path_on_after_train, __model__=self.__model__)
            train_end = datetime.now()
            print_info(f'Training end at - {train_end}.')
            self.__model__.eval()  # switch neuro model back to the eval mode.
            self.__execution_time__(start_train, train_end)
        except Exception as e:
            e.add_note('Game net class - train_model method.')
            print_error(f'Error occurred during model training. - {e.with_traceback(None)}.')

    def test_model(self, test_data_loader: DataLoader, test_epochs_count: int = 5, after_test_save: bool = False, path_on_after_test: str | PathLike = ''):
        """
        Method for testing model on unseen images.
        :param test_data_loader: object for storing data.
        :param test_epochs_count: count of epoch to test the model.
        :param after_test_save: bool value represent needs of saving the model after test.
        :param path_on_after_test: bool value represent needs of saving the model after train.
        :return: nothing.
        """
        try:
            squeezed_labels: Tensor
            self.__model__.eval()
            print_info('Testing start:')
            start_time = datetime.now()
            for epoch in range(test_epochs_count):
                running_loss: float = 0.0
                for images, labels in test_data_loader:
                    images, labels = images.to(device), labels.to(device)
                    squeezed_labels = labels.unsqueeze(0).unsqueeze(1).unsqueeze(2)
                    for image in images:
                        running_loss = self.__neuro_learn__(image, squeezed_labels, running_loss, False)
                    print_info(f'Current test epoch - №{epoch + 1} of {epoch}\'s, Loss: {running_loss / len(test_data_loader):.4f}.')
            if after_test_save:
                print_error('After test save occurred.')
                GameNet.save_model(path_on_after_test, __model__=self.__model__)
            end_time = datetime.now()
            self.__execution_time__(start_time, end_time)
        except Exception as e:
            e.add_note('Game net class - test_model method.')
            print_error(f'Error occurred during model testing. - {e.with_traceback(None)}.')

    def __neuro_learn__(self, image, labels, running_loss: float, mode: bool) -> float:
        """
        Private inner logic of neuro learning.
        :param image: Image instance to proceed.
        :param labels: Labels of identifiers.
        :param running_loss: Loss value of model.
        :param mode: Value of train or test.
        :return: Updated running_loss value of model.
        """
        self.__optimizer__.zero_grad()
        outputs = self.__model__(image)

        additional_tensor =  torch.ones(64)

        scalar_tensor = labels
        to_recog = torch.add(additional_tensor, scalar_tensor)
        loss = self.__loss_fn__(torch.permute(outputs.squeeze(0), (0, 2, 1)), to_recog)
        print(f'\tPicture analyzed at {datetime.now().time()}.')
        with torch.set_grad_enabled(mode):
            loss.backward()
            self.__optimizer__.step()
        running_loss += loss.item()
        return running_loss

    @staticmethod
    def __execution_time__(start_time: datetime, end_time: datetime):
        """
        Private static function for show how much time spent on training or test.
        :param start_time: test or train start execution time.
        :param end_time: test or train end execution time.
        """
        difference = end_time - start_time
        seconds_in_day = 24 * 60 * 60
        train_time = divmod(difference.days * seconds_in_day + difference.seconds, 60)
        print_info(f'Training has taken - {train_time[0]} minutes and {train_time[1]} seconds.')

    def create_optim(self, is_manual_create: bool = False):
        """
        :param is_manual_create: represents manual creation with loss function selection.
        None safety method to create optimizer and loss function.
        :return: nothing.
        """
        try:
            if self.__model__ is not None:
                parameters: Iterator[Parameter] = self.__model__.parameters(recurse=True)
                if parameters is not None:
                    if not is_manual_create:
                        self.__optimizer__ = torch.optim.Adam(self.__model__.parameters(), lr=learning_rate)  # TODO в будущем заменить на Nadam.
                        self.__loss_fn__: _Loss = CrossEntropyLoss()
                        print_success('Default optimizer created.')
                    else:
                        counter = 0
                        for optim in optimizers:
                            print(f'№{counter}. {optim}')
                        user_choice = int_input_from_user(len(optimizers))
                        returned: str = optimizers[user_choice]
                        match returned:
                            case 'MSELoss':
                                self.__loss_fn__ = MSELoss()
                            case 'L1Loss':
                                self.__loss_fn__ = L1Loss()
                            case 'BCELoss':
                                self.__loss_fn__ = BCELoss()
                            case 'BCEWithLogitsLoss':
                                self.__loss_fn__ = BCEWithLogitsLoss()
                            case 'CrossEntropyLoss':
                                self.__loss_fn__ = CrossEntropyLoss()
                            case 'NLLLoss':
                                self.__loss_fn__ = NLLLoss()
                            case _:
                                print_error(f'Wrong type. Expected - {optimizers}, got {returned} instead.')
                        print_success('Optimizer created.')
                else:
                    print_error('Optimizer cannot created. Because parameters of model is None.')
            else:
                print_error('Error in creating optimizer, because model is None.')
        except Exception as e:
            print_error(f'Error occurred during creating optimizer. - {e.with_traceback(None)}.')


def __get_min_and_max__(tensor: Tensor) -> tuple[int, int]:
    """
    Private static function of GameNet class.
    elem[0] - first value of first array of tensor.
    elem[1] - second value of first array of tensor.
    :param tensor: tensor to proceed.
    :return: tuple of two values, where first value is variety of success and second value is variety of failure.
    """
    max_first: float = 0.0
    max_second: float = 0.0
    for elem in tensor:
        if elem[0] > max_first:
            max_first = elem[0]
        elif elem[1] > max_second:
            max_second = elem[1]
    return int(max_first * 100), int(max_second * 100)
