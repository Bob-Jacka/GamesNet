from inspect import Parameter
from os.path import exists
from typing import Iterator

from torch import nn
from torch.nn import (
    Sequential,
    Softmax,
    BCELoss,
    Module,
    Conv2d,
    Linear
)
from torch.optim import Optimizer

from core.functional.Settings import *
from core.functional.Utils import *
from core.functional.custom_types.NeuroBlock import Neuro_block


class GameNet(nn.Module):
    """
    Class that represents torch model. Implementing nn module of torch library.
    """

    __model__: Sequential
    __optimizer__: Optimizer
    __loss_fn__: BCELoss

    __neuro_block_parameters__: Parameter

    __model_ext: str = '.pth'
    """
        Value represents file extensions.
        In torch library by default use '.pth'
    """

    __model_name: str = 'model'
    """
    Value represents model name to save and load.
    """

    __save_model_path__: str

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        conv2d: type = type(Conv2d)
        linear: type = type(Linear)
        self.__model__ = Sequential(
            Neuro_block(input_img_size, 10, class_to_create=conv2d),  # conv layer
            Neuro_block(input_img_size, 8, class_to_create=conv2d),  # conv layer
            Neuro_block(input_img_size, class_to_create=linear),  # linear layer
            Softmax(2)  # need only two test results
        ).to(device)
        self.create_optim()
        print('GameNet constructor created.')

    def load_model(self, load_path: str) -> Module:
        """
        Loads model, using load path as a path where model saved.
        :param load_path: path string from load model.
        :return: loaded object from file.
        """
        if exists(load_path) and len(os.listdir(load_path)) != 0:
            state_dict = torch.load(load_path + self.model_name + self.ext, weights_only=True)
            loaded_model = torch.nn.Module()
            if 'module.' in next(iter(state_dict.keys())):
                state_dict = {k.replace('module.', ''):
                                  v for k, v in state_dict.items()}
            loaded_model.load_state_dict(state_dict, strict=False)
            loaded_model.eval()
            print("Model loaded and ready.")
            self.__save_model_path__ = load_path
            return loaded_model
        else:
            print("Model is not exist.")

    def forward(self, X):
        return self.model(X)

    def save_model(self, save_path: str):
        """
        Method for saving model on saving path.
        :param save_path: where to save (directory) by default equals save_model_dir in Utils.py.
        :return: nothing.
        """
        if save_path is not None and self.__model__ is not None:
            path = save_path + self.__model_name + self.__model_ext
            if exists(path):
                self.__save_model_path__ = save_path
                torch.save(self.__model__.state_dict(), path)
                print(f"Model save on path - '{path}'")
            else:
                print('Path does not exists.')
        else:
            print('Given save path is None.')

    def get_result(self, input_img) -> str:
        """
        Method for get result of the test classify.
        :return: 'success' for successful test, 'failed' for failed test, 'skip' for skipped test.
        """
        self.__model__(input_img)
        return Test_results.SUCCESS.value  # TODO сделать возврат значения из метода

    def train_model(self, data_loader: DataLoader, epochs_count: int = 40, after_train_save: bool = False):
        """
        Method for training model.
        :param data_loader: object for storing data.
        :param epochs_count: count of epoch to train the model.
        :param after_train_save: bool value represents needs of saving the model after train.
        :return: nothing.
        """
        if self.__model__ is not None:
            self.__model__.train()
            print('Training start:')
            for epoch in range(epochs_count):
                running_loss = 0.0
                for images, labels in data_loader:
                    images, labels = images.to(device), labels.to(device)
                    for image in images:
                        squeezed = image.unsqueeze(0)
                        self.__optimizer__.zero_grad()
                        outputs = self.__model__(squeezed)
                        loss = self.__loss_fn__(outputs, labels)
                        loss.backward()
                        self.__optimizer__.step()
                        running_loss += loss.item()
                    print(f'Epoch [{epoch + 1}/{epoch}], Loss: {running_loss / len(data_loader):.4f}')
            if after_train_save:
                self.save_model(self.__save_model_path__)
        else:
            print("Load model first.")

    def create_optim(self):
        """
        None safety function to create optimizer and loss function.
        :return: nothing.
        """
        if self.__model__ is not None:
            parameters: Iterator[Parameter] = self.__model__.parameters(recurse=True)
            if parameters is not None:
                self.__optimizer__ = torch.optim.Adam(parameters, lr=learning_rate)
                self.__loss_fn__ = BCELoss()
                print('Optimizer created.')
            else:
                print('Optimizer cannot created. Because parameters of model is None.')
        else:
            print('Error in creating optimizer, because model is None.')
