"""
                                *Games Net.*
Entry point in this neuro network.
This file may be used to train neuro network or some tests.
"""

import os
import threading

from torch.nn import Module
from torch.utils.data import DataLoader

from core.Entities.GameNet import GameNet
from core.functional import Utils
from core.functional.Settings import (
    model_name,
    model_ext
)
from core.functional.Utils import (
    int_input_from_user,
    print_error,
    print_info,
    duo_vals_input_from_user
)

train_labels_dir_path = 'core/data/train_data/'
test_labels_dir_path = 'core/data/validate_data/'

img_dir_path_train = 'core/data/train_data/images/'
img_dir_path_validate = 'core/data/validate_data/images/'

save_path = 'core/save_model/'

train_dataloader: tuple[DataLoader, DataLoader | None] | None
test_dataloader: tuple[DataLoader, DataLoader | None] | None

model: GameNet | Module
"""
Global instance of model.
"""


def __init_dataloaders__():
    """
    Private helper function for threading.
    :return: in global statements dataloader will be initialized.
    """
    global train_dataloader, test_dataloader
    try:
        train_dataloader = Utils.get_dataloader(labels_dir_path=f'{train_labels_dir_path}labels.csv',
                                                img_dir_path=img_dir_path_train)

        test_dataloader = Utils.get_dataloader(labels_dir_path=f'{train_labels_dir_path}labels.csv',
                                               img_dir_path=img_dir_path_validate)
    except Exception as e:
        print_error(f'Error in dataloaders. Dataloaders are None. - {e.with_traceback(None)}.')


def __init_model__():
    """
    Private helper function for threading.
    :return: nothing.
    """
    global model
    model = GameNet()


def __result__():
    """
    Private helper function for threading.
    :return: nothing.
    """
    global model
    if model is not None:
        model.get_result(Utils.select_terminal(img_dir_path_train, is_full_path_ret=True))
    else:
        print_error('Model is None.')


def model_action_menu():
    """
    Private helper function for model actions.
    :return: nothing.
    """
    while True:
        global model
        try:
            if model is not None:
                print()
                print('Select model action.')
                print('1. Train model,')
                print('2. Test model,')
                print('3. Load model,')
                print('4. Save model,')
                print('5. Model parameters,')
                print('6. Model weights,')
                print('7. Disable model,')
                print('8. Back.')
                user_input_action = int_input_from_user(7)
                match user_input_action:
                    case 1:
                        model.train_model(
                            train_dataloader[0],
                            train_epochs_count=Utils.int_input_from_user(values_range=2, topic='Enter value of epochs to train'),
                            after_train_save=True,
                            path_on_after_train=save_path
                        )
                        continue
                    case 2:
                        model.test_model(
                            train_dataloader[1],
                            test_epochs_count=Utils.int_input_from_user(values_range=2, topic='Enter value of epochs to test'),
                            after_test_save=True,
                            path_on_after_test=save_path
                        )
                        continue
                    case 3:
                        model = GameNet.load_model(save_path, model_name, model_ext)
                        continue
                    case 4:
                        GameNet.save_model(save_path, model, model_name, model_ext)
                        continue
                    case 5:
                        Utils.get_model_parameters(model.state_dict())
                        continue
                    case 6:
                        Utils.get_model_weight(model.state_dict())
                        continue
                    case 7:
                        model = None
                        continue
                    case 8:
                        break
                    case _:
                        print_error('Wrong argument. Try again.')
                        continue
            else:
                print('Create model first.')
        except Exception as e:
            print_error(f'Error occurred in labels function - {e.with_traceback(None)}.')


def labels_menu():
    """
    Private helper menu function.
    :return: nothing.
    """
    while True:
        try:
            print()
            print('Select labels action')
            print('1. Update - Train labels.')
            print('2. Update - Test labels.')
            print('3. Change labels identifiers.')
            print('4. Back.')
            user_input_labels = int_input_from_user(4)
            match user_input_labels:
                case 1:
                    Utils.update_labels(train_labels_dir_path)
                    continue
                case 2:
                    Utils.update_labels(test_labels_dir_path)
                    continue
                case 3:
                    Utils.change_labels(
                        new_label_str=Utils.str_input_from_user(topic='Enter new labels name.'),
                        old_label_str=Utils.str_input_from_user(topic='Enter old labels name.'),
                        labels_dir_path=train_labels_dir_path
                    )
                    continue
                case 4:
                    break
                case _:
                    print_error('Wrong argument. Try again.')
                    continue
        except Exception as e:
            print_error(f'Error occurred in labels function - {e.with_traceback(None)}.')


if __name__ == '__main__':
    init_dataloaders_script = lambda: threading.Thread(target=__init_dataloaders__()).run()
    init_model_script = lambda: threading.Thread(target=__init_model__()).run()

    if len(os.listdir(save_path)) == 0:
        init_dataloaders_script()
        init_model_script()
    else:
        print_info('Save directory is not empty, you may load model, instead of using not trained.')
        print_info('Do you want to load model, y / n.')
        res = duo_vals_input_from_user()
        match res:
            case True:
                init_dataloaders_script()
                model = GameNet.load_model(save_path, model_name, model_ext)
            case False:
                init_dataloaders_script()
                init_model_script()

    # Main action cycle
    while True:
        print('... - Means nested menu.')
        print('!!! Exit from program will not save neuro model.')
        print('\t Main menu')
        print('1. Get result,')
        print('2. Model action...,')
        print('3. Labels action...,')
        print('4. Get max size of image,')
        print('5. See train image,')
        print('6. Exit program.')
        user_input = Utils.int_input_from_user()
        try:
            match user_input:
                case 1:
                    __result__()
                    continue
                case 2:
                    model_action_menu()
                    continue
                case 3:
                    labels_menu()
                    continue
                case 4:
                    Utils.get_max_size(img_dir_path_train)
                    continue
                case 5:
                    Utils.show_img(Utils.select_terminal(img_dir_path_train, is_full_path_ret=True))
                    continue
                case 6:
                    print('Bye.')
                    break
                case _:
                    print_error('Wrong argument.')
                    continue
        except Exception as e:
            print_error(f'Error in user input - {e.with_traceback(None)}.')
