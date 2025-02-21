"""
                                *Games Net.*
Entry point in this neuro network.
This file may be used to train neuro network or some tests.
"""
import threading

from torch.utils.data import DataLoader

from core.Entities.GameNet import GameNet
from core.functional import Utils
from core.functional.Utils import input_from_user, print_error

train_labels_dir_path = 'core/data/train_data/'
test_labels_dir_path = 'core/data/validate_data/'

img_dir_path_train = 'core/data/train_data/images/'
img_dir_path_validate = 'core/data/validate_data/images/'

save_path = 'core/save_model/'

train_dataloader: tuple[DataLoader, DataLoader | None] | None
test_dataloader: tuple[DataLoader, DataLoader | None] | None

model: GameNet
"""
Global instance of model.
"""


def init_dataloaders():
    """
    Helper function for threading.
    :return: nothing.
    """
    global train_dataloader, test_dataloader
    try:
        train_dataloader = Utils.get_dataloader(labels_dir_path=f'{train_labels_dir_path}labels.csv',
                                                img_dir_path=img_dir_path_train)

        test_dataloader = Utils.get_dataloader(labels_dir_path=f'{train_labels_dir_path}labels.csv',
                                               img_dir_path=img_dir_path_validate)
    except Exception as e:
        print(e.__cause__)
        print_error(f'Error in dataloaders. Dataloaders are None. - {e.with_traceback(None)}')


def init_model():
    """
    Helper function for threading.
    :return: nothing.
    """
    global model
    model = GameNet()


def result():
    """
    Helper function for threading.
    :return: nothing.
    """
    if model is not None:
        model.get_result(Utils.select_terminal(img_dir_path_train))
    else:
        print_error('Model is None.')


def model_action_menu():
    """
    Helper function for model actions.
    :return: nothing.
    """
    while True:
        try:
            if model is not None:
                print()
                print('Select model action.')
                print('1. Train model.')
                print('2. Load model.')
                print('3. Save model.')
                print('4. Model parameters.')
                print('5. Back.')
                user_input_action = input_from_user()
                match user_input_action:
                    case 1:
                        model.train_model(train_dataloader[0], train_epochs_count=10, after_train_save=True)
                        continue
                    case 2:
                        model.test_model(train_dataloader[1], test_epochs_count=10, after_test_save=True)
                        continue
                    case 3:
                        model.save_model(save_path, 'network', '.pth')
                        continue
                    case 4:
                        Utils.get_model_parameters(model.state_dict())
                        continue
                    case 5:
                        break
                    case _:
                        print('Wrong argument. Try again.')
            else:
                print('Create model first.')
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred in labels function - {e.with_traceback(None)}.')


def labels_menu():
    """
    Helper function.
    :return: nothing.
    """
    while True:
        try:
            print()
            print('Select which labels to update.')
            print('1. Train labels.')
            print('2. Test labels.')
            print('3. Back.')
            user_input_labels = input_from_user()
            match user_input_labels:
                case 1:
                    Utils.update_labels(train_labels_dir_path)
                    continue
                case 2:
                    Utils.update_labels(test_labels_dir_path)
                    continue
                case 3:
                    break
                case _:
                    print_error('Wrong argument. Try again.')
                    continue
        except Exception as e:
            print(e.__cause__)
            print_error(f'Error occurred in labels function - {e.with_traceback(None)}.')


if __name__ == '__main__':
    dataloaders_thread = threading.Thread(target=init_dataloaders())
    dataloaders_thread.run()

    init_model_thread = threading.Thread(target=init_model())
    init_model_thread.run()
    while True:
        print()
        print('\t Main menu')
        print('1. Get result,')
        print('2. Model action...,')
        print('3. Update labels...,')
        print('4. Get max size of image,')
        print('5. See train images,')
        print('6. Exit program.')
        user_input = input_from_user()
        try:
            match user_input:
                case 1:
                    result()
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
            print(e.__cause__)
            print_error(f'Error in user input - {e.with_traceback(None)}.')
