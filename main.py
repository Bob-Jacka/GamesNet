"""
Entry point in this neuro network.
This file may be used to train neuro network or some tests.
"""

labels_dir_path = 'core/data/train_data/'
save_path = 'core/save_model/'

img_dir_path_train = 'core/data/train_data/images/'
img_dir_path_validate = 'core/data/validate_data/images/'

from core.Entities.GameNet import GameNet
from core.functional import Utils

try:
    train_dataloader = Utils.get_dataloader(labels_dir_path=f'{labels_dir_path}labels.csv',
                                            img_dir_path=img_dir_path_train)

    validate_dataloader = Utils.get_dataloader(labels_dir_path=f'{labels_dir_path}labels.csv',
                                               img_dir_path=img_dir_path_validate)
except Exception as e:
    print(e.__cause__)
    print('Dataloaders are None.')

game_net = GameNet()
game_net.train_model(train_dataloader[0], epochs_count=10, after_train_save=True)
game_net.save_model(save_path)

if __name__ == '__main__':
    print('Action.')
