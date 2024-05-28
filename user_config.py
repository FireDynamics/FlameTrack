import configparser
import os

# create config file if not exists



def create_missing_config():
    if not os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['DEFAULT']['data_prefix_path'] = '/Volumes/Tam Backup/IR'
        config['DEFAULT']['data_folder'] = 'data/'
        config['DEFAULT']['dewarped_data_folder'] = 'dewarped_data/'
        config['DEFAULT']['edge_results_folder'] = 'edge_results/'
        config['DEFAULT']['saved_data'] = 'saved_data/'

        with open('config.ini', 'w') as configfile:
            config.write(configfile)


def get_path(name):
    create_missing_config()
    config = configparser.ConfigParser()
    config.read('config.ini')
    folder = config['DEFAULT'][name]
    path_prefix = config['DEFAULT']['data_prefix_path']
    result = os.path.join(path_prefix, folder)
    if not os.path.exists(result):
        raise FileNotFoundError(
            f'Path {result} does not exist, either create it or change the path for \"{name}\" in config.ini')
    return result

create_missing_config()