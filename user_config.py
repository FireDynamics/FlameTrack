import configparser
import os

# create config file if not exists



def get_default_value(name):
    default_values = {}
    default_values ['data_prefix_path'] = '/Volumes/Tam Backup/IR'
    default_values ['data_folder'] = 'data/'
    default_values ['dewarped_data_folder'] = 'dewarped_data/'
    default_values ['edge_results_folder'] = 'edge_results/'
    default_values ['saved_data'] = 'saved_data/'
    default_values ['canon_folder'] = r'/Volumes/Tam Backup/OM/'
    return default_values.get(name, None)

def create_missing_config():
    if not os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config['DEFAULT'] = {}
        config['DEFAULT']['data_prefix_path'] = '/Volumes/Tam Backup/IR'
        config['DEFAULT']['data_folder'] = 'data/'
        config['DEFAULT']['dewarped_data_folder'] = 'dewarped_data/'
        config['DEFAULT']['edge_results_folder'] = 'edge_results/'
        config['DEFAULT']['saved_data'] = 'saved_data/'
        config['DEFAULT']['canon_folder'] = r'/Volumes/Tam Backup/OM/'

        with open('config.ini', 'w') as configfile:
            config.write(configfile)


def get_path(name):
    create_missing_config()
    config = configparser.ConfigParser()
    config.read('config.ini')
    folder = config['DEFAULT'].get(name, get_default_value(name))
    path_prefix = config['DEFAULT']['data_prefix_path']
    result = os.path.join(path_prefix, folder)
    if folder is None or not os.path.exists(result):
        raise FileNotFoundError(
            f'Folder \"{folder}\" does not exist at \"{path_prefix}\", either create it or change the path for \"{name}\" in config.ini')
    return result

create_missing_config()