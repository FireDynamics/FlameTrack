import configparser
import os



config_mode = 'TESTING'


def __get_default_values():
    default_values = {}
    default_values ['experiment_folder'] = '.'
    default_values ['IR_folder'] = 'exported_data/'
    default_values ['processed_data'] = 'processed_data/'
    return default_values

def __create_missing_config():
    if not os.path.exists('config.ini'):
        config = configparser.ConfigParser()
        config['DEFAULT'] = __get_default_values()

        with open('config.ini', 'w') as configfile:
            config.write(configfile)


def __get_value(name,config,config_mode = 'DEFAULT'):
    value =  config[config_mode].get(name, None)
    if value is None:
        try:
            value = config[config_mode]['name']
        except KeyError:
            raise KeyError(f'No value for {name} found in config file')
    return value

def get_experiments():
    config =get_config()
    exp_folder =__get_value('experiment_folder',config,config_mode)
    folders = os.listdir(exp_folder)
    return sorted([folder for folder in folders if os.path.isdir(os.path.join(exp_folder, folder))])


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
def get_IR_path(exp_name):
    return get_path(exp_name,'IR_folder')




def get_path(exp_name,path_name):
    config = get_config()
    exp_folder = __get_value('experiment_folder',config,config_mode)
    path = __get_value(path_name,config,config_mode)
    return os.path.join(exp_folder,exp_name,path)


__create_missing_config()