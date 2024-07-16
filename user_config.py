import configparser
import os
import DataTypes

# create config file if not exists
config_mode = 'SERVER'


def __get_default_values():
    default_values = {}
    default_values ['exp_folder'] = '.'
    default_values ['IR_folder'] = 'exported_data/'
    default_values ['saved_data'] = 'processed_data/'
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
            value = config['DEFAULT']['name']
        except KeyError:
            raise KeyError(f'No value for {name} found in config file')
    return value

def get_experiments():
    config =get_config()
    exp_folder =__get_value('experiment_folder',config,config_mode)
    folders = os.listdir(exp_folder)
    return [folder for folder in folders if os.path.isdir(os.path.join(exp_folder, folder))]


def get_config():
    config = configparser.ConfigParser()
    config.read('config.ini')
    return config
def get_IR_path(exp_name):
    config = get_config()
    exp_folder = __get_value('experiment_folder',config,config_mode)
    IR_folder = __get_value('IR_folder',config,config_mode)
    path = os.path.join(exp_folder, exp_name, IR_folder)
    return path

def get_IR_data(exp_name):
    return DataTypes.IrData(get_IR_path(exp_name))


__create_missing_config()