def config():
    config_dict={
    "epochs":0,
    "dataset_function":None,
    "dataset_function_params":{},
    "learning_rate_decay_epochs":[],
    "train":True,
    "test": True,
    "validate":  True,
    "restored_path":None,
    "begin_epoch":0
    }
    return config_dict
