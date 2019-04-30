def config():
    config_dict={
    "task_name":None,
    "epochs":0,
    "batch_size":1,
    "test_batch_size":1,
    "model_class":[],
    "model_params":[],
    "optimizer_function":None,
    "optimizer_params":{},
    "device_use":None, #set the device automatic if the device_use is None, else set the certaion numbers
    "summary_writer_open":True #open the summart writer
    }
    return config_dict


def common_solver_config():
    config_dict={
    "task_name":None,
    "epochs":0,
    "batch_size":1,
    "test_batch_size":1,
    "model_class":[],
    "model_params":[],
    "optimizer_function":None,
    "optimizer_params":{},
    "learning_rate_decay_epochs":[],
    "train":True,
    "test": True,
    "validate":  True,
    "restored_path":None,
    "begin_epoch":0,
    "device_use":None, #set the device automatic if the device_use is None, else set the certaion numbers
    "summary_writer_open":True
    }
    return config_dict
