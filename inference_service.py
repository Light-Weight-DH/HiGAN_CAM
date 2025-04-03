# inference_service.py
# 그대로 두고,
from inference_service_sub import exec_init_model, exec_inference_dataframe, exec_inference_file

import logging
logger = logging.getLogger()
logger.setLevel('INFO')

def init_model():
    model_info_dict = exec_init_model()
    logging.info('[hunmin log] the end line of the function [init_model]')
    return { **model_info_dict }


def inference_dataframe(input_data, model_info_dict):
    result = exec_inference_dataframe(input_data, model_info_dict)
    logging.info('[hunmin log] the end line of the function [inference]')
    return result


def inference_file(files, model_info_dict):
    result = exec_inference_file(files, model_info_dict)
    logging.info('[hunmin log] the end line of the function [inference_file]')
    return result