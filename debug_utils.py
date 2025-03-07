import logging
import torch
import datetime
import os
DEBUG = False
DEBUG_CHECK = False
log_path = f"./training_log/"
os.makedirs(log_path, exist_ok=True)
# 配置日志  log_path+'train.log
current_time = datetime.datetime.now()
formatted_time = current_time.strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(
    filename=f"{log_path}/{formatted_time}_DTPP_train_debug.log",  # 输出日志的文件名  
    level=logging.DEBUG,   # 设置日志级别为 DEBUG  
    # format='%(asctime)s - %(levelname)s - %(message)s'  # 日志格式  
    format='%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s'  
)

# 添加一个 StreamHandler，将日志信息输出到控制台(终端）。
logging.getLogger().addHandler(logging.StreamHandler())
# 设置 PyTorch 打印选项，确保张量内容不会被省略  
torch.set_printoptions(threshold=torch.inf)
# 输出调试信息  
logging.debug("This is a debug message")  
logging.info("This is an info message")  
logging.warning("This is a warning message")  
logging.error("This is an error message")  
logging.critical("This is a critical message")  


def check_tensor(tensor, name, is_open=True): 
    if is_open: 
        logging.debug(f"{name} contains NaN: {torch.isnan(tensor).any()}")  
        logging.debug(f"{name} contains Inf: {torch.isinf(tensor).any()}")  
        logging.debug(f"{name} min: {tensor.min()}, max: {tensor.max()}")

def print_tensor_log(tensor, name, is_open = True):
    if is_open:
        if torch.is_tensor(tensor):
            logging.debug(f"tensor {name} is : {tensor}")
        else:
            logging.debug(f"not is tensor {name} is : {tensor}")
            

def print_tensor_shape_log(tensor, name, is_open = True):
    if is_open:
        logging.debug(f"tensor {name} shape is : {tensor.shape}")
