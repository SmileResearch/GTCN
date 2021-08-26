import time
import os
from code_completion import CodeCompletion

def run(args=None):

    """整体流程：
            1.获得model和task类。model和task分开来是为了方便复用，满足高类聚，低耦合。
            2.获取参数，并初始化task。task导入数据。
                2.1task启动多线程读取数据。。
            3.将task传入model，并初始化model
                3.1期间model调用make_model()初始化模型。模型调用了combine_layer作为中间层，同时创建embedding层和输出层。combine_layer调用gcn、ggnn。
                3.2根据参数选择是否load_data（即load weight和优化器参数） ，并设置优化器等参数
            4.开始训练 运行model的train()函数：
                4.1根据train或test，将数据分别送入__run_epoch, __run_epoch负责每一个epoch的训练。
                4.1调用task函数:make_minibatch_iterator，将小图处理成大图。
                4.2调用task函数:make_task_input 处理输入，比如将边组装成密集图。
                4.3进行训练，获得结果
                4.4送入task criterion函数获取loss、metrics
                4.5train函数输出结果。
    """ 

    cc_cls = CodeCompletion

    parser = cc_cls.default_args()
    args = parser.parse_args()
    cc = cc_cls(args)
    cc.train()
    # GCN model and task
    
 


if __name__ == "__main__":
    run()