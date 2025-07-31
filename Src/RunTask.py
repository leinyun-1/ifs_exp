#!/usr/bin/env python3

"""
created by Allan Yu, 23-11-20
专门用于学习任务规划的文件，在算法设计阶段，命令行方式并不是一个很适合的形式，
函数形式的任务，在算法定型之后，改为命令行参数非常容易。

这部分代码本来和Trainer放在一起的，但我在使用时，常常要在Trainer成员函数，
和这些任务之间切换，定位代码行，因此索性分离它们。

"""


from Trainer.Trainer import Trainer
from TaskIFS import task
#from TaskPiFu import task 

if __name__ == "__main__":
    Trainer("ifs", description="refactoring version", taskinfo=task()).train()
