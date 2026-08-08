"""writeup 生图脚本公共路径工具。
所有脚本从 writeup/scripts/ 里运行(或任何目录), 自动定位项目根 + 输出到 writeup/。
需要 train_tx 的脚本(aug/inversion)会 chdir 到项目根再 import(train_tx 按项目根解析数据路径)。"""
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))          # writeup/scripts -> writeup -> 项目根
OUT  = os.path.join(ROOT, "writeup")                    # 图输出目录

def use_train_tx():
    """给需要 train_tx 的脚本: 把项目根加进 path + chdir + 挡 argparse, 返回 train_tx 模块。"""
    sys.path.insert(0, ROOT)
    os.chdir(ROOT)
    sys.argv = ["x"]                                    # train_tx main() 用 argparse, import 时挡掉
    import train_tx as T
    return T
