#实现一个文件夹中（目录中可能还有子目录），拷贝所有的py文件到另一个指定的目录中。
import os

def copy_dir(source_dir, destination_dir):
    """
    copy source_dir 中所有的.py文件到 destination_dir 的目录中去
    :param source_dir:原始目录
    :param destination_dir:目标目录
    :return:返回一共拷贝的文件数量
    """
    count = 0
    #把文件名和目录拼凑成一个完整的绝对路径
    for f in os.listdir(source_dir):
        f_path = os.path.join(source_dir, f)
        if os.path.isfile(f_path) and f.endswith('.py'):
            #要拷贝该文件
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            #拼凑一个拷贝之后的目标文件绝对路线
            sink_path = os.path.join(destination_dir, f)
            # shutil.copy(f_path, sink_path)
            #拷贝文件内容到sink_path中
            num = copy_file(f_path, sink_path)
            count += num
        elif os.path.isdir(f_path):
            # 采用递归函数
            # 为了保持同样的目录结构，目标目录也要跟着变化
            new_destination_path = os.path.join(destination_dir, f)
            copy_dir(f_path, new_destination_path)
    return count

def copy_file(source_file, sink_file):
    """
    copy source_file 到 sink_file 中
    :param source_file: 原始文件的绝对路径
    :param sink_file: 目标文件的绝对路径
    :return: 没有返回值
    """
    #第一种：考虑到文件都是小文件：可以一次性读取全部文件内容，并一次性写入新文件
    # with open(source_file, mode='r+', encoding='UTF-8') as source_f:
    #     content = source_f.read()
    #     with open(sink_file, mode='w+', encoding='UTF-8') as sink_f:
    #         sink_f.write(content)

    #第二张：考虑到文件比较大，每次从源文件中读取一部分内容，并且写入到新文件（循环多次）
    source_f = open(source_file, mode='r+', encoding='UTF-8')
    sink_f = open(sink_file, mode='w+', encoding='UTF-8')
    while True:
        line = source_f.readline(1024*10)
        if line == "" or line is None:
            break
        sink_f.write(line)
    source_f.close()
    sink_f.close()
    return 1

copy_dir(r'/Users/eric/Documents/PycharmProjects/PythonProject', r'/Users/eric/Documents/PycharmProjects/PythonProject1')