# coding: utf-8
from data.combine import load_raw_data
from src.pre_data import transfer_num, prepare_data, prepare_train_batch, get_single_example_graph
from src.train_and_evaluate import *
from src.models import *

import time
import torch.optim
from src.expressions_transfer import *
import json

'''
旨在从指定路径读取一个 JSON 文件，并将其内容以 Python 数据结构返回，
通常是字典或列表，这取决于 JSON 文件的结构。
'''

def read_json(path):
    with open(path,'r') as f:
        file = json.load(f)
    return file

# def read_json(path, percentage=5):
#     with open(path, 'r') as f:
#         # 读取整个 JSON 文件
#         data = json.load(f)
#
#         # 计算应该读取的数据量
#         total_records = len(data)
#         num_records_to_read = int(total_records * (percentage / 100))
#
#         # 从数据中随机选择要读取的记录
#         random_records = random.sample(data, num_records_to_read)
#
#         return random_records

# batch_size = 64
# batch_size = 32 #手动修改
batch_size = 16 #手动修改
embedding_size = 128    # 使用了一个128个单元的单词嵌入（未预先训练）
hidden_size = 512       # 所有其他层的隐藏状态的维度设置为512
n_epochs = 80
learning_rate = 1e-3
weight_decay = 1e-5
beam_size = 5
n_layers = 2
ori_path = './data/'
prefix = '23k_processed.json'

'''
get_train_test_fold()函数的设计
目的是根据预定义的 JSON 文件中的 ID 将数据集分割成训练集、验证集和测试集。
它通过读取训练集、验证集和测试集的独立文件中的数据集 ID，
然后根据这些 ID 将给定的数据（data、pairs、group）相应地组织到这三个集合中。
'''
def get_train_test_fold(ori_path,prefix,data,pairs,group):
    # 1、定义文件路径：函数使用基础路径 ori_path、模式（train, valid, test）和
    # 前缀构建训练集、验证集和测试集的文件路径。
    mode_train = 'train'
    mode_valid = 'valid'
    mode_test = 'test'
    train_path = ori_path + mode_train + prefix
    valid_path = ori_path + mode_valid + prefix
    test_path = ori_path + mode_test + prefix

    # 2、读取 JSON 文件：它读取这些路径上的 JSON 文件，以获取每个数据集的 ID 列表。
    train = read_json(train_path)
    train_id = [item['id'] for item in train]
    valid = read_json(valid_path)
    valid_id = [item['id'] for item in valid]
    test = read_json(test_path)
    test_id = [item['id'] for item in test]

    # 3、初始化折叠列表：初始化三个列表，用于存储每个数据集的数据。
    train_fold = []
    valid_fold = []
    test_fold = []

    # 4、分配数据：函数遍历提供的 data、pairs 和 group。对于每个项，它根据其 ID 确定该项属于哪个数据集。
    # 然后，它将相应的数据（pair 和 group 信息的组合）添加到适当的折叠列表中。
    for item,pair,g in zip(data, pairs, group):
        pair = list(pair)
        pair.append(g['group_num'])
        pair = tuple(pair)
        if item['id'] in train_id:
            train_fold.append(pair)
        elif item['id'] in test_id:
            test_fold.append(pair)
        else:
            valid_fold.append(pair)

    # 5、返回数据：最后，函数返回三个折叠列表，这些列表包含了为训练集、验证集和测试集组织好的数据。
    return train_fold, test_fold, valid_fold

'''
将一个字符串列表 num 中的每个元素转换成浮点数。
这些字符串可能表示分数、百分比或普通数字。
'''
def change_num(num):
    new_num = []
    for item in num:
        # 1、处理分数：如果字符串中包含 '/'，函数会解析这个分数。
        # 它首先去掉右括号后的内容，然后提取括号内的字符串，这个字符串应该是一个分数形式（如 "1/2"）。
        # 然后，它将这个分数分割成分子和分母，并计算其值。
        if '/' in item:
            new_str = item.split(')')[0]
            new_str = new_str.split('(')[1]
            a = float(new_str.split('/')[0])
            b = float(new_str.split('/')[1])
            value = a/b
            # 将value添加到new_num列表中。
            new_num.append(value)
        # 2、处理百分比：如果字符串中包含 '%'，函数会将这个百分比转换成对应的小数。
        # 它去掉百分号，然后将剩余的数字转换为浮点数，并除以 100。
        elif '%' in item:
            value = float(item[0:-1])/100
            new_num.append(value)
        # 3、处理普通数字：如果字符串既不包含 '/' 也不包含 '%'，
        # 函数将其视为一个普通数字并直接转换为浮点数。
        else:
            new_num.append(float(item))
    return new_num

# data = load_raw_data("data/Math_23K.json")

data = load_raw_data("data/Math_23K.json")
group_data = read_json("data/Math_23K_processed.json")

# 2024.3.14/22.25
# 用于存储处理后的数据 | 用于存储筛选后的生成数字 | 记录最大的数字数量
pairs, generate_nums, copy_nums = transfer_num(data)

temp_pairs = []     # 用于存储更新后的数据对
for p in pairs:     # 每个数据对包含处理过的问题文本、方程、数值列表和数值位置列表
    # 将方程 p[1] 从中缀表示法转换为前缀表示法
    temp_pairs.append((p[0], from_infix_to_prefix(p[1]), p[2], p[3]))
pairs = temp_pairs

train_fold, test_fold, valid_fold = get_train_test_fold(ori_path,prefix,data,pairs,group_data)


best_acc_fold = []

pairs_tested = test_fold
#pairs_trained = valid_fold
pairs_trained = train_fold

'''
这种方法通常用于交叉验证，一个常见的模型评估方法，可以有效地利用有限的数据来评估模型的性能。
每一次迭代中，不同的数据折被轮流用作测试集，其余的用作训练集，
从而确保每个数据点都被用作了测试，且模型能在多个训练集上进行训练和验证。
'''
# for fold_t in range(5):
#    if fold_t == fold:
#        # pairs_tested 将包含单独一折的数据对作为测试集
#        pairs_tested += fold_pairs[fold_t]
#    else:
#        # pairs_trained 将包含其他四折的数据对作为训练集
#        pairs_trained += fold_pairs[fold_t]

# 用于处理输入的词汇 | 用于处理输出的词汇 | 训练数据对 | 测试数据对
input_lang, output_lang, train_pairs, test_pairs = prepare_data(pairs_trained, pairs_tested, 5, generate_nums,
                                                                copy_nums, tree=True)

#print('train_pairs[0]')
#print(train_pairs[0])
#exit()
#Initialize models
encoder = EncoderSeq(input_size=input_lang.n_words, embedding_size=embedding_size, hidden_size=hidden_size,
                     n_layers=n_layers)
predict = Prediction(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                     input_size=len(generate_nums))
generate = GenerateNode(hidden_size=hidden_size, op_nums=output_lang.n_words - copy_nums - 1 - len(generate_nums),
                        embedding_size=embedding_size)
merge = Merge(hidden_size=hidden_size, embedding_size=embedding_size)
# the embedding layer is  only for generated number embeddings, operators, and paddings

# encoder_optimizer：这是优化器的变量名。
# torch.optim.Adam：这指定使用Adam优化器。
# encoder.parameters()：指定了encoder的参数应该被优化。在PyTorch中，parameters()函数返回模型参数（权重和偏差）的生成器。
# lr=learning_rate：学习率是一个超参数，控制每次更新模型权重时对估计误差的响应程度。选择合适的学习率对模型收敛至解决方案至关重要。
# weight_decay=weight_decay：这给优化过程添加了L2正则化。权重衰减是一个正则化项，有助于防止权重过大，
# 可以帮助防止过拟合。本质上，它在损失函数中添加了对权重大小的惩罚。
encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate, weight_decay=weight_decay)
predict_optimizer = torch.optim.Adam(predict.parameters(), lr=learning_rate, weight_decay=weight_decay)
generate_optimizer = torch.optim.Adam(generate.parameters(), lr=learning_rate, weight_decay=weight_decay)
merge_optimizer = torch.optim.Adam(merge.parameters(), lr=learning_rate, weight_decay=weight_decay)

# encoder_scheduler：这是学习率调度器的变量名。它被用来调整 encoder_optimizer 的学习率，
# encoder_optimizer 是之前定义的，用于优化编码器模型参数的优化器。

# torch.optim.lr_scheduler.StepLR：这指定了学习率调度器的类型为 StepLR，这是一种每隔固定的训练步数降低学习率的调度器。
# StepLR 让学习率在每个 step_size 周期后乘以一个因子 gamma，从而实现逐步减小学习率。

# step_size=20：这个参数定义了学习率变化的周期。
# 具体来说，每过 20 个训练周期，学习率将按照 gamma 指定的比例进行调整。

# gamma=0.5：这是学习率衰减的因子。
# 在这个例子中，每经过 step_size 指定的周期数，学习率会乘以 0.5，即学习率减半。
encoder_scheduler = torch.optim.lr_scheduler.StepLR(encoder_optimizer, step_size=20, gamma=0.5)
predict_scheduler = torch.optim.lr_scheduler.StepLR(predict_optimizer, step_size=20, gamma=0.5)
generate_scheduler = torch.optim.lr_scheduler.StepLR(generate_optimizer, step_size=20, gamma=0.5)
merge_scheduler = torch.optim.lr_scheduler.StepLR(merge_optimizer, step_size=20, gamma=0.5)

# Move models to GPU
# .cuda() 方法是 PyTorch 中的一个函数，用于将模型的所有参数和缓冲区移动到 GPU 内存中。
if USE_CUDA:
    encoder.cuda()  # 将 encoder 模型移动到 GPU 上。
    predict.cuda()  # 将 predict 模型移动到 GPU 上。
    generate.cuda() # 将 generate 模型移动到 GPU 上。
    merge.cuda()    # 将 merge 模型移动到 GPU 上。

generate_num_ids = []       # 用于存储从 generate_nums 转换而来的索引。
for num in generate_nums:   # 每个元素 num 代表 generate_nums 列表中的一个单词或符号。
    generate_num_ids.append(output_lang.word2index[num])    # 这部分代码查找 num 在 output_lang 的词汇表中对应的索引。

for epoch in range(n_epochs):
    # 学习率调度器更新,这有助于在训练过程中调整学习率。
    encoder_scheduler.step()
    predict_scheduler.step()
    generate_scheduler.step()
    merge_scheduler.step()

    loss_total = 0      # 在每个训练周期开始时初始化总损失为0。

    # 数据批处理：通过 prepare_train_batch 函数准备训练批次。
    # 这个函数返回一系列批次，每个批次包含多个训练样本。
    input_batches, input_lengths, output_batches, output_lengths, nums_batches, \
   num_stack_batches, num_pos_batches, num_size_batches, num_value_batches, graph_batches = prepare_train_batch(train_pairs, batch_size)
    print("epoch:", epoch + 1)
    # 初始化时间计数
    start = time.time()
    # 监控每个训练周期的平均损失和训练时间:
    for idx in range(len(input_lengths)):
        # 执行训练步骤
        # train_tree 函数是树形模型训练步骤的详细实现，这种模型可能用于数学方程求解等任务，其中输出结构为树形。
        # 该函数封装了前向传播、损失计算和反向传播。
        loss = train_tree(
            input_batches[idx], input_lengths[idx], output_batches[idx], output_lengths[idx],
            num_stack_batches[idx], num_size_batches[idx], generate_num_ids, encoder, predict, generate, merge,
            encoder_optimizer, predict_optimizer, generate_optimizer, merge_optimizer, output_lang, num_pos_batches[idx], graph_batches[idx])
        loss_total += loss      # 损失累加

    print("loss:", loss_total / len(input_lengths))              # 打印平均损失
    print("training time:", time_since(time.time() - start))     # 计算并打印训练时间
    print("--------------------------------")

    '''
        这段代码是在模型训练循环中，对模型进行定期评估和保存的部分。
        具体来说，它在每两个训练周期或在最后五个训练周期内对模型进行评估，并保存模型的状态。
    '''
    # 1、定期评估：每两个周期评估一次，以及在训练的最后五个周期内每个周期都进行评估。
    if epoch % 2 == 0 or epoch > n_epochs - 5:
        # 2、初始化计数器：value_ac、equation_ac 和 eval_total
        # 分别用于累计值的准确率、方程的准确率和评估的总数。
        value_ac = 0
        equation_ac = 0
        eval_total = 0
        start = time.time()
        # 3、评估循环：
        # 通过遍历 test_pairs（测试数据对），对每个测试实例进行评估。
        # 这涉及构建图（batch_graph），执行模型评估（evaluate_tree），并计算结果的准确性（compute_prefix_tree_result）。
        for test_batch in test_pairs:
            #print(test_batch)
            batch_graph = get_single_example_graph(test_batch[0], test_batch[1], test_batch[7], test_batch[4], test_batch[5])
            test_res = evaluate_tree(test_batch[0], test_batch[1], generate_num_ids, encoder, predict, generate,
                                     merge, output_lang, test_batch[5], batch_graph, beam_size=beam_size)
            val_ac, equ_ac, _, _ = compute_prefix_tree_result(test_res, test_batch[2], output_lang, test_batch[4], test_batch[6])
            if val_ac:
                value_ac += 1
            if equ_ac:
                equation_ac += 1
            eval_total += 1
        # 4、准确率计算和输出：在评估所有测试实例后，计算并打印出值准确率和方程准确率。
        # 此外，还会打印出整个评估过程所花费的时间。
        print(equation_ac, value_ac, eval_total)
        print("test_answer_acc", float(equation_ac) / eval_total, float(value_ac) / eval_total)
        print("testing time", time_since(time.time() - start))
        print("------------------------------------------------------")
        # 5、模型保存：在每次评估后，使用 torch.save 保存当前的模型状态，包括编码器、预测器、生成器和合并器的状态。
        # 这样可以在训练过程中定期保存模型的快照，有助于后续的模型恢复和进一步分析。
        torch.save(encoder.state_dict(), "model_traintest/encoder")
        torch.save(predict.state_dict(), "model_traintest/predict")
        torch.save(generate.state_dict(), "model_traintest/generate")
        torch.save(merge.state_dict(), "model_traintest/merge")
        # 6、最佳准确率记录：在训练的最后一个周期，
        # 将该周期的评估结果添加到 best_acc_fold 列表中，用于记录训练过程中的最佳准确率。
        if epoch == n_epochs - 1:
            best_acc_fold.append((equation_ac, value_ac, eval_total))

# 1、初始化累加器：变量 a、b 和 c 被初始化为0。
# 它们将分别用于累加方程准确率（a）、值准确率（b）和评估总数（c）。
a, b, c = 0, 0, 0
for bl in range(len(best_acc_fold)):
    # 2、累加过程：通过一个循环遍历 best_acc_fold 中的每一个元素（每一折的结果）。
    # 对于每一个元素，增加 a、b 和 c 的值，分别对应于当前折的方程准确率、值准确率和评估总数。
    a += best_acc_fold[bl][0]
    b += best_acc_fold[bl][1]
    c += best_acc_fold[bl][2]
    # 3、打印每一折的结果：在循环中打印出 best_acc_fold 的每一个元素，即每一折的具体准确率。
    print(best_acc_fold[bl])
# 4、计算并打印总体准确率：循环结束后，计算总的方程准确率（a / float(c)）
# 和值准确率（b / float(c)），然后打印出这两个总体准确率。
print(a / float(c), b / float(c))

