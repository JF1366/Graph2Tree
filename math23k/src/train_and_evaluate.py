from math23k.src.masked_cross_entropy import masked_cross_entropy
from math23k.src.pre_data import PAD_token
from math23k.src.expressions_transfer import *
from math23k.src.models import *
import math
import torch
import torch.optim
import torch.nn.functional as f
import random
import time

import sys
sys.path.append("E:\MySelf_WorkCodes\PyCharm\Research_direction_code\Graph2Tree-master\math23k\src")

MAX_OUTPUT_LENGTH = 45
MAX_INPUT_LENGTH = 120

# 这个函数检查当前环境是否支持 CUDA。
# 它主要检查两件事：一是是否安装了 CUDA，二是是否有可用的支持 CUDA 的 GPU。
USE_CUDA = torch.cuda.is_available()



class Beam:  # the class save the beam node
    def __init__(self, score, input_var, hidden, all_output):
        self.score = score
        self.input_var = input_var
        self.hidden = hidden
        self.all_output = all_output


def time_since(s):  # compute time
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    return '%dh %dm %ds' % (h, m, s)


def generate_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums, generate_nums,
                       english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in generate_nums:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index[")"], word2index["+"], word2index["-"],
                        word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + [word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + \
                      [word2index["["], word2index["("]] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] == word2index["["] or decoder_input[i] == word2index["("]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["("]] + generate_nums
            elif decoder_input[i] == word2index[")"]:
                res += [word2index["]"], word2index[")"], word2index["+"],
                        word2index["-"], word2index["/"], word2index["^"],
                        word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["]"]:
                res += [word2index["+"], word2index["*"], word2index["-"], word2index["/"], word2index["EOS"]]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"],
                                      word2index["*"], word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] +\
                  [word2index["["], word2index["("]] + generate_nums
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_pre_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                    generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                      [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"],
                        word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_post_tree_seq_rule_mask(decoder_input, nums_batch, word2index, batch_size, nums_start, copy_nums,
                                     generate_nums, english):
    rule_mask = torch.FloatTensor(batch_size, nums_start + copy_nums).fill_(-float("1e12"))
    if english:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums +\
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    else:
        if decoder_input[0] == word2index["SOS"]:
            for i in range(batch_size):
                res = [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums
                for j in res:
                    rule_mask[i, j] = 0
            return rule_mask
        for i in range(batch_size):
            res = []
            if decoder_input[i] >= nums_start or decoder_input[i] in generate_nums:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"]
                        ]
            elif decoder_input[i] == word2index["EOS"] or decoder_input[i] == PAD_token:
                res += [PAD_token]
            elif decoder_input[i] in [word2index["+"], word2index["-"], word2index["/"], word2index["*"],
                                      word2index["^"]]:
                res += [_ for _ in range(nums_start, nums_start + nums_batch[i])] + generate_nums + \
                       [word2index["+"], word2index["-"], word2index["/"], word2index["*"], word2index["^"],
                        word2index["EOS"]
                        ]
            for j in res:
                rule_mask[i, j] = 0
    return rule_mask


def generate_tree_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    target_input = copy.deepcopy(target)
    for i in range(len(target)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
        if target_input[i] >= num_start:
            target_input[i] = 0
    return torch.LongTensor(target), torch.LongTensor(target_input)


def generate_decoder_input(target, decoder_output, nums_stack_batch, num_start, unk):
    # when the decoder input is copied num but the num has two pos, chose the max
    if USE_CUDA:
        decoder_output = decoder_output.cpu()
    for i in range(target.size(0)):
        if target[i] == unk:
            num_stack = nums_stack_batch[i].pop()
            max_score = -float("1e12")
            for num in num_stack:
                if decoder_output[i, num_start + num] > max_score:
                    target[i] = num + num_start
                    max_score = decoder_output[i, num_start + num]
    return target


def mask_num(encoder_outputs, decoder_input, embedding_size, nums_start, copy_nums, num_pos):
    # mask the decoder input number and return the mask tensor and the encoder position Hidden vector
    up_num_start = decoder_input >= nums_start
    down_num_end = decoder_input < (nums_start + copy_nums)
    num_mask = up_num_start == down_num_end
    num_mask_encoder = num_mask < 1
    num_mask_encoder = num_mask_encoder.unsqueeze(1)  # ByteTensor size: B x 1
    repeat_dims = [1] * num_mask_encoder.dim()
    repeat_dims[1] = embedding_size
    num_mask_encoder = num_mask_encoder.repeat(*repeat_dims)  # B x 1 -> B x Decoder_embedding_size

    all_embedding = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_embedding.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    indices = decoder_input - nums_start
    indices = indices * num_mask.long()  # 0 or the num pos in sentence
    indices = indices.tolist()
    for k in range(len(indices)):
        indices[k] = num_pos[k][indices[k]]
    indices = torch.LongTensor(indices)
    if USE_CUDA:
        indices = indices.cuda()
    batch_size = decoder_input.size(0)
    sen_len = encoder_outputs.size(0)
    batch_num = torch.LongTensor(range(batch_size))
    batch_num = batch_num * sen_len
    if USE_CUDA:
        batch_num = batch_num.cuda()
    indices = batch_num + indices
    num_encoder = all_embedding.index_select(0, indices)
    return num_mask, num_encoder, num_mask_encoder


def out_equation(test, output_lang, num_list, num_stack=None):
    test = test[:-1]
    max_index = len(output_lang.index2word) - 1
    test_str = ""
    for i in test:
        if i < max_index:
            c = output_lang.index2word[i]
            if c == "^":
                test_str += "**"
            elif c == "[":
                test_str += "("
            elif c == "]":
                test_str += ")"
            elif c[0] == "N":
                if int(c[1:]) >= len(num_list):
                    return None
                x = num_list[int(c[1:])]
                if x[-1] == "%":
                    test_str += "(" + x[:-1] + "/100" + ")"
                else:
                    test_str += x
            else:
                test_str += c
        else:
            if len(num_stack) == 0:
                print(test_str, num_list)
                return ""
            n_pos = num_stack.pop()
            test_str += num_list[n_pos[0]]
    return test_str

'''
compute_prefix_tree_result 函数用于计算并比较模型生成的表达式和目标表达式的正确性。
这个函数对于评估模型在解决如数学问题求解等需要输出结构化表达式的任务中的表现非常关键。
'''
def compute_prefix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    # 1、直接比较：首先检查如果没有数值堆栈（num_stack 为空）
    # 且模型输出（test_res）与目标输出（test_tar）直接相等，那么返回正确的结果。
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar

    # 2、表达式解析：使用 out_expression_list 函数将模型输出和目标输出从序列形式转换为数学表达式。
    # 这涉及将输出序列中的符号和数字映射回其代表的数学意义。
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)

    # 3、比较表达式：
    # 如果转换后的模型输出表达式（test）与目标表达式（tar）完全一致，表明模型完全正确地生成了目标表达式。
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    # 4、数值计算和比较：如果表达式不完全相同，进一步通过 compute_prefix_expression 函数
    # 计算两个表达式的数值结果，并比较这些结果。
    # 如果结果相差小于一定阈值（如 1e-4），则认为模型在数值上正确解决了问题，但表达式形式上有误差。
    try:
        if abs(compute_prefix_expression(test) - compute_prefix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    # 5、异常处理：在计算表达式数值时可能会出现异常（例如，除以零），此时捕获异常并返回表达式计算结果为不正确。
    except:
        return False, False, test, tar


def compute_postfix_tree_result(test_res, test_tar, output_lang, num_list, num_stack):
    # print(test_res, test_tar)

    if len(num_stack) == 0 and test_res == test_tar:
        return True, True, test_res, test_tar
    test = out_expression_list(test_res, output_lang, num_list)
    tar = out_expression_list(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    # print(test, tar)
    if test is None:
        return False, False, test, tar
    if test == tar:
        return True, True, test, tar
    try:
        if abs(compute_postfix_expression(test) - compute_postfix_expression(tar)) < 1e-4:
            return True, False, test, tar
        else:
            return False, False, test, tar
    except:
        return False, False, test, tar


def compute_result(test_res, test_tar, output_lang, num_list, num_stack):
    if len(num_stack) == 0 and test_res == test_tar:
        return True, True
    test = out_equation(test_res, output_lang, num_list)
    tar = out_equation(test_tar, output_lang, num_list, copy.deepcopy(num_stack))
    if test is None:
        return False, False
    if test == tar:
        return True, True
    try:
        if abs(eval(test) - eval(tar)) < 1e-4:
            return True, False
        else:
            return False, False
    except:
        return False, False


def get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size, hidden_size):
    indices = list()
    sen_len = encoder_outputs.size(0)
    masked_index = []
    temp_1 = [1 for _ in range(hidden_size)]
    temp_0 = [0 for _ in range(hidden_size)]
    for b in range(batch_size):
        for i in num_pos[b]:
            indices.append(i + b * sen_len)
            masked_index.append(temp_0)
        indices += [0 for _ in range(len(num_pos[b]), num_size)]
        masked_index += [temp_1 for _ in range(len(num_pos[b]), num_size)]
    indices = torch.LongTensor(indices)
    masked_index = torch.ByteTensor(masked_index)
    masked_index = masked_index.view(batch_size, num_size, hidden_size)
    if USE_CUDA:
        indices = indices.cuda()
        masked_index = masked_index.cuda()
    all_outputs = encoder_outputs.transpose(0, 1).contiguous()
    all_embedding = all_outputs.view(-1, encoder_outputs.size(2))  # S x B x H -> (B x S) x H
    all_num = all_embedding.index_select(0, indices)
    all_num = all_num.view(batch_size, num_size, hidden_size)
    return all_num.masked_fill_(masked_index, 0.0)


def train_attn(input_batch, input_length, target_batch, target_length, num_batch, nums_stack_batch, copy_nums,
               generate_nums, encoder, decoder, encoder_optimizer, decoder_optimizer, output_lang, clip=0,
               use_teacher_forcing=1, beam_size=1, english=False):
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_start = output_lang.n_words - copy_nums - 2
    unk = output_lang.word2index["UNK"]
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)
    target = torch.LongTensor(target_batch).transpose(0, 1)

    batch_size = len(input_length)

    encoder.train()
    decoder.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, input_length, None)

    # Prepare input and output variables
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]] * batch_size)

    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder

    max_target_length = max(target_length)
    all_decoder_outputs = torch.zeros(max_target_length, batch_size, decoder.output_size)

    # Move new Variables to CUDA
    if USE_CUDA:
        all_decoder_outputs = all_decoder_outputs.cuda()

    if random.random() < use_teacher_forcing:
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            if USE_CUDA:
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            all_decoder_outputs[t] = decoder_output
            decoder_input = generate_decoder_input(
                target[t], decoder_output, nums_stack_batch, num_start, unk)
            target[t] = decoder_input
    else:
        beam_list = list()
        score = torch.zeros(batch_size)
        if USE_CUDA:
            score = score.cuda()
        beam_list.append(Beam(score, decoder_input, decoder_hidden, all_decoder_outputs))
        # Run through decoder one time step at a time
        for t in range(max_target_length):
            beam_len = len(beam_list)
            beam_scores = torch.zeros(batch_size, decoder.output_size * beam_len)
            all_hidden = torch.zeros(decoder_hidden.size(0), batch_size * beam_len, decoder_hidden.size(2))
            all_outputs = torch.zeros(max_target_length, batch_size * beam_len, decoder.output_size)
            if USE_CUDA:
                beam_scores = beam_scores.cuda()
                all_hidden = all_hidden.cuda()
                all_outputs = all_outputs.cuda()

            for b_idx in range(len(beam_list)):
                decoder_input = beam_list[b_idx].input_var
                decoder_hidden = beam_list[b_idx].hidden

                rule_mask = generate_rule_mask(decoder_input, num_batch, output_lang.word2index, batch_size,
                                               num_start, copy_nums, generate_nums, english)
                if USE_CUDA:
                    rule_mask = rule_mask.cuda()
                    decoder_input = decoder_input.cuda()

                decoder_output, decoder_hidden = decoder(
                    decoder_input, decoder_hidden, encoder_outputs, seq_mask)

                score = f.log_softmax(decoder_output, dim=1) + rule_mask
                beam_score = beam_list[b_idx].score
                beam_score = beam_score.unsqueeze(1)
                repeat_dims = [1] * beam_score.dim()
                repeat_dims[1] = score.size(1)
                beam_score = beam_score.repeat(*repeat_dims)
                score += beam_score
                beam_scores[:, b_idx * decoder.output_size: (b_idx + 1) * decoder.output_size] = score
                all_hidden[:, b_idx * batch_size:(b_idx + 1) * batch_size, :] = decoder_hidden

                beam_list[b_idx].all_output[t] = decoder_output
                all_outputs[:, batch_size * b_idx: batch_size * (b_idx + 1), :] = \
                    beam_list[b_idx].all_output
            topv, topi = beam_scores.topk(beam_size, dim=1)
            beam_list = list()

            for k in range(beam_size):
                temp_topk = topi[:, k]
                temp_input = temp_topk % decoder.output_size
                temp_input = temp_input.data
                if USE_CUDA:
                    temp_input = temp_input.cpu()
                temp_beam_pos = temp_topk / decoder.output_size

                indices = torch.LongTensor(range(batch_size))
                if USE_CUDA:
                    indices = indices.cuda()
                indices += temp_beam_pos * batch_size

                temp_hidden = all_hidden.index_select(1, indices)
                temp_output = all_outputs.index_select(1, indices)

                beam_list.append(Beam(topv[:, k], temp_input, temp_hidden, temp_output))
        all_decoder_outputs = beam_list[0].all_output

        for t in range(max_target_length):
            target[t] = generate_decoder_input(
                target[t], all_decoder_outputs[t], nums_stack_batch, num_start, unk)
    # Loss calculation and backpropagation

    if USE_CUDA:
        target = target.cuda()


    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(),  # -> batch x seq
        target.transpose(0, 1).contiguous(),  # -> batch x seq
        target_length
    )

    loss.backward()
    return_loss = loss.item()

    # Clip gradient norms
    if clip:
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip)
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return return_loss


def evaluate_attn(input_seq, input_length, num_list, copy_nums, generate_nums, encoder, decoder, output_lang,
                  beam_size=1, english=False, max_length=MAX_OUTPUT_LENGTH):
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    num_start = output_lang.n_words - copy_nums - 2

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_seq).unsqueeze(1)
    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()

    # Set to not-training mode to disable dropout
    encoder.eval()
    decoder.eval()

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_var, [input_length], None)

    # Create starting vectors for decoder
    decoder_input = torch.LongTensor([output_lang.word2index["SOS"]])  # SOS
    decoder_hidden = encoder_hidden[:decoder.n_layers]  # Use last (forward) hidden state from encoder
    beam_list = list()
    score = 0
    beam_list.append(Beam(score, decoder_input, decoder_hidden, []))

    # Run through decoder
    for di in range(max_length):
        temp_list = list()
        beam_len = len(beam_list)
        for xb in beam_list:
            if int(xb.input_var[0]) == output_lang.word2index["EOS"]:
                temp_list.append(xb)
                beam_len -= 1
        if beam_len == 0:
            return beam_list[0].all_output
        beam_scores = torch.zeros(decoder.output_size * beam_len)
        hidden_size_0 = decoder_hidden.size(0)
        hidden_size_2 = decoder_hidden.size(2)
        all_hidden = torch.zeros(beam_len, hidden_size_0, 1, hidden_size_2)
        if USE_CUDA:
            beam_scores = beam_scores.cuda()
            all_hidden = all_hidden.cuda()
        all_outputs = []
        current_idx = -1

        for b_idx in range(len(beam_list)):
            decoder_input = beam_list[b_idx].input_var
            if int(decoder_input[0]) == output_lang.word2index["EOS"]:
                continue
            current_idx += 1
            decoder_hidden = beam_list[b_idx].hidden

            # rule_mask = generate_rule_mask(decoder_input, [num_list], output_lang.word2index,
            #                                1, num_start, copy_nums, generate_nums, english)
            if USE_CUDA:
                # rule_mask = rule_mask.cuda()
                decoder_input = decoder_input.cuda()

            decoder_output, decoder_hidden = decoder(
                decoder_input, decoder_hidden, encoder_outputs, seq_mask)
            # score = f.log_softmax(decoder_output, dim=1) + rule_mask.squeeze()
            score = f.log_softmax(decoder_output, dim=1)
            score += beam_list[b_idx].score
            beam_scores[current_idx * decoder.output_size: (current_idx + 1) * decoder.output_size] = score
            all_hidden[current_idx] = decoder_hidden
            all_outputs.append(beam_list[b_idx].all_output)
        topv, topi = beam_scores.topk(beam_size)

        for k in range(beam_size):
            word_n = int(topi[k])
            word_input = word_n % decoder.output_size
            temp_input = torch.LongTensor([word_input])
            indices = int(word_n / decoder.output_size)

            temp_hidden = all_hidden[indices]
            temp_output = all_outputs[indices]+[word_input]
            temp_list.append(Beam(float(topv[k]), temp_input, temp_hidden, temp_output))

        temp_list = sorted(temp_list, key=lambda x: x.score, reverse=True)

        if len(temp_list) < beam_size:
            beam_list = temp_list
        else:
            beam_list = temp_list[:beam_size]
    return beam_list[0].all_output


def copy_list(l):
    r = []
    if len(l) == 0:
        return r
    for i in l:
        if type(i) is list:
            r.append(copy_list(i))
        else:
            r.append(i)
    return r


class TreeBeam:  # the class save the beam node
    def __init__(self, score, node_stack, embedding_stack, left_childs, out):
        self.score = score
        self.embedding_stack = copy_list(embedding_stack)
        self.node_stack = copy_list(node_stack)
        self.left_childs = copy_list(left_childs)
        self.out = copy.deepcopy(out)


class TreeEmbedding:  # the class save the tree
    def __init__(self, embedding, terminal=False):
        self.embedding = embedding
        self.terminal = terminal

'''
train_tree 函数是树形模型训练步骤的详细实现，
这种模型可能用于数学方程求解等任务，其中输出结构为树形。
该函数封装了前向传播、损失计算和反向传播。
'''
def train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch, generate_nums,
               encoder, predict, generate, merge, encoder_optimizer, predict_optimizer, generate_optimizer,
               merge_optimizer, output_lang, num_pos, batch_graph, english=False):

    # 1、注意力掩码：函数创建序列掩码（seq_mask）和数字掩码（num_mask）以在模型的前向传播过程中促进注意力机制。
    # 这些掩码帮助模型专注于输入序列的相关部分并忽略填充区域。
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)

    # 2、数据准备：将输入和目标转换为张量并转置，以符合批处理期望的维度。
    # 如果模型包含基于图的信息，则还准备图表示（batch_graph）。
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)
    batch_graph = torch.LongTensor(batch_graph)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)
    # 3、模型训练状态：将编码器、预测器、生成器和合并器模块设置为训练模式，
    # 使得丢弃和批量归一化层在训练模式下功能激活。
    encoder.train()
    predict.train()
    generate.train()
    merge.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()

    # 4、模型前向传播：编码器生成输出和问题特定表示。
    # 这些输出连同节点栈（用问题输出初始化）一起使用，以迭代地预测操作和数字来构建解决方案树。
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    merge_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length, batch_graph)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    # 5、节点处理：在每次迭代中（直到达到最大目标长度），函数预测下一个操作或数字，并相应更新节点栈和嵌入栈。
    # 这个过程包括为树构建生成输入，更新左右子节点的嵌入，并处理终端节点。
    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        left_childs = []
        for idx, l, r, node_stack, i, o in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                               node_stacks, target[t].tolist(), embeddings_stacks):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                left_childs.append(None)
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))
                o.append(TreeEmbedding(node_label[idx].unsqueeze(0), False))
            else:
                current_num = current_nums_embeddings[idx, i - num_start].unsqueeze(0)
                while len(o) > 0 and o[-1].terminal:
                    sub_stree = o.pop()
                    op = o.pop()
                    current_num = merge(op.embedding, sub_stree.embedding, current_num)
                o.append(TreeEmbedding(current_num, True))
            if len(o) > 0 and o[-1].terminal:
                left_childs.append(o[-1].embedding)
            else:
                left_childs.append(None)

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # 6、损失计算：函数使用掩码交叉熵函数计算损失，
    # 该函数只考虑目标序列的非填充部分，有助于准确计算可变长度输出的损失。
    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    # 7、反向传播和优化：函数执行反向传播（loss.backward()）
    # 并使用每个模块的优化器（编码器、预测器、生成器、合并器）更新模型参数。
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    merge_optimizer.step()
    # 8、返回损失：最后，函数返回当前批次的计算损失，可用于监控训练进度。
    return loss.item()  # , loss_0.item(), loss_1.item()

'''
evaluate_tree 函数是用于评估树形模型的函数，特别适用于解决需要输出结构化数据（如数学表达式）的问题。
该函数实现了带有束搜索（beam search）的解码过程，
'''
def evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, merge, output_lang, num_pos, batch_graph, beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    # 1、初始化掩码和变量：使用 torch.ByteTensor 创建序列掩码 (seq_mask) 和数字掩码 (num_mask)。
    # 这些掩码帮助模型在处理输入时忽略无关的部分。
    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)

    # 2、输入变量处理：将输入数据 input_batch 和图结构 batch_graph 转换成适合模型处理的格式（例如，将它们转换成张量）。
    input_var = torch.LongTensor(input_batch).unsqueeze(1)
    batch_graph = torch.LongTensor(batch_graph)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    # 3、模型评估状态：将编码器（encoder）、预测器（predict）、生成器（generate）
    # 和合并器（merge）置于评估模式，这会关闭 dropout，确保在评估过程中模型的行为是确定的。
    encoder.eval()
    predict.eval()
    generate.eval()
    merge.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
        batch_graph = batch_graph.cuda()
    # Run words through encoder

    # 4、前向传播：通过编码器处理输入，获取编码输出和问题特定输出，这些输出随后被用于初始化节点栈。
    encoder_outputs, problem_output = encoder(input_var, [input_length], batch_graph)

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    # 5、初始化束搜索：创建一个包含初始节点栈、嵌入栈
    # 和左子节点的 TreeBeam 对象列表，用于存储在束搜索过程中的候选解。
    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    # 5、束搜索解码：在最大长度 max_length 内迭代地生成新的节点，并使用预测器、生成器
    # 和合并器更新当前的树结构。每一步中，选择得分最高的 beam_size 个候选解作为新的候选。
    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)
            left_childs = b.left_childs

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_left_childs = []
                current_embeddings_stacks = copy_list(b.embedding_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                    current_embeddings_stacks[0].append(TreeEmbedding(node_label[0].unsqueeze(0), False))
                else:
                    current_num = current_nums_embeddings[0, out_token - num_start].unsqueeze(0)

                    while len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                        sub_stree = current_embeddings_stacks[0].pop()
                        op = current_embeddings_stacks[0].pop()
                        current_num = merge(op.embedding, sub_stree.embedding, current_num)
                    current_embeddings_stacks[0].append(TreeEmbedding(current_num, True))
                if len(current_embeddings_stacks[0]) > 0 and current_embeddings_stacks[0][-1].terminal:
                    current_left_childs.append(current_embeddings_stacks[0][-1].embedding)
                else:
                    current_left_childs.append(None)
                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, current_embeddings_stacks,
                                              current_left_childs, current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    # 7、生成输出序列：当所有节点栈为空或达到最大长度时，停止解码过程。
    # 从得分最高的候选解中提取出生成的数学表达式序列。
    return beams[0].out


def topdown_train_tree(input_batch, input_length, target_batch, target_length, nums_stack_batch, num_size_batch,
                       generate_nums, encoder, predict, generate, encoder_optimizer, predict_optimizer,
                       generate_optimizer, output_lang, num_pos, english=False):
    # sequence mask for attention
    seq_mask = []
    max_len = max(input_length)
    for i in input_length:
        seq_mask.append([0 for _ in range(i)] + [1 for _ in range(i, max_len)])
    seq_mask = torch.ByteTensor(seq_mask)

    num_mask = []
    max_num_size = max(num_size_batch) + len(generate_nums)
    for i in num_size_batch:
        d = i + len(generate_nums)
        num_mask.append([0] * d + [1] * (max_num_size - d))
    num_mask = torch.ByteTensor(num_mask)

    unk = output_lang.word2index["UNK"]

    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).transpose(0, 1)

    target = torch.LongTensor(target_batch).transpose(0, 1)

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)
    batch_size = len(input_length)

    encoder.train()
    predict.train()
    generate.train()

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()

    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    predict_optimizer.zero_grad()
    generate_optimizer.zero_grad()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, input_length)
    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    max_target_length = max(target_length)

    all_node_outputs = []
    # all_leafs = []

    copy_num_len = [len(_) for _ in num_pos]
    num_size = max(copy_num_len)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, num_pos, batch_size, num_size,
                                                              encoder.hidden_size)

    num_start = output_lang.num_start
    left_childs = [None for _ in range(batch_size)]
    for t in range(max_target_length):
        num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
            node_stacks, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden, seq_mask, num_mask)

        # all_leafs.append(p_leaf)
        outputs = torch.cat((op, num_score), 1)
        all_node_outputs.append(outputs)

        target_t, generate_input = generate_tree_input(target[t].tolist(), outputs, nums_stack_batch, num_start, unk)
        target[t] = target_t
        if USE_CUDA:
            generate_input = generate_input.cuda()
        left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)
        for idx, l, r, node_stack, i in zip(range(batch_size), left_child.split(1), right_child.split(1),
                                            node_stacks, target[t].tolist()):
            if len(node_stack) != 0:
                node = node_stack.pop()
            else:
                continue

            if i < num_start:
                node_stack.append(TreeNode(r))
                node_stack.append(TreeNode(l, left_flag=True))

    # all_leafs = torch.stack(all_leafs, dim=1)  # B x S x 2
    all_node_outputs = torch.stack(all_node_outputs, dim=1)  # B x S x N

    target = target.transpose(0, 1).contiguous()
    if USE_CUDA:
        # all_leafs = all_leafs.cuda()
        all_node_outputs = all_node_outputs.cuda()
        target = target.cuda()

    # op_target = target < num_start
    # loss_0 = masked_cross_entropy_without_logit(all_leafs, op_target.long(), target_length)
    loss = masked_cross_entropy(all_node_outputs, target, target_length)
    # loss = loss_0 + loss_1
    loss.backward()
    # clip the grad
    # torch.nn.utils.clip_grad_norm_(encoder.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(predict.parameters(), 5)
    # torch.nn.utils.clip_grad_norm_(generate.parameters(), 5)

    # Update parameters with optimizers
    encoder_optimizer.step()
    predict_optimizer.step()
    generate_optimizer.step()
    return loss.item()  # , loss_0.item(), loss_1.item()


def topdown_evaluate_tree(input_batch, input_length, generate_nums, encoder, predict, generate, output_lang, num_pos,
                          beam_size=5, english=False, max_length=MAX_OUTPUT_LENGTH):

    seq_mask = torch.ByteTensor(1, input_length).fill_(0)
    # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
    input_var = torch.LongTensor(input_batch).unsqueeze(1)

    num_mask = torch.ByteTensor(1, len(num_pos) + len(generate_nums)).fill_(0)

    # Set to not-training mode to disable dropout
    encoder.eval()
    predict.eval()
    generate.eval()

    padding_hidden = torch.FloatTensor([0.0 for _ in range(predict.hidden_size)]).unsqueeze(0)

    batch_size = 1

    if USE_CUDA:
        input_var = input_var.cuda()
        seq_mask = seq_mask.cuda()
        padding_hidden = padding_hidden.cuda()
        num_mask = num_mask.cuda()
    # Run words through encoder

    encoder_outputs, problem_output = encoder(input_var, [input_length])

    # Prepare input and output variables
    node_stacks = [[TreeNode(_)] for _ in problem_output.split(1, dim=0)]

    num_size = len(num_pos)
    all_nums_encoder_outputs = get_all_number_encoder_outputs(encoder_outputs, [num_pos], batch_size, num_size,
                                                              encoder.hidden_size)
    num_start = output_lang.num_start
    # B x P x N
    embeddings_stacks = [[] for _ in range(batch_size)]
    left_childs = [None for _ in range(batch_size)]

    beams = [TreeBeam(0.0, node_stacks, embeddings_stacks, left_childs, [])]

    for t in range(max_length):
        current_beams = []
        while len(beams) > 0:
            b = beams.pop()
            if len(b.node_stack[0]) == 0:
                current_beams.append(b)
                continue
            # left_childs = torch.stack(b.left_childs)

            num_score, op, current_embeddings, current_context, current_nums_embeddings = predict(
                b.node_stack, left_childs, encoder_outputs, all_nums_encoder_outputs, padding_hidden,
                seq_mask, num_mask)

            # leaf = p_leaf[:, 0].unsqueeze(1)
            # repeat_dims = [1] * leaf.dim()
            # repeat_dims[1] = op.size(1)
            # leaf = leaf.repeat(*repeat_dims)
            #
            # non_leaf = p_leaf[:, 1].unsqueeze(1)
            # repeat_dims = [1] * non_leaf.dim()
            # repeat_dims[1] = num_score.size(1)
            # non_leaf = non_leaf.repeat(*repeat_dims)
            #
            # p_leaf = torch.cat((leaf, non_leaf), dim=1)
            out_score = nn.functional.log_softmax(torch.cat((op, num_score), dim=1), dim=1)

            # out_score = p_leaf * out_score

            topv, topi = out_score.topk(beam_size)

            # is_leaf = int(topi[0])
            # if is_leaf:
            #     topv, topi = op.topk(1)
            #     out_token = int(topi[0])
            # else:
            #     topv, topi = num_score.topk(1)
            #     out_token = int(topi[0]) + num_start

            for tv, ti in zip(topv.split(1, dim=1), topi.split(1, dim=1)):
                current_node_stack = copy_list(b.node_stack)
                current_out = copy.deepcopy(b.out)

                out_token = int(ti)
                current_out.append(out_token)

                node = current_node_stack[0].pop()

                if out_token < num_start:
                    generate_input = torch.LongTensor([out_token])
                    if USE_CUDA:
                        generate_input = generate_input.cuda()
                    left_child, right_child, node_label = generate(current_embeddings, generate_input, current_context)

                    current_node_stack[0].append(TreeNode(right_child))
                    current_node_stack[0].append(TreeNode(left_child, left_flag=True))

                current_beams.append(TreeBeam(b.score+float(tv), current_node_stack, embeddings_stacks, left_childs,
                                              current_out))
        beams = sorted(current_beams, key=lambda x: x.score, reverse=True)
        beams = beams[:beam_size]
        flag = True
        for b in beams:
            if len(b.node_stack[0]) != 0:
                flag = False
        if flag:
            break

    return beams[0].out
