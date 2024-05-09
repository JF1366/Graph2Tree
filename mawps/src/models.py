import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import math
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


class EncoderRNN(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderRNN, self).__init__()

        # 初始化函数__init__
        self.input_size = input_size            # 输入的大小，通常是词汇表的大小。
        self.embedding_size = embedding_size    # 嵌入层的维度，这决定了词汇被映射到多大的向量空间。
        self.hidden_size = hidden_size          # 隐藏层的维度，这是GRU层内部状态的大小。
        self.n_layers = n_layers                # GRU层的层数，默认为2层。
        self.dropout = dropout                  # 在训练过程中，为了防止过拟合，随机丢弃一定比例的神经网络单元，默认为0.5。

        # 组件定义
        # 一个嵌入层，用于将输入的整数（通常表示词汇）转换为固定大小的嵌入向量。
        # padding_idx=0，表示索引为0的词汇对应的向量将被初始化为零向量，
        # 这通常用于处理变长的输入序列中的填充部分。
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        # 一个Dropout层，用于在嵌入向量上应用dropout操作，以减少模型对特定单词嵌入的过拟合。
        self.em_dropout = nn.Dropout(dropout)
        # 一个GRU层，用于处理序列数据。这里设置为双向（bidirectional=True）
        # 意味着网络将会同时处理正向和反向的序列信息，以获得更全面的上下文理解。
        self.gru = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # 将输入的词汇索引通过嵌入层转换成对应的向量表示，形状为S x B x E（序列长度 x 批次大小 x 嵌入维度）。
        embedded = self.embedding(input_seqs)  # S x B x E
        # 对嵌入向量应用dropout，减少过拟合，增强模型的泛化能力。
        embedded = self.em_dropout(embedded)
        # 使用pack_padded_sequence函数对嵌入向量和对应的序列长度进行打包，
        # 这样GRU在处理时会忽略填充的部分，提高运算效率和模型性能。
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        # 将打包的序列和初始隐藏状态输入到GRU层，GRU层处理后返回输出序列和最终的隐藏状态。
        # 输出序列包含了每个时间步的输出，最终隐藏状态包含了序列的综合信息。
        outputs, hidden = self.gru(packed, hidden)
        # 使用pad_packed_sequence将输出序列解包回原始的填充形式，方便后续处理。
        outputs, output_lengths = torch.nn.utils.rnn.pad_packed_sequence(outputs)  # unpack (back to padded)
        # 因为GRU设置为双向，输出的隐藏状态维度会是隐藏层维度的两倍。
        # 这里将双向的结果相加，简化为单向输出的维度，从而合并来自两个方向的信息。
        outputs = outputs[:, :, :self.hidden_size] + outputs[:, :, self.hidden_size:]  # Sum bidirectional outputs
        # S x B x H
        return outputs, hidden  # 返回处理过的输出序列和最终的隐藏状态，它们将被用于模型的后续部分或用作解码器的输入。


class Attn(nn.Module):
    def __init__(self, hidden_size):
        super(Attn, self).__init__()
        # 类初始化函数 __init__
        self.hidden_size = hidden_size  # 隐藏层的大小，这是传递给注意力机制的特征数量。
        self.attn = nn.Linear(hidden_size * 2, hidden_size)     # 一个线性层，用于将来自编码器和解码器的隐藏状态映射到一个新的空间。
        # 另一个线性层，用于将经过self.attn处理的状态映射到一个单一的分数（或能量），
        # 这个分数定义了每个编码器输出对当前解码步骤的重要性。
        self.score = nn.Linear(hidden_size, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)    # 用于将注意力分数转化为权重，这些权重的和为1。

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)   # 首先获取编码器输出的最大序列长度。
        repeat_dims = [1] * hidden.dim()    # 创建一个列表，其长度等于隐藏状态的维度数，所有元素初始化为1。
        repeat_dims[0] = max_len            # 将列表的第一个元素（对应序列长度的维度）设置为max_len。
        # 将解码器的隐藏状态沿着序列长度维度重复扩展，使其维度与编码器输出匹配。
        # 这允许每个时间步的隐藏状态与对应的编码器输出合并。
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        # For each position of encoder outputs
        this_batch_size = encoder_outputs.size(1)   # 编码器输出的每个位置
        # 将扩展后的隐藏状态与编码器输出在最后一个维度（特征维度）上连接，然后将结果展平，以适应线性层的输入要求。
        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, 2 * self.hidden_size)
        # 通过self.attn线性层处理连接后的数据，应用torch.tanh作为激活函数，
        # 然后通过self.score线性层计算得分，得到每个时间步的注意力能量。
        attn_energies = self.score(torch.tanh(self.attn(energy_in)))  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)    # 压缩单一维度，去除多余的维度。
        # 将注意力能量调整为max_len x this_batch_size的形状，并转置以匹配批次维度和序列维度。
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:    # 如果提供了序列掩码（用于标记不应被模型注意的序列部分，如填充部分）
            # 掩码指示的位置应用一个非常小的值（-1e12），确保这些位置在后续softmax操作中的权重接近零。
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        # 使用softmax函数归一化注意力能量，转换为一个概率分布（权重），使得权重的和为1。
        attn_energies = self.softmax(attn_energies)
        # Normalize energies to weights in range 0 to 1, resize to B x 1 x S
        return attn_energies.unsqueeze(1)


class AttnDecoderRNN(nn.Module):
    def __init__(
            self, hidden_size, embedding_size, input_size, output_size, n_layers=2, dropout=0.5):
        super(AttnDecoderRNN, self).__init__()

        # Keep for reference
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout

        # Define layers
        self.em_dropout = nn.Dropout(dropout)
        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.gru = nn.GRU(hidden_size + embedding_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 2, hidden_size)
        self.out = nn.Linear(hidden_size, output_size)
        # Choose attention model
        self.attn = Attn(hidden_size)

    def forward(self, input_seq, last_hidden, encoder_outputs, seq_mask):
        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.em_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.embedding_size)  # S=1 x B x N

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(last_hidden[-1].unsqueeze(0), encoder_outputs, seq_mask)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # B x S=1 x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.gru(torch.cat((embedded, context.transpose(0, 1)), 2), last_hidden)

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        output = self.out(torch.tanh(self.concat(torch.cat((rnn_output.squeeze(0), context.squeeze(1)), 1))))

        # Return final output, hidden state
        return output, hidden


class TreeNode:  # the class save the tree node
    def __init__(self, embedding, left_flag=False):
        self.embedding = embedding
        self.left_flag = left_flag


class Score(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(Score, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden, num_embeddings, num_mask=None):
        max_len = num_embeddings.size(1)
        repeat_dims = [1] * hidden.dim()
        repeat_dims[1] = max_len
        hidden = hidden.repeat(*repeat_dims)  # B x O x H
        # For each position of encoder outputs
        this_batch_size = num_embeddings.size(0)
        energy_in = torch.cat((hidden, num_embeddings), 2).view(-1, self.input_size + self.hidden_size)
        score = self.score(torch.tanh(self.attn(energy_in)))  # (B x O) x 1
        score = score.squeeze(1)
        score = score.view(this_batch_size, -1)  # B x O
        if num_mask is not None:
            score = score.masked_fill_(num_mask, -1e12)
        return score


class TreeAttn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TreeAttn, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size + input_size, hidden_size)
        self.score = nn.Linear(hidden_size, 1)

    def forward(self, hidden, encoder_outputs, seq_mask=None):
        max_len = encoder_outputs.size(0)

        repeat_dims = [1] * hidden.dim()
        repeat_dims[0] = max_len
        hidden = hidden.repeat(*repeat_dims)  # S x B x H
        this_batch_size = encoder_outputs.size(1)

        energy_in = torch.cat((hidden, encoder_outputs), 2).view(-1, self.input_size + self.hidden_size)

        score_feature = torch.tanh(self.attn(energy_in))
        attn_energies = self.score(score_feature)  # (S x B) x 1
        attn_energies = attn_energies.squeeze(1)
        attn_energies = attn_energies.view(max_len, this_batch_size).transpose(0, 1)  # B x S
        if seq_mask is not None:
            attn_energies = attn_energies.masked_fill_(seq_mask, -1e12)
        attn_energies = nn.functional.softmax(attn_energies, dim=1)  # B x S

        return attn_energies.unsqueeze(1)


class EncoderSeq(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, n_layers=2, dropout=0.5):
        super(EncoderSeq, self).__init__()

        self.input_size = input_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout

        self.embedding = nn.Embedding(input_size, embedding_size, padding_idx=0)
        self.em_dropout = nn.Dropout(dropout)
        # 一个双向GRU网络，用于从序列中学习长期依赖关系。
        self.gru_pade = nn.GRU(embedding_size, hidden_size, n_layers, dropout=dropout, bidirectional=True)
        # 一个图卷积网络模块，用于处理图结构数据，将图的结构信息与序列信息结合。
        self.gcn = Graph_Module(hidden_size, hidden_size, hidden_size)

    def forward(self, input_seqs, input_lengths, batch_graph, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        # 1、嵌入和Dropout处理：
        embedded = self.embedding(input_seqs)  # S x B x E
        embedded = self.em_dropout(embedded)
        # 2、处理变长序列：
        # 将嵌入向量打包，以有效处理变长输入。
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        pade_hidden = hidden
        # 3、GRU处理：
        # 将打包的嵌入向量送入双向GRU网络，获取输出和更新的隐藏状态。
        pade_outputs, pade_hidden = self.gru_pade(packed, pade_hidden)
        # 解包GRU输出。
        pade_outputs, _ = torch.nn.utils.rnn.pad_packed_sequence(pade_outputs)
        # 4、特殊输出处理：
        # 结合最后一个时间步和第一个时间步的GRU输出，生成特定的问题输出。
        problem_output = pade_outputs[-1, :, :self.hidden_size] + pade_outputs[0, :, self.hidden_size:]
        # 5、结合双向GRU输出：
        # 合并双向GRU的输出，简化为单向输出的维度。
        pade_outputs = pade_outputs[:, :, :self.hidden_size] + pade_outputs[:, :, self.hidden_size:]  # S x B x H
        # 6、图卷积网络处理：
        # 将合并后的GRU输出和图结构数据送入GCN进行处理。
        _, pade_outputs = self.gcn(pade_outputs, batch_graph)
        # 7、调整输出形状：
        # 转置输出以匹配期望的批次维度和序列维度。
        pade_outputs = pade_outputs.transpose(0, 1)
        # 返回处理过的序列输出（pade_outputs）和针对特定问题的输出（problem_output）
        return pade_outputs, problem_output


class Prediction(nn.Module):
    # a seq2tree decoder with Problem aware dynamic encoding

    def __init__(self, hidden_size, op_nums, input_size, dropout=0.5):
        super(Prediction, self).__init__()

        # Keep for reference
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.op_nums = op_nums

        # Define layers
        self.dropout = nn.Dropout(dropout)

        self.embedding_weight = nn.Parameter(torch.randn(1, input_size, hidden_size))

        # for Computational symbols and Generated numbers
        self.concat_l = nn.Linear(hidden_size, hidden_size)
        self.concat_r = nn.Linear(hidden_size * 2, hidden_size)
        self.concat_lg = nn.Linear(hidden_size, hidden_size)
        self.concat_rg = nn.Linear(hidden_size * 2, hidden_size)

        self.ops = nn.Linear(hidden_size * 2, op_nums)

        self.attn = TreeAttn(hidden_size, hidden_size)
        self.score = Score(hidden_size * 2, hidden_size)

    def forward(self, node_stacks, left_childs, encoder_outputs, num_pades, padding_hidden, seq_mask, mask_nums):
        current_embeddings = []

        for st in node_stacks:
            if len(st) == 0:
                current_embeddings.append(padding_hidden)
            else:
                current_node = st[-1]
                current_embeddings.append(current_node.embedding)

        current_node_temp = []
        for l, c in zip(left_childs, current_embeddings):
            if l is None:
                c = self.dropout(c)
                g = torch.tanh(self.concat_l(c))
                t = torch.sigmoid(self.concat_lg(c))
                current_node_temp.append(g * t)
            else:
                ld = self.dropout(l)
                c = self.dropout(c)
                g = torch.tanh(self.concat_r(torch.cat((ld, c), 1)))
                t = torch.sigmoid(self.concat_rg(torch.cat((ld, c), 1)))
                current_node_temp.append(g * t)

        current_node = torch.stack(current_node_temp)

        current_embeddings = self.dropout(current_node)

        current_attn = self.attn(current_embeddings.transpose(0, 1), encoder_outputs, seq_mask)
        current_context = current_attn.bmm(encoder_outputs.transpose(0, 1))  # B x 1 x N

        # the information to get the current quantity
        batch_size = current_embeddings.size(0)
        # predict the output (this node corresponding to output(number or operator)) with PADE

        repeat_dims = [1] * self.embedding_weight.dim()
        repeat_dims[0] = batch_size
        embedding_weight = self.embedding_weight.repeat(*repeat_dims)  # B x input_size x N
        embedding_weight = torch.cat((embedding_weight, num_pades), dim=1)  # B x O x N

        leaf_input = torch.cat((current_node, current_context), 2)
        leaf_input = leaf_input.squeeze(1)
        leaf_input = self.dropout(leaf_input)

        # p_leaf = nn.functional.softmax(self.is_leaf(leaf_input), 1)
        # max pooling the embedding_weight
        embedding_weight_ = self.dropout(embedding_weight)
        num_score = self.score(leaf_input.unsqueeze(1), embedding_weight_, mask_nums)

        # num_score = nn.functional.softmax(num_score, 1)

        op = self.ops(leaf_input)

        # return p_leaf, num_score, op, current_embeddings, current_attn

        return num_score, op, current_node, current_context, embedding_weight


class GenerateNode(nn.Module):
    def __init__(self, hidden_size, op_nums, embedding_size, dropout=0.5):
        super(GenerateNode, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.embeddings = nn.Embedding(op_nums, embedding_size)
        self.em_dropout = nn.Dropout(dropout)
        self.generate_l = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_r = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_lg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.generate_rg = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, node_label, current_context):
        node_label_ = self.embeddings(node_label)
        node_label = self.em_dropout(node_label_)
        node_embedding = node_embedding.squeeze(1)
        current_context = current_context.squeeze(1)
        node_embedding = self.em_dropout(node_embedding)
        current_context = self.em_dropout(current_context)

        l_child = torch.tanh(self.generate_l(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child_g = torch.sigmoid(self.generate_lg(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child = torch.tanh(self.generate_r(torch.cat((node_embedding, current_context, node_label), 1)))
        r_child_g = torch.sigmoid(self.generate_rg(torch.cat((node_embedding, current_context, node_label), 1)))
        l_child = l_child * l_child_g
        r_child = r_child * r_child_g
        return l_child, r_child, node_label_


class Merge(nn.Module):
    def __init__(self, hidden_size, embedding_size, dropout=0.5):
        super(Merge, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size

        self.em_dropout = nn.Dropout(dropout)
        self.merge = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)
        self.merge_g = nn.Linear(hidden_size * 2 + embedding_size, hidden_size)

    def forward(self, node_embedding, sub_tree_1, sub_tree_2):
        sub_tree_1 = self.em_dropout(sub_tree_1)
        sub_tree_2 = self.em_dropout(sub_tree_2)
        node_embedding = self.em_dropout(node_embedding)

        sub_tree = torch.tanh(self.merge(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree_g = torch.sigmoid(self.merge_g(torch.cat((node_embedding, sub_tree_1, sub_tree_2), 1)))
        sub_tree = sub_tree * sub_tree_g
        return sub_tree
    
    
    
# Graph Module
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff,d_out, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Graph_Module(nn.Module):
    def __init__(self, indim, hiddim, outdim, dropout=0.3):
        super(Graph_Module, self).__init__()
        '''
        ## Variables:
        - indim: dimensionality of input node features
        - hiddim: dimensionality of the joint hidden embedding
        - outdim: dimensionality of the output node features
        - combined_feature_dim: dimensionality of the joint hidden embedding for graph
        - K: number of graph nodes/objects on the image
        '''
        self.in_dim = indim
        #self.combined_dim = outdim
        
        #self.edge_layer_1 = nn.Linear(indim, outdim)
        #self.edge_layer_2 = nn.Linear(outdim, outdim)
        
        #self.dropout = nn.Dropout(p=dropout)
        #self.edge_layer_1 = nn.utils.weight_norm(self.edge_layer_1)
        #self.edge_layer_2 = nn.utils.weight_norm(self.edge_layer_2)
        self.h = 4
        self.d_k = outdim//self.h
        
        #layer = GCN(indim, hiddim, self.d_k, dropout)
        self.graph = clones(GCN(indim, hiddim, self.d_k, dropout), 4)
        
        #self.Graph_0 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_1 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_2 = GCN(indim, hiddim, outdim//4, dropout)
        #self.Graph_3 = GCN(indim, hiddim, outdim//4, dropout)
        
        self.feed_foward = PositionwiseFeedForward(indim, hiddim, outdim, dropout)
        self.norm = LayerNorm(outdim)

    def get_adj(self, graph_nodes):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - adjacency matrix (batch_size, K, K)
        '''
        self.K = graph_nodes.size(1)
        graph_nodes = graph_nodes.contiguous().view(-1, self.in_dim)
        
        # layer 1
        h = self.edge_layer_1(graph_nodes)
        h = F.relu(h)
        
        # layer 2
        h = self.edge_layer_2(h)
        h = F.relu(h)

        # outer product
        h = h.view(-1, self.K, self.combined_dim)
        adjacency_matrix = torch.matmul(h, h.transpose(1, 2))
        
        adjacency_matrix = self.b_normal(adjacency_matrix)

        return adjacency_matrix
    
    def normalize(self, A, symmetric=True):
        '''
        ## Inputs:
        - adjacency matrix (K, K) : A
        ## Returns:
        - adjacency matrix (K, K) 
        '''
        A = A + torch.eye(A.size(0)).cuda().float()
        d = A.sum(1)
        if symmetric:
            # D = D^{-1/2}
            D = torch.diag(torch.pow(d, -0.5))
            return D.mm(A).mm(D)
        else :
            D = torch.diag(torch.pow(d,-1))
            return D.mm(A)
       
    def b_normal(self, adj):
        batch = adj.size(0)
        for i in range(batch):
            adj[i] = self.normalize(adj[i])
        return adj

    def forward(self, graph_nodes, graph):
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        ## Returns:
        - graph_encode_features (batch_size, K, out_feat_dim)
        '''
        nbatches = graph_nodes.size(0)
        mbatches = graph.size(0)
        if nbatches != mbatches:
            graph_nodes = graph_nodes.transpose(0, 1)
        # adj (batch_size, K, K): adjacency matrix
        if not bool(graph.numel()):
            adj = self.get_adj(graph_nodes)
            #adj = adj.unsqueeze(1)
            #adj = torch.cat((adj,adj,adj),1)
            adj_list = [adj,adj,adj,adj]
        else:
            adj = graph.float()
            adj_list = [adj[:,1,:],adj[:,1,:],adj[:,4,:],adj[:,4,:]]
        #print(adj)
        
        g_feature = \
            tuple([l(graph_nodes,x) for l, x in zip(self.graph,adj_list)])
        #g_feature_0 = self.Graph_0(graph_nodes,adj[0])
        #g_feature_1 = self.Graph_1(graph_nodes,adj[1])
        #g_feature_2 = self.Graph_2(graph_nodes,adj[2])
        #g_feature_3 = self.Graph_3(graph_nodes,adj[3])
        #print('g_feature')
        #print(type(g_feature))
        
        
        g_feature = self.norm(torch.cat(g_feature,2)) + graph_nodes
        #print('g_feature')
        #print(g_feature.shape)
        
        graph_encode_features = self.feed_foward(g_feature) + g_feature
        
        return adj, graph_encode_features

# GCN
class GCN(nn.Module):
    def __init__(self, in_feat_dim, nhid, out_feat_dim, dropout):
        super(GCN, self).__init__()
        '''
        ## Inputs:
        - graph_nodes (batch_size, K, in_feat_dim): input features
        - adjacency matrix (batch_size, K, K)
        ## Returns:
        - gcn_enhance_feature (batch_size, K, out_feat_dim)
        '''
        self.gc1 = GraphConvolution(in_feat_dim, nhid)
        self.gc2 = GraphConvolution(nhid, out_feat_dim)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x
    
# Graph_Conv
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        #print(input.shape)
        #print(self.weight.shape)
        support = torch.matmul(input, self.weight)
        #print(adj.shape)
        #print(support.shape)
        output = torch.matmul(adj, support)
        
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
