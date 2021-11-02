from fastNLP.embeddings.embedding import TokenEmbedding
from fastNLP.core import Vocabulary
from fastNLP.io.file_utils import PRETRAIN_STATIC_FILES, _get_embedding_url, cached_path
import os
import collections
import unicodedata
import warnings
from collections import defaultdict
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn

from fastNLP.core import logger
# from fastNLP.modules.utils import _get_file_name_base_on_postfix
from utils import MyDropout
from fastNLP.embeddings.contextual_embedding import ContextualEmbedding
# from fastNLP.embeddings.bert_embedding import _WordBertModel  # 直接搬到下边了
from fastNLP.io.file_utils import PRETRAINED_BERT_MODEL_DIR
from fastNLP.modules import BertModel
from itertools import chain


def _get_file_name_base_on_postfix(dir_path, postfix):
    r"""
    在dir_path中寻找后缀为postfix的文件.
    :param dir_path: str, 文件夹
    :param postfix: 形如".bin", ".json"等
    :return: str，文件的路径
    """
    files = list(filter(lambda filename: filename.endswith(postfix), os.listdir(os.path.join(dir_path))))
    if len(files) == 0:
        raise FileNotFoundError(f"There is no file endswith {postfix} file in {dir_path}")
    elif len(files) > 1:
        raise FileExistsError(f"There are multiple *{postfix} files in {dir_path}")
    return os.path.join(dir_path, files[0])


class StaticEmbedding(TokenEmbedding):
    """
    StaticEmbedding组件. 给定预训练embedding的名称或路径，根据vocab从embedding中抽取相应的数据(只会将出现在vocab中的词抽取出来，
    如果没有找到，则会随机初始化一个值(但如果该word是被标记为no_create_entry的话，则不会单独创建一个值，而是会被指向unk的index))。
    当前支持自动下载的预训练vector有以下的几种(待补充);

    Example::

        >>> from fastNLP import Vocabulary
        >>> from fastNLP.embeddings import StaticEmbedding
        >>> vocab = Vocabulary().add_word_lst("The whether is good .".split())
        >>> embed = StaticEmbedding(vocab, model_dir_or_name='en-glove-50d')

        >>> vocab = Vocabulary().add_word_lst(["The", 'the', "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name="en-glove-50d", lower=True)
        >>> # "the", "The", "THE"它们共用一个vector，且将使用"the"在预训练词表中寻找它们的初始化表示。

        >>> vocab = Vocabulary().add_word_lst(["The", "the", "THE"])
        >>> embed = StaticEmbedding(vocab, model_dir_or_name=None, embedding_dim=5, lower=True)
        >>> words = torch.LongTensor([[vocab.to_index(word) for word in ["The", "the", "THE"]]])
        >>> embed(words)
        >>> tensor([[[ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849],
                     [ 0.5773,  0.7251, -0.3104,  0.0777,  0.4849]]],
                   grad_fn=<EmbeddingBackward>)  # 每种word的输出是一致的。

    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en', embedding_dim=-1, requires_grad: bool = True,
                 init_method=None, lower=False, dropout=0, word_dropout=0, normalize=False, min_freq=1, **kwargs):
        """

        :param vocab: Vocabulary. 若该项为None则会读取所有的embedding。
        :param model_dir_or_name: 可以有两种方式调用预训练好的static embedding：第一种是传入embedding文件夹(文件夹下应该只有一个
            以.txt作为后缀的文件)或文件路径；第二种是传入embedding的名称，第二种情况将自动查看缓存中是否存在该模型，没有的话将自动下载。
            如果输入为None则使用embedding_dim的维度随机初始化一个embedding。
        :param int embedding_dim: 随机初始化的embedding的维度，当该值为大于0的值时，将忽略model_dir_or_name。
        :param bool requires_grad: 是否需要gradient. 默认为True
        :param callable init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法, 传入的方法应该接受一个tensor，并
            inplace地修改其值。
        :param bool lower: 是否将vocab中的词语小写后再和预训练的词表进行匹配。如果你的词表中包含大写的词语，或者就是需要单独
            为大写的词语开辟一个vector表示，则将lower设置为False。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param bool normalize: 是否对vector进行normalize，使得每个vector的norm为1。
        :param int min_freq: Vocabulary词频数小于这个数量的word将被指向unk。
        :param dict kwarngs: only_train_min_freq, 仅对train中的词语使用min_freq筛选; only_norm_found_vector是否仅对在预训练中找到的词语使用normalize。
        """
        super(StaticEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        if embedding_dim > 0:
            model_dir_or_name = None

        # 得到cache_path
        if model_dir_or_name is None:
            assert embedding_dim >= 1, "The dimension of embedding should be larger than 1."
            embedding_dim = int(embedding_dim)
            model_path = None
        elif model_dir_or_name.lower() in PRETRAIN_STATIC_FILES:
            model_url = _get_embedding_url('static', model_dir_or_name.lower())
            model_path = cached_path(model_url, name='embedding')
            # 检查是否存在
        elif os.path.isfile(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = os.path.abspath(os.path.expanduser(model_dir_or_name))
        elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
            model_path = _get_file_name_base_on_postfix(os.path.abspath(os.path.expanduser(model_dir_or_name)), '.txt')
        else:
            raise ValueError(f"Cannot recognize {model_dir_or_name}.")

        # 根据min_freq缩小vocab
        truncate_vocab = (vocab.min_freq is None and min_freq > 1) or (vocab.min_freq and vocab.min_freq < min_freq)
        if truncate_vocab:
            truncated_vocab = deepcopy(vocab)
            truncated_vocab.min_freq = min_freq
            truncated_vocab.word2idx = None
            if lower:  # 如果有lower，将大小写的的freq需要同时考虑到
                lowered_word_count = defaultdict(int)
                for word, count in truncated_vocab.word_count.items():
                    lowered_word_count[word.lower()] += count
                for word in truncated_vocab.word_count.keys():
                    word_count = truncated_vocab.word_count[word]
                    if lowered_word_count[word.lower()] >= min_freq and word_count < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - word_count),
                                                     no_create_entry=truncated_vocab._is_word_no_create_entry(word))

            # 只限制在train里面的词语使用min_freq筛选
            if kwargs.get('only_train_min_freq', False) and model_dir_or_name is not None:
                for word in truncated_vocab.word_count.keys():
                    if truncated_vocab._is_word_no_create_entry(word) and truncated_vocab.word_count[word] < min_freq:
                        truncated_vocab.add_word_lst([word] * (min_freq - truncated_vocab.word_count[word]),
                                                     no_create_entry=True)
            truncated_vocab.build_vocab()
            truncated_words_to_words = torch.arange(len(vocab)).long()
            for word, index in vocab:
                truncated_words_to_words[index] = truncated_vocab.to_index(word)
            logger.info(
                f"{len(vocab) - len(truncated_vocab)} out of {len(vocab)} words have frequency less than {min_freq}.")
            vocab = truncated_vocab

        self.only_norm_found_vector = kwargs.get('only_norm_found_vector', False)
        # 读取embedding
        if lower:
            lowered_vocab = Vocabulary(padding=vocab.padding, unknown=vocab.unknown)
            for word, index in vocab:
                if vocab._is_word_no_create_entry(word):
                    lowered_vocab.add_word(word.lower(), no_create_entry=True)
                else:
                    lowered_vocab.add_word(word.lower())  # 先加入需要创建entry的
            logger.info(f"All word in the vocab have been lowered. There are {len(vocab)} words, {len(lowered_vocab)} "
                        f"unique lowered words.")
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=lowered_vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            if lowered_vocab.unknown:
                unknown_idx = lowered_vocab.unknown_idx
            else:
                unknown_idx = embedding.size(0) - 1  # 否则是最后一个为unknow
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
            words_to_words = torch.full((len(vocab),), fill_value=unknown_idx).long()
            for word, index in vocab:
                if word not in lowered_vocab:
                    word = word.lower()
                    if word not in lowered_vocab and lowered_vocab._is_word_no_create_entry(word):
                        continue  # 如果不需要创建entry,已经默认unknown了
                words_to_words[index] = self.words_to_words[lowered_vocab.to_index(word)]
            self.register_buffer('words_to_words', words_to_words)
            self._word_unk_index = lowered_vocab.unknown_idx  # 替换一下unknown的index
        else:
            if model_path:
                embedding = self._load_with_vocab(model_path, vocab=vocab, init_method=init_method)
            else:
                embedding = self._randomly_init_embed(len(vocab), embedding_dim, init_method)
                self.register_buffer('words_to_words', torch.arange(len(vocab)).long())
        if not self.only_norm_found_vector and normalize:
            embedding /= (torch.norm(embedding, dim=1, keepdim=True) + 1e-12)

        if truncate_vocab:
            for i in range(len(truncated_words_to_words)):
                index_in_truncated_vocab = truncated_words_to_words[i]
                truncated_words_to_words[i] = self.words_to_words[index_in_truncated_vocab]
            del self.words_to_words
            self.register_buffer('words_to_words', truncated_words_to_words)
        self.embedding = nn.Embedding(num_embeddings=embedding.shape[0], embedding_dim=embedding.shape[1],
                                      padding_idx=vocab.padding_idx,
                                      max_norm=None, norm_type=2, scale_grad_by_freq=False,
                                      sparse=False, _weight=embedding)
        self._embed_size = self.embedding.weight.size(1)
        self.requires_grad = requires_grad
        self.dropout = MyDropout(dropout)

    def _randomly_init_embed(self, num_embedding, embedding_dim, init_embed=None):
        """

        :param int num_embedding: embedding的entry的数量
        :param int embedding_dim: embedding的维度大小
        :param callable init_embed: 初始化方法
        :return: torch.FloatTensor
        """
        embed = torch.zeros(num_embedding, embedding_dim)

        if init_embed is None:
            nn.init.uniform_(embed, -np.sqrt(3 / embedding_dim), np.sqrt(3 / embedding_dim))
        else:
            init_embed(embed)

        return embed

    def _load_with_vocab(self, embed_filepath, vocab, dtype=np.float32, padding='<pad>', unknown='<unk>',
                         error='ignore', init_method=None):
        """
        从embed_filepath这个预训练的词向量中抽取出vocab这个词表的词的embedding。EmbedLoader将自动判断embed_filepath是
        word2vec(第一行只有两个元素)还是glove格式的数据。

        :param str embed_filepath: 预训练的embedding的路径。
        :param vocab: 词表 :class:`~fastNLP.Vocabulary` 类型，读取出现在vocab中的词的embedding。
            没有出现在vocab中的词的embedding将通过找到的词的embedding的正态分布采样出来，以使得整个Embedding是同分布的。
        :param dtype: 读出的embedding的类型
        :param str padding: 词表中padding的token
        :param str unknown: 词表中unknown的token
        :param str error: `ignore` , `strict` ; 如果 `ignore` ，错误将自动跳过; 如果 `strict` , 错误将抛出。
            这里主要可能出错的地方在于词表有空行或者词表出现了维度不一致。
        :param init_method: 如何初始化没有找到的值。可以使用torch.nn.init.*中各种方法。默认使用torch.nn.init.zeros_
        :return torch.tensor:  shape为 [len(vocab), dimension], dimension由pretrain的embedding决定。
        """
        assert isinstance(vocab, Vocabulary), "Only fastNLP.Vocabulary is supported."
        if not os.path.exists(embed_filepath):
            raise FileNotFoundError("`{}` does not exist.".format(embed_filepath))
        with open(embed_filepath, 'r', encoding='utf-8') as f:
            line = f.readline().strip()
            parts = line.split()
            start_idx = 0
            if len(parts) == 2:
                dim = int(parts[1])
                start_idx += 1
            else:
                dim = len(parts) - 1
                f.seek(0)
            matrix = {}
            if vocab.padding:
                matrix[vocab.padding_idx] = torch.zeros(dim)
            if vocab.unknown:
                matrix[vocab.unknown_idx] = torch.zeros(dim)
            found_count = 0
            found_unknown = False
            for idx, line in enumerate(f, start_idx):
                try:
                    parts = line.strip().split()
                    word = ''.join(parts[:-dim])
                    nums = parts[-dim:]
                    # 对齐unk与pad
                    if word == padding and vocab.padding is not None:
                        word = vocab.padding
                    elif word == unknown and vocab.unknown is not None:
                        word = vocab.unknown
                        found_unknown = True
                    if word in vocab:
                        index = vocab.to_index(word)
                        matrix[index] = torch.from_numpy(np.fromstring(' '.join(nums), sep=' ', dtype=dtype, count=dim))
                        if self.only_norm_found_vector:
                            matrix[index] = matrix[index] / np.linalg.norm(matrix[index])
                        found_count += 1
                except Exception as e:
                    if error == 'ignore':
                        warnings.warn("Error occurred at the {} line.".format(idx))
                    else:
                        logger.error("Error occurred at the {} line.".format(idx))
                        raise e
            logger.info("Found {} out of {} words in the pre-training embedding.".format(found_count, len(vocab)))
            for word, index in vocab:
                if index not in matrix and not vocab._is_word_no_create_entry(word):
                    if found_unknown:  # 如果有unkonwn，用unknown初始化
                        matrix[index] = matrix[vocab.unknown_idx]
                    else:
                        matrix[index] = None
            # matrix中代表是需要建立entry的词
            vectors = self._randomly_init_embed(len(matrix), dim, init_method)

            if vocab.unknown is None:  # 创建一个专门的unknown
                unknown_idx = len(matrix)
                vectors = torch.cat((vectors, torch.zeros(1, dim)), dim=0).contiguous()
            else:
                unknown_idx = vocab.unknown_idx
            self.register_buffer('words_to_words', torch.full((len(vocab),), fill_value=unknown_idx).long())
            for index, (index_in_vocab, vec) in enumerate(matrix.items()):
                if vec is not None:
                    vectors[index] = vec
                self.words_to_words[index_in_vocab] = index

            return vectors

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            mask = torch.rand(words.size())
            mask = mask.to(words.device)
            mask = mask.lt(self.word_dropout)
            # mask = torch.full_like(words, fill_value=self.word_dropout, dtype=torch.float)
            #             # mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
            #             # mask = mask.to(words.device)
            pad_mask = words.ne(self._word_pad_index)
            mask = mask.__and__(pad_mask)
            words = words.masked_fill(mask, self._word_unk_index)
        return words

    def forward(self, words):
        """
        传入words的index

        :param words: torch.LongTensor, [batch_size, max_len]
        :return: torch.FloatTensor, [batch_size, max_len, embed_size]
        """
        if hasattr(self, 'words_to_words'):
            words = self.words_to_words[words]
        words = self.drop_word(words)
        words = self.embedding(words)
        words = self.dropout(words)
        return words


class BertEmbedding(ContextualEmbedding):
    """
    使用BERT对words进行编码的Embedding。建议将输入的words长度限制在430以内，而不要使用512(根据预训练模型参数，可能有变化)。这是由于
    预训练的bert模型长度限制为512个token，而因为输入的word是未进行word piece分割的(word piece的分割有BertEmbedding在输入word
    时切分)，在分割之后长度可能会超过最大长度限制。

    BertEmbedding可以支持自动下载权重，当前支持的模型有以下的几种(待补充):

    Example::

        import torch
        from fastNLP import Vocabulary
        from fastNLP.embeddings import BertEmbedding
        vocab = Vocabulary().add_word_lst("The whether is good .".split())
        embed = BertEmbedding(vocab, model_dir_or_name='en-base-uncased', requires_grad=False, layers='4,-2,-1')
        words = torch.LongTensor([[vocab.to_index(word) for word in "The whether is good .".split()]])
        outputs = embed(words)
        outputs.size()
        # torch.Size([1, 5, 2304])
    """

    def __init__(self, vocab: Vocabulary, model_dir_or_name: str = 'en-base-uncased', layers: str = '-1',
                 pool_method: str = 'first', word_dropout=0, dropout=0, include_cls_sep: bool = False,
                 pooled_cls=True, requires_grad: bool = True, auto_truncate: bool = False):
        """

        :param ~fastNLP.Vocabulary vocab: 词表
        :param str model_dir_or_name: 模型所在目录或者模型的名称。当传入模型所在目录时，目录中应该包含一个词表文件(以.txt作为后缀名),
            权重文件(以.bin作为文件后缀名), 配置文件(以.json作为后缀名)。
        :param str layers: 输出embedding表示来自于哪些层，不同层的结果按照layers中的顺序在最后一维concat起来。以','隔开层数，层的序号是
            从0开始，可以以负数去索引倒数几层。
        :param str pool_method: 因为在bert中，每个word会被表示为多个word pieces, 当获取一个word的表示的时候，怎样从它的word pieces
            中计算得到它对应的表示。支持 ``last`` , ``first`` , ``avg`` , ``max``。
        :param float word_dropout: 以多大的概率将一个词替换为unk。这样既可以训练unk也是一定的regularize。
        :param float dropout: 以多大的概率对embedding的表示进行Dropout。0.1即随机将10%的值置为0。
        :param bool include_cls_sep: bool，在bert计算句子的表示的时候，需要在前面加上[CLS]和[SEP], 是否在结果中保留这两个内容。 这样
            会使得word embedding的结果比输入的结果长两个token。如果该值为True，则在使用 :class::StackEmbedding 可能会与其它类型的
            embedding长度不匹配。
        :param bool pooled_cls: 返回的[CLS]是否使用预训练中的BertPool映射一下，仅在include_cls_sep时有效。如果下游任务只取[CLS]做预测，
            一般该值为True。
        :param bool requires_grad: 是否需要gradient以更新Bert的权重。
        :param bool auto_truncate: 当句子words拆分为word pieces长度超过bert最大允许长度(一般为512), 自动截掉拆分后的超过510个
            word pieces后的内容，并将第512个word piece置为[SEP]。超过长度的部分的encode结果直接全部置零。一般仅有只使用[CLS]
            来进行分类的任务将auto_truncate置为True。
        """
        super(BertEmbedding, self).__init__(vocab, word_dropout=word_dropout, dropout=dropout)
        self.device_cpu = torch.device('cpu')
        if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
            if 'cn' in model_dir_or_name.lower() and pool_method not in ('first', 'last'):
                logger.warning("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                               " faster speed.")
                warnings.warn("For Chinese bert, pooled_method should choose from 'first', 'last' in order to achieve"
                              " faster speed.")
        self.dropout_p = dropout
        self._word_sep_index = None
        if '[SEP]' in vocab:
            self._word_sep_index = vocab['[SEP]']

        self.model = _WordBertModel(model_dir_or_name=model_dir_or_name, vocab=vocab, layers=layers,
                                    pool_method=pool_method, include_cls_sep=include_cls_sep,
                                    pooled_cls=pooled_cls, auto_truncate=auto_truncate, min_freq=2)

        self.requires_grad = requires_grad
        self._embed_size = len(self.model.layers) * self.model.encoder.hidden_size

    def _delete_model_weights(self):
        del self.model

    def forward(self, words):
        """
        计算words的bert embedding表示。计算之前会在每句话的开始增加[CLS]在结束增加[SEP], 并根据include_cls_sep判断要不要
            删除这两个token的表示。

        :param torch.LongTensor words: [batch_size, max_len]
        :return: torch.FloatTensor. batch_size x max_len x (768*len(self.layers))
        """
        words = self.drop_word(words)
        outputs = self._get_sent_reprs(words)
        if outputs is not None:
            if self.dropout_p >1e-5:
                return self.dropout(outputs)
            else:
                return outputs
        outputs = self.model(words)
        # print(outputs.size())

        outputs = torch.cat([*outputs], dim=-1)
        # print(outputs.size())
        # exit()
        if self.dropout_p > 1e-5:
            return self.dropout(outputs)
        else:
            return outputs

    def drop_word(self, words):
        """
        按照设定随机将words设置为unknown_index。

        :param torch.LongTensor words: batch_size x max_len
        :return:
        """
        if self.word_dropout > 0 and self.training:
            with torch.no_grad():
                if self._word_sep_index:  # 不能drop sep
                    sep_mask = words.eq(self._word_sep_index)

                mask = torch.full(words.size(), fill_value=self.word_dropout, dtype=torch.float)
                # print(mask.device)
                # print(mask)
                # print(mask.device)
                # exit()
                # mask = mask.to(self.device_cpu)
                mask = torch.bernoulli(mask).eq(1)  # dropout_word越大，越多位置为1
                mask = mask.to(words.device)
                pad_mask = words.ne(0)
                mask = pad_mask.__and__(mask)  # pad的位置不为unk
                words = words.masked_fill(mask, self._word_unk_index)
                if self._word_sep_index:
                    words.masked_fill_(sep_mask, self._word_sep_index)
        return words
    
    
# class _BertWordModel(nn.Module):
class _WordBertModel(nn.Module):
    def __init__(self, model_dir_or_name: str, vocab: Vocabulary, layers: str = '-1', pool_method: str = 'first',
                 include_cls_sep: bool = False, pooled_cls: bool = False, auto_truncate: bool = False, min_freq=2,
                 only_use_pretrain_bpe=False, truncate_embed=True):
        super().__init__()

        self.tokenzier = BertTokenizer.from_pretrained(model_dir_or_name)
        self.encoder = BertModel.from_pretrained(model_dir_or_name)
        self._max_position_embeddings = self.encoder.config.max_position_embeddings
        #  检查encoder_layer_number是否合理
        encoder_layer_number = len(self.encoder.encoder.layer)
        if isinstance(layers, list):
            self.layers = [int(l) for l in layers]
        elif isinstance(layers, str):
            self.layers = list(map(int, layers.split(',')))
        else:
            raise TypeError("`layers` only supports str or list[int]")
        for layer in self.layers:
            if layer < 0:
                assert -layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                       f"a bert model with {encoder_layer_number} layers."
            else:
                assert layer <= encoder_layer_number, f"The layer index:{layer} is out of scope for " \
                                                     f"a bert model with {encoder_layer_number} layers."

        assert pool_method in ('avg', 'max', 'first', 'last')
        self.pool_method = pool_method
        self.include_cls_sep = include_cls_sep
        self.pooled_cls = pooled_cls
        self.auto_truncate = auto_truncate

        # 将所有vocab中word的wordpiece计算出来, 需要额外考虑[CLS]和[SEP]
        logger.info("Start to generate word pieces for word.")
        self._has_sep_in_vocab = '[SEP]' in vocab  # 用来判断传入的数据是否需要生成token_ids

        # 第一步统计出需要的word_piece, 然后创建新的embed和word_piece_vocab, 然后填入值
        word_piece_dict = {'[CLS]': 1, '[SEP]': 1}  # 用到的word_piece以及新增的
        new_add_to_bpe_vocab = 0

        unsegment_count = 0
        if '[sep]' in vocab:
            warnings.warn("Lower cased [sep] detected, it cannot be correctly recognized as [SEP] by BertEmbedding.")
        if "[CLS]" in vocab:
            warnings.warn("[CLS] detected in your vocabulary. BertEmbedding will add [CLS] and [SEP] to the begin "
                          "and end of the input automatically, make sure you don't add [CLS] and [SEP] at the begin"
                          " and end.")
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            _words = self.tokenzier.basic_tokenizer._tokenize_chinese_chars(word).split()
            word_pieces = []
            for w in _words:
                word_pieces.extend(self.tokenzier.wordpiece_tokenizer.tokenize(w))
            if len(word_pieces) == 1:
                if not vocab._is_word_no_create_entry(word):  # 如果是train中的值, 但是却没有找到
                    if index != vocab.unknown_idx and word_pieces[0] == '[UNK]':  # 说明这个词不在原始的word里面
                        if vocab.word_count[word] >= min_freq and not vocab._is_word_no_create_entry(
                                word) and not only_use_pretrain_bpe:  # 出现次数大于这个次数才新增
                            word_piece_dict[word] = 1  # 新增一个值
                            new_add_to_bpe_vocab += 1
                        unsegment_count += 1

                        continue
            for word_piece in word_pieces:
                word_piece_dict[word_piece] = 1
        original_embed = self.encoder.embeddings.word_embeddings.weight.data

        # 特殊词汇要特殊处理
        if not truncate_embed:# 如果不删除的话需要将已有的加上
            word_piece_dict.update(self.tokenzier.vocab)
        embed = nn.Embedding(len(word_piece_dict), original_embed.size(1))  # 新的embed
        new_word_piece_vocab = collections.OrderedDict()

        for index, token in enumerate(['[PAD]', '[UNK]']):
            index = word_piece_dict.pop(token, None)
            if index is not None:
                new_word_piece_vocab[token] = len(new_word_piece_vocab)
                embed.weight.data[new_word_piece_vocab[token]] = original_embed[self.tokenzier.vocab[token]]
        for token in word_piece_dict.keys():
            if token not in new_word_piece_vocab:
                new_word_piece_vocab[token] = len(new_word_piece_vocab)
            index = new_word_piece_vocab[token]
            if token in self.tokenzier.vocab:
                embed.weight.data[index] = original_embed[self.tokenzier.vocab[token]]
            else:
                embed.weight.data[index] = original_embed[self.tokenzier.vocab['[UNK]']]

        self.tokenzier._reinit_on_new_vocab(new_word_piece_vocab)
        self.encoder.embeddings.word_embeddings = embed

        self.encoder.config.vocab_size = len(new_word_piece_vocab)
        if unsegment_count>0:
            if only_use_pretrain_bpe or new_add_to_bpe_vocab==0:
                logger.info(f"{unsegment_count} words are unsegmented.")
            else:
                logger.info(f"{unsegment_count} words are unsegmented. Among them, {new_add_to_bpe_vocab} added to the BPE vocab.")

        word_to_wordpieces = []
        word_pieces_lengths = []
        for word, index in vocab:
            if index == vocab.padding_idx:  # pad是个特殊的符号
                word = '[PAD]'
            elif index == vocab.unknown_idx:
                word = '[UNK]'
            word_pieces = self.tokenzier.wordpiece_tokenizer.tokenize(word)
            word_pieces = self.tokenzier.convert_tokens_to_ids(word_pieces)
            word_to_wordpieces.append(word_pieces)
            word_pieces_lengths.append(len(word_pieces))
        self._cls_index = self.tokenzier.vocab['[CLS]']
        self._sep_index = self.tokenzier.vocab['[SEP]']
        self._word_pad_index = vocab.padding_idx
        self._wordpiece_pad_index = self.tokenzier.vocab['[PAD]']  # 需要用于生成word_piece
        self.word_to_wordpieces = np.array(word_to_wordpieces)
        self.register_buffer('word_pieces_lengths', torch.LongTensor(word_pieces_lengths))
        logger.debug("Successfully generate word pieces.")

    def forward(self, words):
        r"""

        :param words: torch.LongTensor, batch_size x max_len
        :return: num_layers x batch_size x max_len x hidden_size或者num_layers x batch_size x (max_len+2) x hidden_size
        """
        with torch.no_grad():
            batch_size, max_word_len = words.size()
            word_mask = words.ne(self._word_pad_index)  # 为1的地方有word
            seq_len = word_mask.sum(dim=-1)
            batch_word_pieces_length = self.word_pieces_lengths[words].masked_fill(word_mask.eq(False),
                                                                                   0)  # batch_size x max_len
            word_pieces_lengths = batch_word_pieces_length.sum(dim=-1)  # batch_size
            max_word_piece_length = batch_word_pieces_length.sum(dim=-1).max().item()  # 表示word piece的长度(包括padding)
            if max_word_piece_length + 2 > self._max_position_embeddings:
                if self.auto_truncate:
                    word_pieces_lengths = word_pieces_lengths.masked_fill(
                        word_pieces_lengths + 2 > self._max_position_embeddings,
                        self._max_position_embeddings - 2)
                else:
                    raise RuntimeError(
                        "After split words into word pieces, the lengths of word pieces are longer than the "
                        f"maximum allowed sequence length:{self._max_position_embeddings} of bert. You can set "
                        f"`auto_truncate=True` for BertEmbedding to automatically truncate overlong input.")

            # +2是由于需要加入[CLS]与[SEP]
            word_pieces = words.new_full((batch_size, min(max_word_piece_length + 2, self._max_position_embeddings)),
                                         fill_value=self._wordpiece_pad_index)
            attn_masks = torch.zeros_like(word_pieces)
            # 1. 获取words的word_pieces的id，以及对应的span范围
            word_indexes = words.cpu().numpy()
            for i in range(batch_size):
                word_pieces_i = list(chain(*self.word_to_wordpieces[word_indexes[i, :seq_len[i]]]))
                if self.auto_truncate and len(word_pieces_i) > self._max_position_embeddings - 2:
                    word_pieces_i = word_pieces_i[:self._max_position_embeddings - 2]
                word_pieces[i, 1:word_pieces_lengths[i] + 1] = torch.LongTensor(word_pieces_i)
                attn_masks[i, :word_pieces_lengths[i] + 2].fill_(1)
            # 添加[cls]和[sep]
            word_pieces[:, 0].fill_(self._cls_index)
            batch_indexes = torch.arange(batch_size).to(words)
            word_pieces[batch_indexes, word_pieces_lengths + 1] = self._sep_index
            if self._has_sep_in_vocab:  # 但[SEP]在vocab中出现应该才会需要token_ids
                sep_mask = word_pieces.eq(self._sep_index).long()  # batch_size x max_len
                sep_mask_cumsum = sep_mask.flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
                token_type_ids = sep_mask_cumsum.fmod(2)
                if token_type_ids[0, 0].item():  # 如果开头是奇数，则需要flip一下结果，因为需要保证开头为0
                    token_type_ids = token_type_ids.eq(0).long()
            else:
                token_type_ids = torch.zeros_like(word_pieces)
        # 2. 获取hidden的结果，根据word_pieces进行对应的pool计算
        # all_outputs: [batch_size x max_len x hidden_size, batch_size x max_len x hidden_size, ...]
        bert_outputs, pooled_cls = self.encoder(word_pieces, token_type_ids=token_type_ids, attention_mask=attn_masks,
                                                output_all_encoded_layers=True)
        # output_layers = [self.layers]  # len(self.layers) x batch_size x real_word_piece_length x hidden_size

        if self.include_cls_sep:
            s_shift = 1
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len + 2,
                                                     bert_outputs[-1].size(-1))

        else:
            s_shift = 0
            outputs = bert_outputs[-1].new_zeros(len(self.layers), batch_size, max_word_len,
                                                 bert_outputs[-1].size(-1))
        batch_word_pieces_cum_length = batch_word_pieces_length.new_zeros(batch_size, max_word_len + 1)
        batch_word_pieces_cum_length[:, 1:] = batch_word_pieces_length.cumsum(dim=-1)  # batch_size x max_len

        if self.pool_method == 'first':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, :seq_len.max()]
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))
        elif self.pool_method == 'last':
            batch_word_pieces_cum_length = batch_word_pieces_cum_length[:, 1:seq_len.max()+1] - 1
            batch_word_pieces_cum_length.masked_fill_(batch_word_pieces_cum_length.ge(max_word_piece_length), 0)
            _batch_indexes = batch_indexes[:, None].expand((batch_size, batch_word_pieces_cum_length.size(1)))

        for l_index, l in enumerate(self.layers):
            output_layer = bert_outputs[l]
            real_word_piece_length = output_layer.size(1) - 2
            if max_word_piece_length > real_word_piece_length:  # 如果实际上是截取出来的
                paddings = output_layer.new_zeros(batch_size,
                                                  max_word_piece_length - real_word_piece_length,
                                                  output_layer.size(2))
                output_layer = torch.cat((output_layer, paddings), dim=1).contiguous()
            # 从word_piece collapse到word的表示
            truncate_output_layer = output_layer[:, 1:-1]  # 删除[CLS]与[SEP] batch_size x len x hidden_size
            if self.pool_method == 'first':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp

            elif self.pool_method == 'last':
                tmp = truncate_output_layer[_batch_indexes, batch_word_pieces_cum_length]
                tmp = tmp.masked_fill(word_mask[:, :batch_word_pieces_cum_length.size(1), None].eq(False), 0)
                outputs[l_index, :, s_shift:batch_word_pieces_cum_length.size(1)+s_shift] = tmp
            elif self.pool_method == 'max':
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift], _ = torch.max(truncate_output_layer[i, start:end], dim=-2)
            else:
                for i in range(batch_size):
                    for j in range(seq_len[i]):
                        start, end = batch_word_pieces_cum_length[i, j], batch_word_pieces_cum_length[i, j + 1]
                        outputs[l_index, i, j + s_shift] = torch.mean(truncate_output_layer[i, start:end], dim=-2)
            if self.include_cls_sep:
                if l in (len(bert_outputs) - 1, -1) and self.pooled_cls:
                    outputs[l_index, :, 0] = pooled_cls
                else:
                    outputs[l_index, :, 0] = output_layer[:, 0]
                outputs[l_index, batch_indexes, seq_len + s_shift] = output_layer[batch_indexes, word_pieces_lengths + s_shift]

        # 3. 最终的embedding结果
        return outputs
    
    
    
# tokenizer
VOCAB_NAME = 'vocab.txt'

PRETRAINED_INIT_CONFIGURATION = {
    "en": {"do_lower_case": False},
    "en-base-uncased": {'do_lower_case': True},
    'en-base-cased': {'do_lower_case':False},
    "en-large-cased-wwm": {"do_lower_case": False},
    'en-large-cased': {'do_lower_case':False},
    'en-large-uncased': {'do_lower_case':True},
    'en-large-uncased-wwm': {'do_lower_case':True},
    'cn': {'do_lower_case':True},
    'cn-base': {'do_lower_case': True},
    'cn-wwm-ext': {'do_lower_case': True},
    'multi-base-cased': {'do_lower_case': False},
    'multi-base-uncased': {'do_lower_case': True},
}

def _is_control(char):
    r"""Checks whether `chars` is a control character."""
    # These are technically control characters but we count them as whitespace
    # characters.
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False


def _is_punctuation(char):
    r"""Checks whether `chars` is a punctuation character."""
    cp = ord(char)
    # We treat all non-letter/number ASCII as punctuation.
    # Characters such as "^", "$", and "`" are not in the Unicode
    # Punctuation class but we treat them as punctuation anyways, for
    # consistency.
    if (((cp >= 33) and (cp <= 47)) or ((cp >= 58) and (cp <= 64)) or
       ((cp >= 91) and (cp <= 96)) or ((cp >= 123) and (cp <= 126))):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False


def _is_whitespace(char):
    r"""Checks whether `chars` is a whitespace character."""
    # \t, \n, and \r are technically contorl characters but we treat them
    # as whitespace since they are generally considered as such.
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False


def whitespace_tokenize(text):
    r"""Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

def load_vocab(vocab_file):
    r"""Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


class BertTokenizer(object):
    r"""Runs end-to-end tokenization: punctuation splitting + wordpiece"""

    def __init__(self, vocab_file, do_lower_case=True, max_len=None, do_basic_tokenize=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        r"""Constructs a BertTokenizer.

        Args:
          vocab_file: Path to a one-wordpiece-per-line vocabulary file
          do_lower_case: Whether to lower case the input
                         Only has an effect when do_wordpiece_only=False
          do_basic_tokenize: Whether to do basic tokenization before wordpiece.
          max_len: An artificial maximum length to truncate tokenized sequences to;
                         Effective maximum length is always the minimum of this
                         value (if specified) and the underlying BERT model's
                         sequence length.
          never_split: List of tokens which will never be split during tokenization.
                         Only has an effect when do_wordpiece_only=False
        """
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'. To load the vocabulary from a Google pretrained "
                "model use `tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`".format(vocab_file))
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()])
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(do_lower_case=do_lower_case,
                                                  never_split=never_split)
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)
        self.max_len = max_len if max_len is not None else int(1e12)

    @property
    def unk_index(self):
        return self.vocab['[UNK]']

    @property
    def pad_index(self):
        return self.vocab['[PAD]']

    @property
    def cls_index(self):
        return self.vocab['[CLS]']

    @property
    def sep_index(self):
        return self.vocab['[SEP]']

    def _reinit_on_new_vocab(self, vocab):
        r"""
        在load bert之后，可能会对vocab进行重新排列。重新排列之后调用这个函数重新初始化与vocab相关的性质

        :param vocab:
        :return:
        """
        self.vocab = vocab
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab)

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        r"""Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab[token])
        if len(ids) > self.max_len:
            logger.warning(
                "Token indices sequence length is longer than the specified maximum "
                " sequence length for this BERT model ({} > {}). Running this"
                " sequence through BERT will result in indexing errors".format(len(ids), self.max_len)
            )
        return ids

    def convert_ids_to_tokens(self, ids):
        r"""将token ids转换为一句话"""
        tokens = []
        for i in ids:
            tokens.append(self.ids_to_tokens[i])
        return self._convert_tokens_to_string(tokens)

    def _convert_tokens_to_string(self, tokens):
        """ Converts a sequence of tokens (string) in a single string. """
        out_string = " ".join(tokens).replace(" ##", "").strip()
        return out_string

    def save_vocabulary(self, vocab_path):
        r"""Save the tokenizer vocabulary to a directory or file."""
        index = 0
        if os.path.isdir(vocab_path):
            vocab_file = os.path.join(vocab_path, VOCAB_NAME)
        else:
            vocab_file = vocab_path
        with open(vocab_file, "w", encoding="utf-8") as writer:
            for token, token_index in sorted(self.vocab.items(), key=lambda kv: kv[1]):
                if index != token_index:
                    logger.warning("Saving vocabulary to {}: vocabulary indices are not consecutive."
                                   " Please check that the vocabulary is not corrupted!".format(vocab_file))
                    index = token_index
                writer.write(token + u'\n')
                index += 1
        return vocab_file

    @classmethod
    def from_pretrained(cls, model_dir_or_name, *inputs, **kwargs):
        r"""
        给定模型的名字或者路径，直接读取vocab.
        """
        model_dir = _get_bert_dir(model_dir_or_name)
        pretrained_model_name_or_path = _get_file_name_base_on_postfix(model_dir, '.txt')
        logger.info("loading vocabulary file {}".format(pretrained_model_name_or_path))
        max_len = 512
        kwargs['max_len'] = min(kwargs.get('max_position_embeddings', int(1e12)), max_len)
        # Instantiate tokenizer.
        if 'do_lower_case' not in kwargs:
            if model_dir_or_name in PRETRAINED_INIT_CONFIGURATION:
                kwargs['do_lower_case'] = PRETRAINED_INIT_CONFIGURATION[model_dir_or_name]['do_lower_case']
            else:
                if 'case' in model_dir_or_name:
                    kwargs['do_lower_case'] = False
                elif 'uncase' in model_dir_or_name:
                    kwargs['do_lower_case'] = True

        tokenizer = cls(pretrained_model_name_or_path, *inputs, **kwargs)
        return tokenizer

    def encode(self, text, add_special_tokens=True):
        """
        给定text输入将数据encode为index的形式。

        Example::

            >>> from fastNLP.modules import BertTokenizer
            >>> bert_tokenizer = BertTokenizer.from_pretrained('en')
            >>> print(bert_tokenizer.encode('from'))
            >>> print(bert_tokenizer.encode("This is a demo sentence"))
            >>> print(bert_tokenizer.encode(["This", "is", 'a']))


        :param List[str],str text: 输入的一条认为是一句话。
        :param bool add_special_tokens: 是否保证句首和句尾是cls和sep。
        :return:
        """

        word_pieces = []
        if isinstance(text, str):
            words = text.split()
        elif isinstance(text, list):
            words = text
        else:
            raise TypeError("Only support str or List[str]")
        for word in words:
            _words = self.basic_tokenizer._tokenize_chinese_chars(word).split()
            tokens = []
            for word in _words:
                tokens.extend(self.wordpiece_tokenizer.tokenize(word))
            word_piece_ids = self.convert_tokens_to_ids(tokens)
            word_pieces.extend(word_piece_ids)
        if add_special_tokens:
            if word_pieces[0] != self.cls_index:
                word_pieces.insert(0, self.cls_index)
            if word_pieces[-1] != self.sep_index:
                word_pieces.append(self.sep_index)
        return word_pieces
    
    
def _get_bert_dir(model_dir_or_name: str = 'en-base-uncased'):
    if model_dir_or_name.lower() in PRETRAINED_BERT_MODEL_DIR:
        model_url = _get_embedding_url('bert', model_dir_or_name.lower())
        model_dir = cached_path(model_url, name='embedding')
        # 检查是否存在
    elif os.path.isdir(os.path.abspath(os.path.expanduser(model_dir_or_name))):
        model_dir = os.path.abspath(os.path.expanduser(model_dir_or_name))
    else:
        logger.error(f"Cannot recognize BERT dir or name ``{model_dir_or_name}``.")
        raise ValueError(f"Cannot recognize BERT dir or name ``{model_dir_or_name}``.")
    return str(model_dir)


class BasicTokenizer(object):
    r"""Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self,
                 do_lower_case=True,
                 never_split=("[UNK]", "[SEP]", "[PAD]", "[CLS]", "[MASK]")):
        r"""Constructs a BasicTokenizer.

        Args:
          do_lower_case: Whether to lower case the input.
        """
        self.do_lower_case = do_lower_case
        self.never_split = never_split

    def tokenize(self, text):
        r"""Tokenizes a piece of text."""
        text = self._clean_text(text)
        # This was added on November 1st, 2018 for the multilingual and Chinese
        # models. This is also applied to the English models now, but it doesn't
        # matter since the English models were not trained on any Chinese data
        # and generally don't have any Chinese data in them (there are Chinese
        # characters in the vocabulary because Wikipedia does have some Chinese
        # words in the English Wikipedia.).
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

    def _run_strip_accents(self, text):
        r"""Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def _run_split_on_punc(self, text):
        r"""Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        r"""Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.append(" ")
                output.append(char)
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        r"""Checks whether CP is the codepoint of a CJK character."""
        # This defines a "chinese character" as anything in the CJK Unicode block:
        #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
        #
        # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
        # despite its name. The modern Korean Hangul alphabet is a different block,
        # as is Japanese Hiragana and Katakana. Those alphabets are used to write
        # space-separated words, so they are not treated specially and handled
        # like the all of the other languages.
        if (((cp >= 0x4E00) and (cp <= 0x9FFF)) or  #
            ((cp >= 0x3400) and (cp <= 0x4DBF)) or  #
            ((cp >= 0x20000) and (cp <= 0x2A6DF)) or  #
            ((cp >= 0x2A700) and (cp <= 0x2B73F)) or  #
            ((cp >= 0x2B740) and (cp <= 0x2B81F)) or  #
            ((cp >= 0x2B820) and (cp <= 0x2CEAF)) or
            ((cp >= 0xF900) and (cp <= 0xFAFF)) or  #
            ((cp >= 0x2F800) and (cp <= 0x2FA1F))):  #
            return True

        return False

    def _clean_text(self, text):
        r"""Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)
    
class WordpieceTokenizer(object):
    r"""Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="[UNK]", max_input_chars_per_word=100):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        r"""Tokenizes a piece of text into its word pieces.

        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.

        For example:
          input = "unaffable"
          output = ["un", "##aff", "##able"]

        Args:
          text: A single token or whitespace separated tokens. This should have
            already been passed through `BasicTokenizer`.

        Returns:
          A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []
            while start < len(chars):
                end = len(chars)
                cur_substr = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_substr = substr
                        break
                    end -= 1
                if cur_substr is None:
                    is_bad = True
                    break
                sub_tokens.append(cur_substr)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        if len(output_tokens) == 0:  # 防止里面全是空格或者回车符号
            return [self.unk_token]
        return output_tokens

