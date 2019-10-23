"""
Acknowledgment:
We refer to https://github.com/ZichaoHuang/TransE to implement the variant of TransEdge for link prediction.
"""
import ast
import time
import argparse
import tensorflow as tf
import multiprocessing as mp
import numpy as np
from model_funcs import embed_init, xavier_init, limit_loss
from test_funcs import early_stop
from sklearn.metrics.pairwise import euclidean_distances
from context_operator import context_compression, context_projection
from kg_lp import KnowledgeGraph


class TransEdge_LP:
    def __init__(self, kg, args):
        self.kg = kg
        self.args = args
        self.embed_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate
        self.n_generator = args.n_generator
        self.n_rank_calculator = args.n_rank_calculator
        self.train_op = None
        self.loss = None

        self.operator = context_projection
        if self.args.mode == 'compression':
            self.operator = context_compression

        self.global_step = tf.Variable(initial_value=0, trainable=False, name='global_step')
        '''ops for evaluation'''
        self.eval_triple = tf.placeholder(dtype=tf.int32, shape=[3])
        self.idx_head_prediction = None
        self.idx_tail_prediction = None

        gpu_config = tf.GPUOptions(allow_growth=True)
        sess_config = tf.ConfigProto(gpu_options=gpu_config)
        self.session = tf.Session(config=sess_config)

        self._generate_variables()
        self._generate_graph()
        self._build_eval_graph()
        print('-----Initializing tf graph-----')
        tf.global_variables_initializer().run(session=self.session)
        print('-----Initialization accomplished-----')

    def _generate_variables(self):
        with tf.variable_scope('relation' + 'embedding'):
            self.ent_embeds = xavier_init(self.kg.n_entity, self.embed_dim, "ent_embeds", is_l2=self.args.ent_norm)
            self.rel_embeds = xavier_init(self.kg.n_relation, self.embed_dim, "rel_embeds", is_l2=self.args.rel_norm)
            self.interact_ent_embeds = xavier_init(self.kg.n_entity, self.embed_dim, "head_ent_embeds",
                                                   is_l2=self.args.interact_ent_norm)

    def _generate_graph(self):
        self.pos_hs = tf.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.placeholder(tf.int32, shape=[None])

        phs = tf.nn.embedding_lookup(self.ent_embeds, self.pos_hs)
        prs = tf.nn.embedding_lookup(self.rel_embeds, self.pos_rs)
        pts = tf.nn.embedding_lookup(self.ent_embeds, self.pos_ts)
        nhs = tf.nn.embedding_lookup(self.ent_embeds, self.neg_hs)
        nrs = tf.nn.embedding_lookup(self.rel_embeds, self.neg_rs)
        nts = tf.nn.embedding_lookup(self.ent_embeds, self.neg_ts)

        c_phs = tf.nn.embedding_lookup(self.interact_ent_embeds, self.pos_hs)
        c_pts = tf.nn.embedding_lookup(self.interact_ent_embeds, self.pos_ts)
        c_nhs = tf.nn.embedding_lookup(self.interact_ent_embeds, self.neg_hs)
        c_nts = tf.nn.embedding_lookup(self.interact_ent_embeds, self.neg_ts)

        prs = self.operator(c_phs, prs, c_pts, self.embed_dim, is_tanh=self.args.op_is_tanh,
                            is_norm=self.args.op_is_norm, layers=self.args.mlp_layers, act=self.args.act)
        nrs = self.operator(c_nhs, nrs, c_nts, self.embed_dim, is_tanh=self.args.op_is_tanh,
                            is_norm=self.args.op_is_norm, layers=self.args.mlp_layers, act=self.args.act)
        self.loss = limit_loss(phs, prs, pts, nhs, nrs, nts,
                               self.args.pos_margin, self.args.neg_margin, self.args.neg_param)
        optimizer = tf.train.AdagradOptimizer(learning_rate=self.learning_rate)
        self.train_op = optimizer.minimize(self.loss, global_step=self.global_step)

    def _build_eval_graph(self):
        with tf.name_scope('evaluation'):
            self.idx_head_prediction, self.idx_tail_prediction = self.evaluate(self.eval_triple)

    def evaluate(self, eval_triple):
        with tf.name_scope('lookup'):
            head = tf.nn.embedding_lookup(self.ent_embeds, [eval_triple[0]])
            tail = tf.nn.embedding_lookup(self.ent_embeds, [eval_triple[1]])
            head = tf.tile(head, [self.kg.n_entity, 1])
            tail = tf.tile(tail, [self.kg.n_entity, 1])

            head_ent_embeds = self.interact_ent_embeds
            tail_ent_embeds = self.interact_ent_embeds
            c_head = tf.nn.embedding_lookup(self.interact_ent_embeds, [eval_triple[0]])
            c_tail = tf.nn.embedding_lookup(self.interact_ent_embeds, [eval_triple[1]])

            c_head = tf.tile(c_head, [self.kg.n_entity, 1])
            c_tail = tf.tile(c_tail, [self.kg.n_entity, 1])

            relation = tf.nn.embedding_lookup(self.rel_embeds, [eval_triple[2]])
            relation = tf.tile(relation, [self.kg.n_entity, 1])

        with tf.name_scope('link'):
            distance_head_prediction = self.ent_embeds + self.operator(head_ent_embeds, relation, c_tail,
                                                                       self.embed_dim,
                                                                       is_tanh=self.args.op_is_tanh,
                                                                       is_norm=self.args.op_is_norm,
                                                                       layers=self.args.mlp_layers,
                                                                       act=self.args.act) - tail
            distance_tail_prediction = head + self.operator(c_head, relation, tail_ent_embeds, self.embed_dim,
                                                            is_tanh=self.args.op_is_tanh,
                                                            is_norm=self.args.op_is_norm,
                                                            layers=self.args.mlp_layers,
                                                            act=self.args.act) - self.ent_embeds
        with tf.name_scope('rank'):
            _, idx_head_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_head_prediction), axis=1),
                                                 k=self.kg.n_entity)
            _, idx_tail_prediction = tf.nn.top_k(tf.reduce_sum(tf.square(distance_tail_prediction), axis=1),
                                                 k=self.kg.n_entity)
        return idx_head_prediction, idx_tail_prediction

    def launch_training(self, neighbor, epoch):
        raw_batch_queue = mp.Queue()
        training_batch_queue = mp.Queue()
        for _ in range(self.n_generator):
            mp.Process(target=self.kg.generate_training_batch, kwargs={'in_queue': raw_batch_queue,
                                                                       'out_queue': training_batch_queue,
                                                                       "neighbor": neighbor,
                                                                       "n_triple": self.args.n_neg_triple}).start()
        # print('-----Start training-----')
        start = time.time()
        n_batch = 0
        for raw_batch in self.kg.next_raw_batch(self.batch_size):
            raw_batch_queue.put(raw_batch)
            n_batch += 1
        for _ in range(self.n_generator):
            raw_batch_queue.put(None)
        # print('-----Constructing training batches-----')
        epoch_loss = 0
        n_used_triple = 0
        for i in range(n_batch):
            batch_pos, batch_neg = training_batch_queue.get()
            batch_loss, _ = self.session.run(fetches=[self.loss, self.train_op],
                                             feed_dict={self.pos_hs: [x[0] for x in batch_pos],
                                                        self.pos_rs: [x[2] for x in batch_pos],
                                                        self.pos_ts: [x[1] for x in batch_pos],
                                                        self.neg_hs: [x[0] for x in batch_neg],
                                                        self.neg_rs: [x[2] for x in batch_neg],
                                                        self.neg_ts: [x[1] for x in batch_neg]})

            epoch_loss += batch_loss
            n_used_triple += len(batch_pos)
        print('Epoch {}, epoch loss: {:.4f}, cost time: {:.4f}s'.format(epoch, epoch_loss, time.time() - start))

    def launch_evaluation(self):
        print('-----Start evaluation-----')
        start = time.time()
        eval_result_queue = mp.JoinableQueue()
        rank_result_queue = mp.Queue()
        for _ in range(self.n_rank_calculator):
            mp.Process(target=self.calculate_rank, kwargs={'in_queue': eval_result_queue,
                                                           'out_queue': rank_result_queue}).start()
        n_used_eval_triple = 0
        for eval_triple in self.kg.test_triples:
            idx_head_prediction, idx_tail_prediction = self.session.run(fetches=[self.idx_head_prediction,
                                                                                 self.idx_tail_prediction],
                                                                        feed_dict={self.eval_triple: eval_triple})
            eval_result_queue.put((eval_triple, idx_head_prediction, idx_tail_prediction))
            n_used_eval_triple += 1
        print()
        for _ in range(self.n_rank_calculator):
            eval_result_queue.put(None)
        print('-----Joining all rank calculator-----')
        eval_result_queue.join()
        print('-----All rank calculation accomplished-----')
        print('-----Obtaining evaluation results-----')
        '''Raw'''
        head_mr_raw = 0
        head_mrr_raw = 0
        head_hits1_raw = 0
        head_hits3_raw = 0
        head_hits5_raw = 0
        head_hits10_raw = 0
        tail_mr_raw = 0
        tail_mrr_raw = 0
        tail_hits1_raw = 0
        tail_hits3_raw = 0
        tail_hits5_raw = 0
        tail_hits10_raw = 0
        '''Filter'''
        head_mr_filter = 0
        head_mrr_filter = 0
        head_hits1_filter = 0
        head_hits3_filter = 0
        head_hits5_filter = 0
        head_hits10_filter = 0
        tail_mr_filter = 0
        tail_mrr_filter = 0
        tail_hits1_filter = 0
        tail_hits3_filter = 0
        tail_hits5_filter = 0
        tail_hits10_filter = 0
        for _ in range(n_used_eval_triple):
            head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter = rank_result_queue.get()
            head_mr_raw += (head_rank_raw + 1)
            head_mrr_raw += 1 / (head_rank_raw + 1)
            if head_rank_raw < 1:
                head_hits1_raw += 1
            if head_rank_raw < 3:
                head_hits3_raw += 1
            if head_rank_raw < 5:
                head_hits5_raw += 1
            if head_rank_raw < 10:
                head_hits10_raw += 1
            tail_mr_raw += (tail_rank_raw + 1)
            tail_mrr_raw += 1 / (tail_rank_raw + 1)
            if tail_rank_raw < 1:
                tail_hits1_raw += 1
            if tail_rank_raw < 3:
                tail_hits3_raw += 1
            if tail_rank_raw < 5:
                tail_hits5_raw += 1
            if tail_rank_raw < 10:
                tail_hits10_raw += 1
            head_mr_filter += (head_rank_filter + 1)
            head_mrr_filter += 1 / (head_rank_filter + 1)
            if head_rank_filter < 1:
                head_hits1_filter += 1
            if head_rank_filter < 3:
                head_hits3_filter += 1
            if head_rank_filter < 5:
                head_hits5_filter += 1
            if head_rank_filter < 10:
                head_hits10_filter += 1
            tail_mr_filter += (tail_rank_filter + 1)
            tail_mrr_filter += 1 / (tail_rank_filter + 1)
            if tail_rank_filter < 1:
                tail_hits1_filter += 1
            if tail_rank_filter < 3:
                tail_hits3_filter += 1
            if tail_rank_filter < 5:
                tail_hits5_filter += 1
            if tail_rank_filter < 10:
                tail_hits10_filter += 1
        print('-----Raw-----')
        head_mr_raw /= n_used_eval_triple
        head_mrr_raw /= n_used_eval_triple
        head_hits1_raw /= n_used_eval_triple
        head_hits3_raw /= n_used_eval_triple
        head_hits5_raw /= n_used_eval_triple
        head_hits10_raw /= n_used_eval_triple
        tail_mr_raw /= n_used_eval_triple
        tail_mrr_raw /= n_used_eval_triple
        tail_hits1_raw /= n_used_eval_triple
        tail_hits3_raw /= n_used_eval_triple
        tail_hits5_raw /= n_used_eval_triple
        tail_hits10_raw /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_raw, head_mrr_raw, head_hits1_raw, head_hits3_raw, head_hits5_raw, head_hits10_raw))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_raw, tail_mrr_raw, tail_hits1_raw, tail_hits3_raw, tail_hits5_raw, tail_hits10_raw))
        print('------Average------')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_raw + tail_mr_raw) / 2,
            (head_mrr_raw + tail_mrr_raw) / 2,
            (head_hits1_raw + tail_hits1_raw) / 2,
            (head_hits3_raw + tail_hits3_raw) / 2,
            (head_hits5_raw + tail_hits5_raw) / 2,
            (head_hits10_raw + tail_hits10_raw) / 2))
        print('-----Filter-----')
        head_mr_filter /= n_used_eval_triple
        head_mrr_filter /= n_used_eval_triple
        head_hits1_filter /= n_used_eval_triple
        head_hits3_filter /= n_used_eval_triple
        head_hits5_filter /= n_used_eval_triple
        head_hits10_filter /= n_used_eval_triple
        tail_mr_filter /= n_used_eval_triple
        tail_mrr_filter /= n_used_eval_triple
        tail_hits1_filter /= n_used_eval_triple
        tail_hits3_filter /= n_used_eval_triple
        tail_hits5_filter /= n_used_eval_triple
        tail_hits10_filter /= n_used_eval_triple
        print('-----Head prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            head_mr_filter, head_mrr_filter, head_hits1_filter, head_hits3_filter, head_hits5_filter,
            head_hits10_filter))
        print('-----Tail prediction-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            tail_mr_filter, tail_mrr_filter, tail_hits1_filter, tail_hits3_filter, tail_hits5_filter,
            tail_hits10_filter))
        print('-----Average-----')
        print('MR: {:.4f}, MRR: {:.4f}, Hits@1: {:.4f}, Hits@3: {:.4f}, Hits@5: {:.4f}, Hits@10: {:.4f}'.format(
            (head_mr_filter + tail_mr_filter) / 2,
            (head_mrr_filter + tail_mrr_filter) / 2,
            (head_hits1_filter + tail_hits1_filter) / 2,
            (head_hits3_filter + tail_hits3_filter) / 2,
            (head_hits5_filter + tail_hits5_filter) / 2,
            (head_hits10_filter + tail_hits10_filter) / 2))
        print('cost time: {:.4f}s'.format(time.time() - start))
        print('-----Finish evaluation-----')
        return (head_hits1_filter + tail_hits1_filter) / 2

    def calculate_rank(self, in_queue, out_queue):
        while True:
            idx_predictions = in_queue.get()
            if idx_predictions is None:
                in_queue.task_done()
                return
            else:
                eval_triple, idx_head_prediction, idx_tail_prediction = idx_predictions
                head, tail, relation = eval_triple
                head_rank_raw = 0
                tail_rank_raw = 0
                head_rank_filter = 0
                tail_rank_filter = 0
                for candidate in idx_head_prediction[::-1]:
                    if candidate == head:
                        break
                    else:
                        head_rank_raw += 1
                        if (candidate, tail, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            head_rank_filter += 1
                for candidate in idx_tail_prediction[::-1]:
                    if candidate == tail:
                        break
                    else:
                        tail_rank_raw += 1
                        if (head, candidate, relation) in self.kg.golden_triple_pool:
                            continue
                        else:
                            tail_rank_filter += 1
                out_queue.put((head_rank_raw, tail_rank_raw, head_rank_filter, tail_rank_filter))
                in_queue.task_done()

    def cal_neighbours(self, k, norm=True):
        if k >= 0.99999:
            return dict()
        truncated_num = int(self.kg.n_entity * (1 - k))
        print("truncated num:", truncated_num)
        t = time.time()
        dic = dict()
        embed = self.ent_embeds.eval(session=self.session)
        if norm:
            sim_mat = np.matmul(embed, embed.T)
        else:
            sim_mat = 1 - euclidean_distances(embed, embed)
        sort_index = np.mat(np.argpartition(-sim_mat, truncated_num + 1, axis=1))
        for i in range(self.kg.n_entity):
            dic[i] = sort_index[i, range(truncated_num + 1)].A1
        print("generate neighbors: {:.4f}s".format(time.time() - t))
        return dic


def main():
    parser = argparse.ArgumentParser(description='TransEdge-LP')
    parser.add_argument('--mode', type=str, default='projection', choices=('projection', 'compression'))
    parser.add_argument('--data_dir', type=str, default='../data/WN18RR/')  # '../data/FB15k-237/'

    parser.add_argument('--n_generator', type=int, default=4)
    parser.add_argument('--n_rank_calculator', type=int, default=6)

    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=500)

    parser.add_argument('--embedding_dim', type=int, default=500)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2000)

    parser.add_argument('--truncated_epsilon', type=float, default=1.0)
    parser.add_argument('--truncated_frequency', type=int, default=10)
    parser.add_argument('--n_neg_triple', type=int, default=30)
    parser.add_argument('--neg_triple_rel_scope', type=ast.literal_eval, default=True)

    parser.add_argument('--if_add_rev', type=ast.literal_eval, default=True)

    parser.add_argument('--mlp_layers', type=int, default=2)
    parser.add_argument('--act', type=str, default='tanh', choices=('linear', 'relu', 'gelu', 'tanh', 'sigmoid'))

    parser.add_argument('--pos_margin', type=float, default=0.2)
    parser.add_argument('--neg_param', type=float, default=0.5)
    parser.add_argument('--neg_margin', type=float, default=3.5)

    parser.add_argument('--op_is_tanh', type=ast.literal_eval, default=False)
    parser.add_argument('--op_is_norm', type=ast.literal_eval, default=False)

    parser.add_argument('--ent_norm', type=ast.literal_eval, default=False)
    parser.add_argument('--interact_ent_norm', type=ast.literal_eval, default=False)
    parser.add_argument('--rel_norm', type=ast.literal_eval, default=False)

    args = parser.parse_args()
    print(args)
    kg = KnowledgeGraph(args.data_dir, args.neg_triple_rel_scope, args.if_add_rev)
    kge_model = TransEdge_LP(kg, args)
    pre_pre_hits1, pre_hits1 = 0, 0
    neighbor = kge_model.cal_neighbours(args.truncated_epsilon)
    for epoch in range(1, args.max_epoch + 1):
        kge_model.launch_training(neighbor, epoch)
        if epoch >= args.start_eval and epoch % args.eval_freq == 0:
            hits1 = kge_model.launch_evaluation()
            pre_pre_hits1, pre_hits1, is_early = early_stop(pre_pre_hits1, pre_hits1, hits1)
            if is_early:
                exit(0)
        if args.truncated_epsilon < 1.0 and epoch % args.truncated_frequency == 0:
            neighbor = kge_model.cal_neighbours(args.truncated_epsilon, norm=args.ent_norm)


if __name__ == '__main__':
    main()
