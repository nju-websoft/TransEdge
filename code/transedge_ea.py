import argparse
import ast
import time
import sys
import tensorflow as tf
import numpy as np

from test_funcs import test_csls
from model_funcs import xavier_init, mul, limit_loss, random_unit_embeddings
from train_funcs import get_model, train_tris_k_epo, train_tris_1epo
from train_bp import bootstrapping
from context_operator import context_projection, context_compression


class TransEdge_EA:
    def __init__(self, args, ent_num, rel_num, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, kb1_ents, kb2_ents,
                 out_path):

        self.args = args
        self.embedding_dim = args.embedding_dim
        self.batch_size = args.batch_size
        self.learning_rate = args.learning_rate

        self.ent_num = ent_num
        self.rel_num = rel_num

        self.operator = context_projection
        if self.args.mode == 'compression':
            self.operator = context_compression

        self.seed_sup_ent1 = seed_sup_ent1
        self.seed_sup_ent2 = seed_sup_ent2
        self.ref_ent1 = ref_ent1
        self.ref_ent2 = ref_ent2
        self.kb1_ents = kb1_ents
        self.kb2_ents = kb2_ents

        self.ppre_hits1, self.pre_hits1 = -1, -1
        self.early = False
        self.out_path = out_path

        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.session = tf.Session(config=config)

        self._generate_variables()
        self._generate_graph()
        self._generate_alignment_graph()

        tf.global_variables_initializer().run(session=self.session)

    def _generate_variables(self):
        with tf.variable_scope('relation' + 'embedding'):
            self.ent_embeddings = xavier_init(self.ent_num, self.embedding_dim, "ent_embeds",
                                              is_l2=self.args.ent_norm)
            self.rel_embeddings = xavier_init(self.rel_num, self.embedding_dim, "rel_embeds",
                                              is_l2=self.args.rel_norm)
            '''
            We find that both the general entity embeddings and interaction embeddings contribute to entity alignment. 
            Thus, we let them have a mapping relationship, which would allow the general embeddings reap the benefits.
            '''
            self.interact_ent_embeds = self._interaction(self.ent_embeddings)
            '''
            For simplicity, we let 'general embeddings = interaction embeddings' to enable them benefit from each other.
            The two strategies lead to similar performance on entity alignment.
            '''
            # self.interact_ent_embeds = self.ent_embeddings

    def _interaction(self, embeds):
        embeds = tf.layers.dense(embeds, self.embedding_dim,
                                 kernel_initializer=tf.orthogonal_initializer(),
                                 activation=tf.tanh, name='interaction', reuse=tf.AUTO_REUSE)
        if self.args.interact_ent_norm:
            return tf.nn.l2_normalize(embeds, 1)
        return embeds

    def generate_optimizer(self, loss, var_list=None):
        optimizer = tf.train.AdagradOptimizer(self.learning_rate).minimize(loss, var_list=var_list)
        return optimizer

    def _generate_graph(self):
        self.pos_hs = tf.placeholder(tf.int32, shape=[None])
        self.pos_rs = tf.placeholder(tf.int32, shape=[None])
        self.pos_ts = tf.placeholder(tf.int32, shape=[None])
        self.neg_hs = tf.placeholder(tf.int32, shape=[None])
        self.neg_rs = tf.placeholder(tf.int32, shape=[None])
        self.neg_ts = tf.placeholder(tf.int32, shape=[None])

        phs = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_hs)
        prs = tf.nn.embedding_lookup(self.rel_embeddings, self.pos_rs)
        pts = tf.nn.embedding_lookup(self.ent_embeddings, self.pos_ts)
        nhs = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_hs)
        nrs = tf.nn.embedding_lookup(self.rel_embeddings, self.neg_rs)
        nts = tf.nn.embedding_lookup(self.ent_embeddings, self.neg_ts)

        c_phs = tf.nn.embedding_lookup(self.interact_ent_embeds, self.pos_hs)
        c_pts = tf.nn.embedding_lookup(self.interact_ent_embeds, self.pos_ts)
        c_nhs = tf.nn.embedding_lookup(self.interact_ent_embeds, self.neg_hs)
        c_nts = tf.nn.embedding_lookup(self.interact_ent_embeds, self.neg_ts)

        prs = self.operator(c_phs, prs, c_pts, self.embedding_dim, is_tanh=self.args.op_is_tanh,
                            is_norm=self.args.op_is_norm, layers=self.args.mlp_layers, act=self.args.act,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        nrs = self.operator(c_nhs, nrs, c_nts, self.embedding_dim, is_tanh=self.args.op_is_tanh,
                            is_norm=self.args.op_is_norm, layers=self.args.mlp_layers, act=self.args.act,
                            initializer=tf.contrib.layers.xavier_initializer(uniform=True))
        loss = limit_loss(phs, prs, pts, nhs, nrs, nts, self.args.pos_margin, self.args.neg_margin, self.args.neg_param)

        self.triple_loss = loss
        self.triple_optimizer = self.generate_optimizer(self.triple_loss)

    def _generate_alignment_graph(self):
        self.align_ents1 = tf.placeholder(tf.int32, shape=[None])
        self.align_ents2 = tf.placeholder(tf.int32, shape=[None])
        ents1 = tf.nn.embedding_lookup(self.ent_embeddings, self.align_ents1)
        ents2 = tf.nn.embedding_lookup(self.ent_embeddings, self.align_ents2)
        self.alignment_loss = self.args.bp_param * tf.reduce_sum(tf.reduce_sum(tf.square(ents1 - ents2), axis=1))
        self.alignment_optimizer = self.generate_optimizer(self.alignment_loss)

    def test(self, is_save=False, is_final_save=False):
        t1 = time.time()
        self.ppre_hits1, self.pre_hits1, self.is_early = test_csls(self.eval_ent_embeds(),
                                                                   self.eval_rel_embeds(),
                                                                   self.eval_kb1_test_embed(),
                                                                   self.eval_kb2_test_embed(),
                                                                   self.ppre_hits1,
                                                                   self.pre_hits1,
                                                                   self.out_path,
                                                                   self.args.n_rank_calculator,
                                                                   csls=self.args.csls,
                                                                   is_save=is_save,
                                                                   is_final_save=is_final_save)
        print("testing ent alignment costs: {:.3f} s\n".format(time.time() - t1))
        return self.is_early

    def eval_ent_embeds(self):
        return self.ent_embeddings.eval(session=self.session)

    def eval_rel_embeds(self):
        return self.rel_embeddings.eval(session=self.session)

    def eval_kb1_embed(self):
        return tf.nn.embedding_lookup(self.ent_embeddings, self.kb1_ents).eval(session=self.session)

    def eval_kb2_embed(self):
        return tf.nn.embedding_lookup(self.ent_embeddings, self.kb2_ents).eval(session=self.session)

    def eval_kb1_test_embed(self):
        embed1 = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent1)
        return tf.nn.l2_normalize(embed1, 1).eval(session=self.session)

    def eval_kb2_test_embed(self):
        embed2 = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent2)
        return tf.nn.l2_normalize(embed2, 1).eval(session=self.session)

    def eval_ref_sim_mat(self):
        refs1_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent1)
        refs2_embeddings = tf.nn.embedding_lookup(self.ent_embeddings, self.ref_ent2)
        refs1_embeddings = tf.nn.l2_normalize(refs1_embeddings, 1)
        refs2_embeddings = tf.nn.l2_normalize(refs2_embeddings, 1)
        return mul(refs1_embeddings, refs2_embeddings, self.session, len(self.ref_ent1), False)

    def save(self):
        # np.save(folder + 'ent1_embeds_' + suffix + '.npy', self.eval_kb1_embed())
        # np.save(folder + 'ent2_embeds_' + suffix + '.npy', self.eval_kb2_embed())
        np.save(self.out_path + 'ent_embeds.npy', self.eval_ent_embeds())


def train():
    parser = argparse.ArgumentParser(description='TransEdge-EA')
    parser.add_argument('--mode', type=str, default='projection', choices=('projection', 'compression'))

    parser.add_argument('--data_dir', type=str, default='../data/DBP15K/zh_en/0_3/')
    # parser.add_argument('--data_dir', type=str, default='../data/DBP15K/ja_en/0_3/')
    # parser.add_argument('--data_dir', type=str, default='../data/DBP15K/fr_en/0_3/')
    # parser.add_argument('--data_dir', type=str, default='../data/DWY100K/dbp_wd/sharing/0_3/')

    parser.add_argument('--n_generator', type=int, default=3)
    parser.add_argument('--n_rank_calculator', type=int, default=4)

    parser.add_argument('--eval_freq', type=int, default=10)
    parser.add_argument('--start_eval', type=int, default=0)
    parser.add_argument('--max_epoch', type=int, default=500)

    parser.add_argument('--embedding_dim', type=int, default=75)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=2000)

    parser.add_argument('--truncated_epsilon', type=float, default=0.95)
    parser.add_argument('--n_neg_triple', type=int, default=20)

    parser.add_argument('--mlp_layers', type=int, default=1)
    parser.add_argument('--act', type=str, default='tanh', choices=('linear', 'relu', 'gelu', 'tanh', 'sigmoid'))

    parser.add_argument('--pos_margin', type=float, default=0.2)
    parser.add_argument('--neg_param', type=float, default=0.8)
    parser.add_argument('--neg_margin', type=float, default=2.0)

    parser.add_argument('--op_is_tanh', type=bool, default=False)
    parser.add_argument('--op_is_norm', type=bool, default=True)

    parser.add_argument('--ent_norm', type=bool, default=True)
    parser.add_argument('--interact_ent_norm', type=bool, default=True)
    parser.add_argument('--rel_norm', type=bool, default=True)

    parser.add_argument('--frequency', type=int, default=5)
    parser.add_argument('--sim_th', type=float, default=0.7)
    parser.add_argument('--top_k', type=int, default=10)
    parser.add_argument('--is_bp', type=ast.literal_eval, default=False)
    parser.add_argument('--csls', type=int, default=5)
    parser.add_argument('--bp_param', type=float, default=0.05)
    parser.add_argument('--start_bp', type=int, default=2)
    parser.add_argument('--bp_freq', type=int, default=1)

    args = parser.parse_args()
    print(args)

    ori_triples1, ori_triples2, triples1, triples2, model = get_model(args, TransEdge_EA)
    start_time = time.time()
    print(len(ori_triples1.heads))
    trunc_ent_num = int(len(ori_triples1.heads) * (1 - args.truncated_epsilon))
    assert trunc_ent_num > 0
    print("trunc ent num:", trunc_ent_num)
    iters = args.frequency
    stop = False
    labeled_alignment, ents1, ents2 = set(), None, None
    for t in range(1, args.max_epoch // iters + 1):
        print("iteration ", t)
        stop = train_tris_k_epo(model, triples1, triples2, iters, trunc_ent_num, ents1, ents2,
                                args.n_neg_triple, args.batch_size, args.n_generator, args.n_rank_calculator,
                                bp_freq=args.bp_freq)
        if stop:
            break
        if args.is_bp and t >= args.start_bp:
            print()
            labeled_alignment, ents1, ents2 = bootstrapping(model.eval_ref_sim_mat(), model.ref_ent1, model.ref_ent2,
                                                            labeled_alignment, args.sim_th, args.top_k)
            print()
    if not stop:
        model.test(is_save=True)
    print("Total training time = {:.4f} s.".format(time.time() - start_time))
    sys.exit(0)


if __name__ == '__main__':
    t = time.time()
    train()
    print("total time = {:.3f} s".format(time.time() - t))
