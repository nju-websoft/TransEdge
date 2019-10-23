import os
import pandas as pd
import numpy as np
import random


def add_dict_kv(dic, k, v):
    vs = dic.get(k, set())
    vs.add(v)
    dic[k] = vs


def add_dict_kkv(dic, k1, k2, v):
    k2vs = dic.get(k1, dict())
    vs = k2vs.get(k2, set())
    vs.add(v)
    k2vs[k2] = vs
    dic[k1] = k2vs


def dic_kv2list(dic):
    dic_new = dict()
    for k, v in dic.items():
        dic_new[k] = list(v)
    return dic_new


def dic_kv2list_hubu(dic, sett):
    dic_new = dict()
    for k, v in dic.items():
        dic_new[k] = list(sett - v)
    return dic_new


def dic_kkv2list(dic):
    dic_new = dict()
    for kk, kv in dic.items():
        new_kvv_dic = dict()
        for k, v in kv.items():
            new_kvv_dic[k] = list(v)
        dic_new[kk] = new_kvv_dic
    return dic_new


class KnowledgeGraph:
    def __init__(self, data_dir, neg_triple_rel_scope, if_add_rev):
        self.reversed_triples = set()
        self.data_dir = data_dir
        self.entity_dict = {}
        self.entities = []
        self.relations = []
        self.relation_dict = {}
        self.n_entity = 0
        self.n_relation = 0
        self.training_triples = []  # list of triples in the form of (h, t, r)
        self.validation_triples = []
        self.test_triples = []
        self.n_training_triple = 0
        self.n_validation_triple = 0
        self.n_test_triple = 0

        self.neg_triple_rel_scope = neg_triple_rel_scope

        self.r_h_ts_train, self.r_t_hs_train = dict(), dict()
        self.r_hs_train, self.r_ts_train = dict(), dict()
        self.h_ts, self.t_hs = dict(), dict()
        self.h_rs = dict()

        '''load dicts and triples'''
        self.load_dicts()
        self.load_triples()

        if if_add_rev:
            self.add_reversed_triples()
        self.generate_rel_dic()

        '''construct pools after loading'''
        self.training_triple_pool = set(self.training_triples)
        self.golden_triple_pool = set(self.training_triples) | set(self.validation_triples) | set(self.test_triples)

    def load_dicts(self):
        entity_dict_file = 'entity2id.txt'
        relation_dict_file = 'relation2id.txt'
        print('-----Loading entity dict-----')
        entity_df = pd.read_table(os.path.join(self.data_dir, entity_dict_file), header=None)
        self.entity_dict = dict(zip(entity_df[0], entity_df[1]))
        self.n_entity = len(self.entity_dict)
        self.entities = list(self.entity_dict.values())
        print('#entity: {}'.format(self.n_entity))
        print('-----Loading relation dict-----')
        relation_df = pd.read_table(os.path.join(self.data_dir, relation_dict_file), header=None)
        self.relation_dict = dict(zip(relation_df[0], relation_df[1]))
        self.relations = list(self.relation_dict.values())
        self.n_relation = len(self.relation_dict)
        print('#relation: {}'.format(self.n_relation))

    def load_triples(self):
        training_file = 'train.txt'
        validation_file = 'valid.txt'
        test_file = 'test.txt'
        print('-----Loading training triples-----')
        training_df = pd.read_table(os.path.join(self.data_dir, training_file), header=None)
        self.training_triples = list(zip([self.entity_dict[h] for h in training_df[0]],
                                         [self.entity_dict[t] for t in training_df[1]],
                                         [self.relation_dict[r] for r in training_df[2]]))
        self.n_training_triple = len(self.training_triples)
        print('#training triple: {}'.format(self.n_training_triple))
        print('-----Loading validation triples-----')
        validation_df = pd.read_table(os.path.join(self.data_dir, validation_file), header=None)
        self.validation_triples = list(zip([self.entity_dict[h] for h in validation_df[0]],
                                           [self.entity_dict[t] for t in validation_df[1]],
                                           [self.relation_dict[r] for r in validation_df[2]]))
        self.n_validation_triple = len(self.validation_triples)
        print('#validation triple: {}'.format(self.n_validation_triple))
        print('-----Loading test triples------')
        test_df = pd.read_table(os.path.join(self.data_dir, test_file), header=None)
        self.test_triples = list(zip([self.entity_dict[h] for h in test_df[0]],
                                     [self.entity_dict[t] for t in test_df[1]],
                                     [self.relation_dict[r] for r in test_df[2]]))
        self.n_test_triple = len(self.test_triples)
        print('#test triple: {}'.format(self.n_test_triple))

    def next_raw_batch(self, batch_size):
        rand_idx = np.random.permutation(self.n_training_triple)
        start = 0
        while start < self.n_training_triple:
            end = min(start + batch_size, self.n_training_triple)
            yield [self.training_triples[i] for i in rand_idx[start:end]]
            start = end

    def generate_training_batch(self, in_queue, out_queue, neighbor, n_triple=1):
        boundary_max = 10
        while True:
            raw_batch = in_queue.get()
            if raw_batch is None:
                return
            else:
                batch_pos = raw_batch
                batch_neg = []
                for head, tail, relation in batch_pos:
                    if len(neighbor) == 0:
                        if self.neg_triple_rel_scope:
                            head_candidates = self.r_hs_train.get(relation, self.entities)
                            tail_candidates = self.r_ts_train.get(relation, self.entities)
                        else:
                            head_candidates = self.entities
                            tail_candidates = self.entities
                    else:
                        head_candidates = neighbor.get(head, self.entities)
                        tail_candidates = neighbor.get(tail, self.entities)

                    head_num = len(head_candidates)
                    tail_num = len(tail_candidates)
                    boundary = head_num / (head_num + tail_num) * boundary_max

                    for i in range(n_triple):
                        while True:
                            head_neg = head
                            tail_neg = tail
                            rel_neg = relation
                            prob = random.uniform(0, boundary_max)
                            if prob <= boundary:
                                head_neg = random.choice(head_candidates)
                            else:
                                tail_neg = random.choice(tail_candidates)
                            if (head_neg, tail_neg, rel_neg) not in self.training_triple_pool:
                                break
                        batch_neg.append((head_neg, tail_neg, relation))
                out_queue.put((batch_pos, batch_neg))

    def add_reversed_triples(self):
        print("before adding reversed triples", len(self.training_triples))
        reversed_rel_set = set()
        for h, t, r in self.training_triples:
            self.reversed_triples.add((t, h, r + self.n_relation))
            reversed_rel_set.add(r + self.n_relation)
        print("reversed_triples", len(self.reversed_triples))
        self.training_triples.extend(list(self.reversed_triples))
        print("after adding reversed triples", len(self.training_triples))
        self.n_relation = 2 * self.n_relation
        self.relations = list(set(self.relations) | reversed_rel_set)
        print("after adding reversed triples, relations", len(self.relations))
        self.n_training_triple = len(self.training_triples)

    def generate_rel_dic(self):
        for (h, t, r) in self.training_triples:
            add_dict_kv(self.r_hs_train, r, h)
            add_dict_kv(self.r_ts_train, r, t)
            add_dict_kv(self.h_ts, h, t)
            add_dict_kv(self.t_hs, t, h)
            add_dict_kkv(self.r_h_ts_train, r, h, t)
            add_dict_kkv(self.r_t_hs_train, r, t, h)
            add_dict_kv(self.h_rs, h, r)

        self.r_hs_train = dic_kv2list(self.r_hs_train)
        self.r_ts_train = dic_kv2list(self.r_ts_train)
        self.h_rs = dic_kv2list(self.h_rs)
        # self.h_rs = dic_kv2list_hubu(self.h_rs, set(self.relations))
        # self.r_hs_train = dic_kv2list_hubu(self.r_hs_train, set(self.entities))
        # self.r_ts_train = dic_kv2list_hubu(self.r_ts_train, set(self.entities))
        print('r_hs_train', len(self.r_hs_train))
        print('r_ts_train', len(self.r_ts_train))
        print('h_ts', len(self.h_ts))
        print('t_hs', len(self.t_hs))
