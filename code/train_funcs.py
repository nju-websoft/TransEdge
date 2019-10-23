import math
import multiprocessing

import gc
import sys

import numpy as np
import time
import random
import os

import psutil
import scipy as sp

from scipy import io
import multiprocessing as mp
import utils as ut

g = 1000000000


def generate_cross_min_sim(embed1, embed2, ents1, ents2):
    # If |e1| = |e2| = 1, then ||e1-e2||^2_2 = 2(1 - cos(e1, e2)).
    dic1, dic2 = dict(), dict()
    dis_mat = 2 * (1 - np.matmul(embed1, embed2.T))
    # dis_mat = np.sqrt(dis_mat)
    print(dis_mat.min(), dis_mat.max())
    row_min = 2 * dis_mat.min(axis=0)
    col_min = 2 * dis_mat.min(axis=1)
    print(row_min.max(), col_min.max())
    print(row_min)
    print(col_min)
    for i in range(len(ents1)):
        dic1[ents1[i]] = row_min[i]
        dic2[ents2[i]] = col_min[i]
    return dic1, dic2


def generate_out_folder(training_data_path, method_name):
    params = training_data_path.strip('/').split('/')
    path = 'None'
    for p in params:
        if "_" in p:
            path = p
            break
    folder = "../out/" + method_name + '/' + path + "/" + str(time.strftime("%Y%m%d%H%M%S")) + "/"
    return folder


def get_model(args, model):
    read_func = ut.read_input
    out_folder = generate_out_folder(args.data_dir, model.__name__)
    print("output folder:", out_folder)
    # if "15" in folder:
    #     read_func = ut.read_dbp15k_input
    # else:
    #     read_func = ut.read_input
    ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, _, ent_n, rel_n = read_func(args.data_dir)
    # triples1, triples2 = ut.add_sup_triples(ori_triples1, ori_triples2, seed_sup_ent1, seed_sup_ent2)

    my_model = model(args, ent_n, rel_n, seed_sup_ent1, seed_sup_ent2, ref_ent1, ref_ent2, ori_triples1.ent_list,
                     ori_triples2.ent_list,
                     out_folder)
    return ori_triples1, ori_triples2, ori_triples1, ori_triples2, my_model


def train_tris_k_epo(model, tris1, tris2, k, trunc_ent_num, ents1, ents2, nums_neg, batch_size,
                     nums_threads_batch, nums_threads, is_test=True, bp_freq=1):
    if trunc_ent_num > 0:
        t1 = time.time()
        kb1_ent_embed = model.eval_kb1_embed()
        kb2_ent_embed = model.eval_kb2_embed()
        nbours1 = generate_neighbours_multi_embed(kb1_ent_embed, model.kb1_ents, trunc_ent_num, nums_threads)
        nbours2 = generate_neighbours_multi_embed(kb2_ent_embed, model.kb2_ents, trunc_ent_num, nums_threads)
        print("generate neighbours: {:.3f} s, size: {:.6f} G".format(time.time() - t1, 2 * sys.getsizeof(nbours1) / g))
    else:
        nbours1, nbours2 = None, None
    for i in range(k):
        if ents1 is not None and len(ents1) > 0 and i % bp_freq == 0:
            train_alignment_1epo(model, tris1, tris2, ents1, ents2, 1)
        loss, t2 = train_tris_1epo(model, tris1, tris2, nbours1, nbours2, nums_neg, batch_size, nums_threads_batch)
        print("triple_loss = {:.3f}, time = {:.3f} s".format(loss, t2))
    if nbours1 is not None:
        del nbours1, nbours2
        gc.collect()
    if is_test:
        return model.test(is_save=True)
    else:
        return False


def train_tris_1epo(model, triples1, triples2, nbours1, nbours2, nums_neg, batch_size, nums_threads_batch):
    loss = 0
    start = time.time()
    triples_num = triples1.triples_num + triples2.triples_num
    triple_steps = int(math.ceil(triples_num / batch_size))
    stepss = ut.div_list(list(range(triple_steps)), nums_threads_batch)
    assert len(stepss) == nums_threads_batch
    batch_queue = mp.Queue()
    for steps in stepss:
        mp.Process(target=generate_batch_via_neighbour_no_pair_queue, kwargs={'que': batch_queue,
                                                                              'triples1': triples1,
                                                                              "triples2": triples2,
                                                                              "steps": steps,
                                                                              "batch_size": batch_size,
                                                                              "nbours1": nbours1,
                                                                              "nbours2": nbours2,
                                                                              "multi": nums_neg}).start()
    for step in range(triple_steps):
        fetches = {"loss": model.triple_loss, "train_op": model.triple_optimizer}
        batch_pos, batch_neg = batch_queue.get()
        triple_feed_dict = {model.pos_hs: [x[0] for x in batch_pos],
                            model.pos_rs: [x[1] for x in batch_pos],
                            model.pos_ts: [x[2] for x in batch_pos],
                            model.neg_hs: [x[0] for x in batch_neg],
                            model.neg_rs: [x[1] for x in batch_neg],
                            model.neg_ts: [x[2] for x in batch_neg]}
        vals = model.session.run(fetches=fetches, feed_dict=triple_feed_dict)
        loss += vals["loss"]
    loss /= triple_steps
    random.shuffle(triples1.triple_list)
    random.shuffle(triples2.triple_list)
    end = time.time()
    return loss, round(end - start, 2)


def train_alignment_1epo(model, tris1, tris2, ents1, ents2, ep):
    if ents1 is None or len(ents1) == 0:
        return
    start = time.time()
    alignment_fetches = {"loss": model.alignment_loss, "train_op": model.alignment_optimizer}
    alignment_feed_dict = {model.align_ents1: ents1,
                           model.align_ents2: ents2}
    alignment_vals = model.session.run(fetches=alignment_fetches, feed_dict=alignment_feed_dict)
    alignment_loss = alignment_vals["loss"]
    print("alignment_loss = {:.3f}, time = {:.3f} s".format(alignment_loss, time.time() - start))


def cal_neighbours_embed(frags, ent_list, sub_embed, embed, k):
    dic = dict()
    sim_mat = np.matmul(sub_embed, embed.T)

    for i in range(sim_mat.shape[0]):
        sort_index = np.argpartition(-sim_mat[i, :], k + 1)
        dic[frags[i]] = ent_list[sort_index[1:k+1]].tolist()

    del sim_mat
    gc.collect()
    # print("gc costs {:.3f} s, mem change {:.6f} G".format(time.time() - t1, (psutil.virtual_memory().used - m1) / g))
    return dic


def generate_neighbours_multi_embed(embed, ent_list, k, nums_threads):
    ent_frags = ut.div_list(np.array(ent_list), nums_threads)
    ent_frag_indexes = ut.div_list(np.array(range(len(ent_list))), nums_threads)
    pool = multiprocessing.Pool(processes=len(ent_frags))
    results = list()
    for i in range(len(ent_frags)):
        results.append(pool.apply_async(cal_neighbours_embed,
                                        (ent_frags[i], np.array(ent_list), embed[ent_frag_indexes[i], :], embed, k)))
    pool.close()
    pool.join()
    dic = dict()
    for res in results:
        dic = ut.merge_dic(dic, res.get())
    t1 = time.time()
    m1 = psutil.virtual_memory().used
    del embed
    gc.collect()
    # print("gc costs {:.3f} s, mem change {:.6f} G".format(time.time() - t1, (psutil.virtual_memory().used - m1) / g))
    return dic


def trunc_sampling(pos_triples, all_triples, dic, ent_list):
    neg_triples = list()
    for (h, r, t, _) in pos_triples:
        h2, r2, t2 = h, r, t
        while True:
            choice = random.randint(0, 999)
            if choice < 500:
                candidates = dic.get(h, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                h2 = candidates[index]
            elif choice >= 500:
                candidates = dic.get(t, ent_list)
                index = random.sample(range(0, len(candidates)), 1)[0]
                t2 = candidates[index]
            if (h2, r2, t2) not in all_triples:
                break
        neg_triples.append((h2, r2, t2))
    return neg_triples


def trunc_sampling_multi(pos_triples, all_triples, dic, ent_list, multi):
    neg_triples = list()
    ent_list = np.array(ent_list)
    for (h, r, t, _) in pos_triples:
        choice = random.randint(0, 999)
        if choice < 500:
            candidates = dic.get(h, ent_list)
            h2s = random.sample(candidates, multi)
            negs = [(h2, r, t) for h2 in h2s]
            neg_triples.extend(negs)
        elif choice >= 500:
            candidates = dic.get(t, ent_list)
            t2s = random.sample(candidates, multi)
            negs = [(h, r, t2) for t2 in t2s]
            neg_triples.extend(negs)
    neg_triples = list(set(neg_triples) - all_triples)
    return neg_triples


def generate_batch_via_neighbour(triples1, triples2, step, batch_size, neighbours_dic1, neighbours_dic2, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    if len(triples1.ent_list) < 10000:
        for i in range(multi):
            neg_triples.extend(trunc_sampling(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list))
            neg_triples.extend(trunc_sampling(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list))
    else:
        neg_triples.extend(
            trunc_sampling_multi(pos_triples1, triples1.triples, neighbours_dic1, triples1.ent_list, multi))
        neg_triples.extend(
            trunc_sampling_multi(pos_triples2, triples2.triples, neighbours_dic2, triples2.ent_list, multi))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_pos_batch(triples1, triples2, step, batch_size):
    num1 = int(len(triples1) / (len(triples1) + len(triples2)) * batch_size)
    num2 = batch_size - num1
    start1 = step * num1
    start2 = step * num2
    end1 = start1 + num1
    end2 = start2 + num2
    if end1 > len(triples1):
        end1 = len(triples1)
    if end2 > len(triples2):
        end2 = len(triples2)
    pos_triples1 = triples1[start1: end1]
    pos_triples2 = triples2[start2: end2]
    # pos_triples1 = random.sample(triples1, num1)
    # pos_triples2 = random.sample(triples2, num2)
    return pos_triples1, pos_triples2


def generate_neg_triples(pos_triples, triples_data):
    all_triples = triples_data.triples
    ents = triples_data.ent_list
    neg_triples = list()
    # for (h, r, t) in pos_triples:
    #     h2, r2, t2 = h, r, t
    #     while True:
    #         choice = random.randint(0, 999)
    #         if choice < 500:
    #             h2 = random.sample(ents, 1)[0]
    #         elif choice >= 500:
    #             t2 = random.sample(ents, 1)[0]
    #         if (h2, r2, t2) not in all_triples:
    #             break
    #     neg_triples.append((h2, r2, t2))
    for (h, r, t) in pos_triples:
        h2, r2, t2 = h, r, t
        choice = random.randint(0, 999)
        if choice < 500:
            h2 = random.sample(ents, 1)[0]
        elif choice >= 500:
            t2 = random.sample(ents, 1)[0]
        neg_triples.append((h2, r2, t2))
    return neg_triples


def generate_neg_triple_ht(triple, all_triples, ents, ht):
    h2, r2, t2 = triple[0], triple[1], triple[2]
    while True:
        choice = random.randint(0, 999)
        if choice < 500:
            h2 = random.sample(ents, 1)[0]
        elif choice >= 500:
            t2 = random.sample(ents, 1)[0]
        if (h2, r2, t2) not in all_triples and (h2, t2) not in ht:
            return h2, r2, t2


def generate_neg_triples_batch(pos_triples, triples_data, is_head):
    all_triples = triples_data.triples
    ents = triples_data.ent_list
    n = len(pos_triples)
    pos_triples_mat = np.matrix(pos_triples)
    neg_ent_mat = np.matrix(np.random.choice(np.array(ents), n)).T
    if is_head:
        neg_triples_mat = np.column_stack((neg_ent_mat, pos_triples_mat[:, [1, 2]]))
    else:
        neg_triples_mat = np.column_stack((pos_triples_mat[:, [0, 1]], neg_ent_mat))
    ii, jj = neg_triples_mat.shape
    neg_triples = list()
    for i in range(ii):
        tr = (neg_triples_mat[i, 0], neg_triples_mat[i, 1], neg_triples_mat[i, 2])
        if tr not in all_triples:
            neg_triples.append(tr)
            # else:
            #     neg_triples.append(generate_neg_triple_ht(pos_triples[i], all_triples, ents, ht))
    # print("neg triples:", len(neg_triples))
    return neg_triples


def generate_pos_neg_batch(triples1, triples2, step, batch_size, multi=1):
    assert multi >= 0
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.triple_list, triples2.triple_list, step, batch_size)
    neg_triples = list()
    if multi > 0:
        for i in range(multi):
            # choice = random.randint(0, 999)
            # if choice < 500:
            #     h = True
            # else:
            #     h = False
            # neg_triples.extend(generate_neg_triples_batch(pos_triples1, triples1, h))
            # neg_triples.extend(generate_neg_triples_batch(pos_triples2, triples2, h))
            neg_triples.extend(generate_neg_triples(pos_triples1, triples1))
            neg_triples.extend(generate_neg_triples(pos_triples2, triples2))
    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_batch_via_neighbour_no_pair(triples1, triples2, step, batch_size, nbours1, nbours2, multi=1):
    assert multi >= 1
    pos_triples1, pos_triples2 = generate_pos_batch(triples1.weighted_triples, triples2.weighted_triples, step,
                                                    batch_size)

    neg_triples = list()
    neg_triples.extend(trunc_sampling_multi(pos_triples1, triples1.triples, nbours1, triples1.ent_list, multi))
    neg_triples.extend(trunc_sampling_multi(pos_triples2, triples2.triples, nbours2, triples2.ent_list, multi))

    pos_triples1.extend(pos_triples2)
    return pos_triples1, neg_triples


def generate_batch_via_neighbour_no_pair_queue(que, triples1, triples2, steps, batch_size, nbours1, nbours2, multi):
    for step in steps:
        pos_triples1, neg_triples = generate_batch_via_neighbour_no_pair(triples1, triples2, step, batch_size, nbours1,
                                                                         nbours2, multi=multi)
        que.put((pos_triples1, neg_triples))
