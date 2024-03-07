import time
import datetime
import logging
import numpy as np
import torch
import torch.nn.functional as F
from lavis.common.dist_utils import is_main_process


def compute_sim_matrix(model, data_loader, **kwargs):
    if is_main_process():
        logging.info("Computing features for evaluation...")
        start_time = time.time()

        vis_embeds_proj_norm = []
        for samples in data_loader:
            vis_embed = samples["vis_embed"].to(model.device)
            vis_embed_proj_norm = F.normalize(model.proj_vis(vis_embed), dim=-1)
            vis_embeds_proj_norm.append(vis_embed_proj_norm)
        vis_embeds_proj_norm = torch.cat(vis_embeds_proj_norm, dim=0)

        text_feats = data_loader.dataset.text_feats_list
        num_text = len(text_feats)
        text_bs = 1024
        text_embeds_proj_norm = []
        for i in range(0, num_text, text_bs):
            text_embed = torch.stack(text_feats[i: min(num_text, i + text_bs)]).to(model.device)
            text_embed_proj_norm = F.normalize(model.proj_text(text_embed), dim=-1)
            text_embeds_proj_norm.append(text_embed_proj_norm)
        text_embeds_proj_norm = torch.cat(text_embeds_proj_norm, dim=0)

        sim_v2t = vis_embeds_proj_norm @ text_embeds_proj_norm.T
        sim_t2v = sim_v2t.T

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return sim_v2t.cpu().numpy(), sim_t2v.cpu().numpy()
    else:
        return None, None


def compute_ap(ranks, nres):
    """
    Computes average precision for given ranked indexes.
    Arguments
    ---------
    ranks : zerro-based ranks of positive images
    nres  : number of positive images
    Returns
    -------
    ap    : average precision
    """

    # number of images ranked by the system
    nimgranks = len(ranks)

    # accumulate trapezoids in PR-plot
    ap = 0

    recall_step = 1. / nres

    for j in np.arange(nimgranks):
        rank = ranks[j]

        if rank == 0:
            precision_0 = 1.
        else:
            precision_0 = float(j) / rank

        precision_1 = float(j + 1) / (rank + 1)

        ap += (precision_0 + precision_1) * recall_step / 2.

    return ap


def compute_map(ranks, gnd, kappas=[]):
    """
    Computes the mAP for a given set of returned results.
         Usage:
           map = compute_map (ranks, gnd)
                 computes mean average precsion (map) only
           map, aps, pr, prs = compute_map (ranks, gnd, kappas)
                 computes mean average precision (map), average precision (aps) for each query
                 computes mean precision at kappas (pr), precision at kappas (prs) for each query
         Notes:
         1) ranks starts from 0, ranks.shape = db_size X #queries
         2) The junk results (e.g., the query itself) should be declared in the gnd stuct array
         3) If there are no positive images for some query, that query is excluded from the evaluation
    """

    map = 0.
    nq = len(gnd)  # number of queries
    aps = np.zeros(nq)
    pr = np.zeros(len(kappas))
    prs = np.zeros((nq, len(kappas)))
    nempty = 0

    for i in np.arange(nq):
        qgnd = np.array(gnd[i]['ok'])

        # no positive images, skip from the average
        if qgnd.shape[0] == 0:
            aps[i] = float('nan')
            prs[i, :] = float('nan')
            nempty += 1
            continue

        try:
            qgndj = np.array(gnd[i]['junk'])
        except:
            qgndj = np.empty(0)

        # sorted positions of positive and junk images (0 based)
        pos = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgnd)]
        junk = np.arange(ranks.shape[0])[np.in1d(ranks[:, i], qgndj)]

        k = 0
        ij = 0
        if len(junk):
            # decrease positions of positives based on the number of
            # junk images appearing before them
            ip = 0
            while (ip < len(pos)):
                while (ij < len(junk) and pos[ip] > junk[ij]):
                    k += 1
                    ij += 1
                pos[ip] = pos[ip] - k
                ip += 1

        # compute ap
        ap = compute_ap(pos, len(qgnd))
        map = map + ap
        aps[i] = ap

        # compute precision @ k
        pos += 1  # get it to 1-based
        for j in np.arange(len(kappas)):
            kq = min(max(pos), kappas[j])
            prs[i, j] = (pos <= kq).sum() / kq
        pr = pr + prs[i, :]

    map = map / (nq - nempty)
    pr = pr / (nq - nempty)

    return map, aps, pr, prs


def compute_image_retrieval_metrics(model, data_loader):
    if is_main_process():
        logging.info("Computing features for image retrieval evaluation...")
        start_time = time.time()

        query_embeds_proj_norm = []
        for samples in data_loader:
            query_embed = samples["query_embed"].to(model.device)
            query_embed_proj_norm = F.normalize(model.proj_vis(query_embed), dim=-1)
            query_embeds_proj_norm.append(query_embed_proj_norm)
        query_embeds_proj_norm = torch.cat(query_embeds_proj_norm, dim=0)

        corpus_feats = data_loader.dataset.corpus_feats
        num_corpus = len(corpus_feats)
        corpus_bs = 1024
        corpus_embeds_proj_norm = []
        for i in range(0, num_corpus, corpus_bs):
            corpus_embed = corpus_feats[i: min(num_corpus, i + corpus_bs)].to(model.device)
            corpus_embed_proj_norm = F.normalize(model.proj_vis(corpus_embed), dim=-1)
            corpus_embeds_proj_norm.append(corpus_embed_proj_norm)
        corpus_embeds_proj_norm = torch.cat(corpus_embeds_proj_norm, dim=0)

        sim = corpus_embeds_proj_norm @ query_embeds_proj_norm.T
        ranks = torch.argsort(-sim, dim=0).cpu().numpy()

        gnd = data_loader.dataset.cfg["gnd"]
        dataset_name = data_loader.dataset.cfg["dataset_name"]

        # evaluate ranks
        ks = [1, 5, 10]
        # search for easy & hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['easy'], gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk']])
            gnd_t.append(g)
        mapM, apsM, mprM, prsM = compute_map(ranks, gnd_t, ks)

        # search for hard
        gnd_t = []
        for i in range(len(gnd)):
            g = {}
            g['ok'] = np.concatenate([gnd[i]['hard']])
            g['junk'] = np.concatenate([gnd[i]['junk'], gnd[i]['easy']])
            gnd_t.append(g)
        mapH, apsH, mprH, prsH = compute_map(ranks, gnd_t, ks)

        results = {
            f'{dataset_name}_M_mAP': np.around(mapM * 100, decimals=2),
            f'{dataset_name}_H_mAP': np.around(mapH * 100, decimals=2),
        }

        for idx, k in enumerate(ks):
            results[f'{dataset_name}_M_mP@{k}'] = np.around(mprM[idx] * 100, decimals=2)
            results[f'{dataset_name}_H_mP@{k}'] = np.around(mprH[idx] * 100, decimals=2)

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        logging.info("Evaluation time {}".format(total_time_str))

        return results
    else:
        return None


def slerp(lam, a, b):
    # assumes a and b are L2 normalized
    omega = torch.acos((a*b).sum(1))
    so = torch.sin(omega)
    res = (torch.sin((1.0-lam)*omega)/so).unsqueeze(1) * a + (torch.sin(lam*omega)/so).unsqueeze(1) * b
    return res
