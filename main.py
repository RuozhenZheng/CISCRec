import torch.utils.data as data
import torch
import numpy as np
from tqdm import tqdm
import os
from model import CISCRec
import argparse
from sampler import Sampler

from utils.data import Data_property,Data_prop_topk
from utils.loader import build_candidates_set, ui_prop
from utils.metrics import precision_at_k, recall_at_k, map_at_k, hr_at_k, ndcg_at_k, mrr_at_k

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='./data/processdata_prop2.npy', help="filename of dataset (*.npy)")
    parser.add_argument('--batch_size', type=int, default=512, help="batch size of each interation (default:10000)")
    parser.add_argument('--latent_dimension', type=int, default=8, help="latent dimensionality K (default:10)")
    parser.add_argument('--learning_rate', type=float, default=0.01, help="learning rate (default:0.01)")
    parser.add_argument('--maximum_epochs', type=int, default=500,
                        help="maximum training epochs (default:2000)")  # 2000
    parser.add_argument('--alpha', type=float, default=1.0, help="coefficient on task T_I (default:1.0)")
    parser.add_argument('--beta', type=float, default=0.1, help="coefficient on task T_R (default:0.1)")
    parser.add_argument('--norm', type=float, default=1.0, help="maximum length of vectors (default:1.0)")
    parser.add_argument('--lambda_bias', type=float, default=1e-4,
                        help="coefficient lambda on bias terms (default:1e-4)")
    parser.add_argument('--gamma', type=float, default=0.5, help="proportion of long-term preferences (default:0.5)")
    #
    parser.add_argument('--num_prop', type=float, default=2, help="number of properties (default:1)")
    parser.add_argument('--candidates_num', type=int, default=500, help="number of candidates (default:50)")
    parser.add_argument('--topk', type=int, default=5, help="top k recommendation (default:5)")

    args = parser.parse_args()

    dataset = np.load(args.dataset, allow_pickle=True)

    # [user_train, user_validation, user_test, Item, user_num, item_num] = dataset
    [user_history,property_dict,user_train,user_test,Item,user_num,item_num,sub_num] = dataset

    dn = os.path.basename(args.dataset).rstrip('.npy')
    num_batch = 20
    f = open('./data/experiment/p1_c/%s_cn%d_topk%d_size%d_epoch%d_batch%d_dim%d_rate%g_bias%g_gamma%g.txt' % (
        dn, args.candidates_num, args.topk, args.batch_size, args.maximum_epochs, num_batch,
        args.latent_dimension, args.learning_rate,args.lambda_bias, args.gamma), 'w')

    # count postive events
    oneiteration = 0
    user_train_set = user_train
    for user in user_train:
        oneiteration += len(user_train[user])

    subreddit = set()
    for item in Item:
        subreddit.add(Item[item]['sub'])
    subreddits = list(subreddit)
    # num_rel = len(Relationships)
    num_rel = sub_num

    sampler = Sampler(user_train, user_train_set, user_history, Item, property_dict, user_num, item_num, subreddits,
                          batch_size=args.batch_size,
                          n_workers=1)
    test_sampler = Sampler(user_train, user_train_set, user_history, Item, property_dict, user_num, item_num, subreddits,
                          batch_size=args.batch_size,is_test=True, User_test=user_test,n_workers=1)


    model = CISCRec(
        user_num,
        item_num,
        sub_num,
        args,
        gpuid='0',
        early_stop=True
    )

    best_test_auc = 0.5
    best_iter = 0
    # num_batch = int(oneiteration / args.batch_size)

    n_batch = 10


    for i in range(args.maximum_epochs):
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            batch = sampler.next_batch()
            train_dataset = Data_property(batch)
            train_loader = data.DataLoader(
                train_dataset,
                batch_size=args.batch_size,
                shuffle=True,
                num_workers=0
            )

            model.fit(train_loader)

        if i % 50 == 0 and i != 0:
            item_pool = set(range(item_num))
            candidates_num = args.candidates_num
            test_ucands = build_candidates_set(user_test, user_train, item_pool, candidates_num)
            preds = {}
            topk = args.topk
            for u in tqdm(test_ucands.keys()):
                tmp_set = ui_prop(u, test_ucands, user_history, Item, property_dict)
                tmp_dataset = Data_prop_topk(tmp_set)
                tmp_loader = data.DataLoader(
                    tmp_dataset,
                    batch_size=candidates_num,
                    shuffle=False,
                    num_workers=0
                )

                for user, i_s, uj_s, j, user_prop in tmp_loader:
                    user = user.cuda()
                    i_s = i_s.cuda()
                    j = j.cuda()
                    uj_s = uj_s.cuda()
                    user_prop = user_prop.cuda()

                    prediction = model.forward_topk(user, i_s, uj_s, j, user_prop)

                    _, indices = torch.topk(prediction, topk)
                    # top_n = torch.take(torch.tensor(test_ucands[u]), indices).cpu().numpy()
                    top_n = torch.take(torch.tensor(test_ucands[u]), indices.cpu()).numpy()

                preds[u] = top_n

            for u in preds.keys():
                preds[u] = [1 if i in user_test[u] else 0 for i in preds[u]]
                #preds[u] = [1 if i == user_test[u].any() else 0 for i in preds[u]]

            tmp_preds = preds.copy()
            tmp_preds = {key: rank_list[:topk] for key, rank_list in tmp_preds.items()}

            pre_k = np.mean([precision_at_k(r, topk) for r in tmp_preds.values()])
            rec_k = recall_at_k(tmp_preds, user_test, topk)
            hr_k = hr_at_k(tmp_preds, user_test)
            map_k = map_at_k(tmp_preds.values())
            mrr_k = mrr_at_k(tmp_preds, topk)
            ndcg_k = np.mean([ndcg_at_k(r, topk) for r in tmp_preds.values()])

            f.write(f'iteration:{i}\n')
            f.write(f'Precision@{topk}: {pre_k:.4f}\n')
            f.write(f'Recall@{topk}: {rec_k:.4f}\n')
            f.write(f'HR@{topk}: {hr_k:.4f}\n')
            f.write(f'MAP@{topk}: {map_k:.4f}\n')
            f.write(f'MRR@{topk}: {mrr_k:.4f}\n')
            f.write(f'NDCG@{topk}: {ndcg_k:.4f}\n')


    sampler.close()
    test_sampler.close()

    f.write('Finished!best AUC: %f\n' % (best_test_auc))
    f.close()