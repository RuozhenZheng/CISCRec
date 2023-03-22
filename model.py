import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

class CISCRec(nn.Module):
    def __init__(self,
                 user_num,
                 item_num,
                 sub_num,
                 args,
                 gpuid='0',
                 early_stop=True):

        super(CISCRec, self).__init__()

        os.environ['CUDA_VISIBLE_DEVICES'] = gpuid
        cudnn.benchmark = True

        self.epochs = 1
        self.dim = args.latent_dimension
        self.batchsize = args.batch_size
        self.lr = args.learning_rate
        self.subnum = sub_num
        self.s_index = [i for i in range(sub_num)]
        self.num_prop = args.num_prop

        d_emb = args.latent_dimension

        self.embed_user = nn.Embedding(user_num, self.dim)
        self.embed_item = nn.Embedding(item_num, self.dim)
        self.embed_s = nn.Embedding(sub_num, self.dim)
        nn.init.normal_(self.embed_user.weight, std= 0.1)
        nn.init.normal_(self.embed_item.weight, std= 0.1)
        nn.init.normal_(self.embed_s.weight, std= 0.1)
        self.s_biases = nn.Embedding(1, sub_num)
        nn.init.constant_(self.s_biases.weight, 0)
        self.i_biases = nn.Embedding(item_num, 1)
        nn.init.constant_(self.i_biases.weight, 0)

        self.early_stop = early_stop
        self.prop_pref = nn.Embedding(user_num,self.num_prop)
        nn.init.normal_(self.prop_pref.weight, std= 0.1)

        self.gamma = args.gamma
        self.alpha = args.alpha
        self.beta = args.beta
        self.lambda_bias = args.lambda_bias

        self.conv = nn.Conv1d(sub_num, 2, args.latent_dimension - 3)  # dim=16 4*(3+1)
        self.act = nn.ReLU()


    def predict_function3(self,user,item_is, item_j,sub_weight,sub_emb,uj_s):

        user = torch.unsqueeze(user, 1)
        item_j = torch.unsqueeze(item_j, 1)

        uj_s_index = F.one_hot(uj_s, self.subnum)
        sub_weight = torch.sum(sub_weight * uj_s_index,dim=1)
        sub_weight = torch.unsqueeze(sub_weight,1)

        value1 = torch.sum(sub_weight * (-torch.sum(torch.square(user + sub_emb - item_j),dim=1)), dim=1)
        value = value1

        return value


    def decision_prop2(self,inp):
        [batch_is_emb,batch_u_emb,s_prop_emb] = inp
        real_bsize = batch_u_emb.shape[0]
        batch_u_emb = torch.unsqueeze(batch_u_emb, 1)
        batch_is_emb = torch.unsqueeze(batch_is_emb, 1)

        batch_s_embed = self.embed_s(torch.LongTensor(self.s_index).cuda()).repeat(real_bsize, 1, 1)
        batch_s_embed = batch_s_embed + s_prop_emb

        batch_s_bias = self.s_biases(torch.LongTensor([0]).cuda()).repeat(real_bsize, 1)

        pred = batch_s_bias - torch.sum(torch.square(batch_u_emb + batch_is_emb - batch_s_embed), 2)
        pred = F.softmax(pred)

        return pred

    def subemb_reinforce(self,user_prop,prop_pref):
        real_bsize = user_prop.shape[0]
        s_embed = self.embed_s(torch.LongTensor(self.s_index).cuda()).repeat(real_bsize, 1, 1) # ijk

        # i:batch; m:property num; j:subreddit num
        # k: subreddit embedding dimension
        prop_pref = F.softmax(prop_pref,dim=1)
        prop_scores = torch.einsum('imj,im->imj', user_prop, prop_pref)

        s_prop_emb = torch.einsum('ijk,imj->imjk', s_embed, prop_scores)

        # a = torch.einsum('imjk,ijk->ijk',s_prop_emb,s_embed)

        s_prop_emb = torch.mean(s_prop_emb, 1)

        conv_s = self.conv(s_prop_emb)
        act_s = self.act(conv_s)
        s_feat = torch.reshape(act_s, (act_s.size(0), -1)) # ik 每个user对应一个sub embedding, 可以搞到user embedding那里去

        return s_prop_emb, s_feat

    def useremb_reinforce(self,user, user_embed, s_feat):

        u_feat = user_embed + s_feat
        return u_feat

    def forward(self,user,i_s,uj_s,uj_sp,j,jp,user_prop):
        real_bsize = user.shape[0]
        user_embed = self.embed_user(user)
        #
        is_embed = self.embed_item(i_s)
        is_embed = torch.mean(is_embed,dim=1)
        j_embed = self.embed_item(j)
        jp_embed = self.embed_item(jp)
        uj_s_embed = self.embed_s(uj_s)
        uj_sp_embed = self.embed_s(uj_sp)
        prop_pref = self.prop_pref(user) # user preference to every property

        j_bias = torch.squeeze(self.i_biases(j), 1)
        jp_bias = torch.squeeze(self.i_biases(jp), 1)

        ######################## edit 20221002
        s_prop_emb, s_feat = self.subemb_reinforce(user_prop, prop_pref)

        subreddit_weight = self.decision_prop2([is_embed, user_embed, s_prop_emb])
        user_embed = self.useremb_reinforce(user, user_embed, s_feat) # new user embedding

        ########################
        updim_user_embed = torch.unsqueeze(user_embed,1)

        updim_uj_s_embed = torch.unsqueeze(uj_s_embed,1)
        updim_uj_sp_embed = torch.unsqueeze(uj_sp_embed,1)
        ##################################################

        updim_s_feat = torch.unsqueeze(s_feat,1)

        pos_dis = j_bias + self.predict_function3(user_embed, is_embed, j_embed, subreddit_weight,
                                                  torch.cat([updim_uj_s_embed, updim_user_embed, updim_s_feat], dim=1), uj_s)
        neg_dis = jp_bias + self.predict_function3(user_embed, is_embed, jp_embed, subreddit_weight,
                                                   torch.cat([updim_uj_sp_embed, updim_user_embed, updim_s_feat], dim=1), uj_sp)


        return pos_dis, neg_dis, j_bias, jp_bias
