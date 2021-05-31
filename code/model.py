import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

class EDUA(torch.nn.Module):
    def __init__(self, user_num, item_num, cate_num, dim_num, margin, num_mem, max_len_item, max_len_user):
        super(EDUA, self).__init__()
        self.num_users = user_num
        self.num_items = item_num
        self.num_cates = cate_num
        self.margin = margin
        self.num_mem = num_mem
        self.max_len_item = max_len_item
        self.max_len_user = max_len_user
        self.dim_num = dim_num
        self.dropout = 0.0

        def init_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

        self.embedding_user_mlp1 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_num)
        self.embedding_item_mlp1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_num)
        nn.init.xavier_uniform_(self.embedding_user_mlp1.weight)
        nn.init.xavier_uniform_(self.embedding_item_mlp1.weight)
        self.embedding_user_mf1 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_num)
        self.embedding_item_mf1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_num)
        nn.init.xavier_uniform_(self.embedding_user_mf1.weight)
        nn.init.xavier_uniform_(self.embedding_item_mf1.weight)

        self.embedding_user_mlp2 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_num)
        self.embedding_item_mlp2 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_num)
        nn.init.xavier_uniform_(self.embedding_user_mlp2.weight)
        nn.init.xavier_uniform_(self.embedding_item_mlp2.weight)
        self.embedding_user_mf2 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.dim_num)
        self.embedding_item_mf2 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.dim_num)
        nn.init.xavier_uniform_(self.embedding_user_mf2.weight)
        nn.init.xavier_uniform_(self.embedding_item_mf2.weight)

        self.embedding_cate = torch.nn.Embedding(num_embeddings=self.num_cates, embedding_dim=self.dim_num * self.num_mem)
        nn.init.xavier_uniform_(self.embedding_cate.weight)

        #---------------------------------------------
        self.dense_attention = torch.nn.Linear(dim_num, dim_num, bias=False)
        init_weights(self.dense_attention)
        self.dense_attention2 = torch.nn.Linear(dim_num, dim_num, bias=False)
        init_weights(self.dense_attention2)

        self.dense_attention_user = torch.nn.Linear(dim_num, dim_num, bias=False)
        init_weights(self.dense_attention_user)
        self.dense_attention_user2 = torch.nn.Linear(dim_num, dim_num, bias=False)
        init_weights(self.dense_attention_user2)

        #------------------------------------------
        self.user_aspect_miu = torch.nn.Parameter(torch.randn(self.num_mem, dim_num), requires_grad=True)
        nn.init.xavier_uniform_(self.user_aspect_miu.data)
        self.user_aspect_sigma = torch.nn.Parameter(torch.randn(self.num_mem, dim_num), requires_grad=True)
        nn.init.xavier_uniform_(self.user_aspect_sigma.data)
        self.item_aspect_miu = torch.nn.Parameter(torch.randn(self.num_mem, dim_num), requires_grad=True)
        nn.init.xavier_uniform_(self.item_aspect_miu.data)
        self.item_aspect_sigma = torch.nn.Parameter(torch.randn(self.num_mem, dim_num), requires_grad=True)
        nn.init.xavier_uniform_(self.item_aspect_sigma.data)

        self.sigmoid = torch.nn.Sigmoid()
        self.tanh = torch.nn.Tanh()
        self.relu = torch.nn.ReLU()


    def func_cal_relation(self, user_emb, item_emb, user_i_emb, item_u_emb, user_i_list, item_u_list, dense_i, dense_u):
        # u-i --> relation
        u_item_joint = torch.bmm(dense_i(user_i_emb), item_emb.unsqueeze(-1))
        u_item_joint = u_item_joint.masked_fill(user_i_list.unsqueeze(-1).eq(0), -np.inf)
        u_item_joint = F.softmax(u_item_joint, dim=1)
        u_item_read = (user_i_emb * u_item_joint).sum(dim=1)

        i_user_joint = torch.bmm(dense_u(item_u_emb), user_emb.unsqueeze(-1))
        i_user_joint = i_user_joint.masked_fill(item_u_list.unsqueeze(-1).eq(0), -np.inf)
        i_user_joint = F.softmax(i_user_joint, dim=1)
        i_user_read = (item_u_emb * i_user_joint).sum(dim=1)

        return u_item_read, i_user_read

    def forward(self, user_indices, item_indices, item_negs, rev_item_indices, rev_item_negs, u_item_list, i_user_list, neg_user_list, rev_i_user_list, rev_neg_user_list, l_alpha, user_aspect, item_aspect, neg_item_aspect, reverse_item_aspect, reverse_neg_aspect, user_alpha_origin):
        """
        :param user_indices: batch user ID
        :param item_indices: batch positive-item ID
        :param item_negs:    batch negative-item ID
        :param rev_item_indices:    batch reversed-sampler positive-item ID
        :param rev_item_negs:       batch reversed-sampler negative-item ID
        :param u_item_list:  batch user-interactions (item)-ID
        :param i_user_list:
        :param neg_user_list:
        :param rev_i_user_list:
        :param rev_neg_user_list:
        :param l_alpha:         T/T_{max}
        :param user_aspect:     user-aspect-distrbution
        :param item_aspect:
        :param neg_item_aspect:
        :param reverse_item_aspect:
        :param reverse_neg_aspect:
        :param user_alpha_origin:   diversity-score
        :return:
        """

        batch_size = user_indices.shape[0]
        neg_num = item_negs.shape[1]
        #-------------------------------------------------

        user_embedding_mlp = self.embedding_user_mlp1(user_indices)
        item_embedding_mlp = self.embedding_item_mlp1(item_indices)
        negs_embedding = self.embedding_item_mlp1(item_negs)

        rev_user_embedding_mlp = self.embedding_user_mlp2(user_indices)
        rev_item_embedding_mlp = self.embedding_item_mlp2(rev_item_indices)
        rev_negs_embedding_mlp = self.embedding_item_mlp2(rev_item_negs)

        user_i_embedding = self.embedding_item_mf1(u_item_list)  #[batch, len, emb]
        item_u_embedding = self.embedding_user_mf1(i_user_list)
        rev_user_i_embedding = self.embedding_item_mf2(u_item_list)
        rev_item_u_embedding = self.embedding_user_mf2(rev_i_user_list)

        # aspect relation ----------
        user_aspect_tmp_miu = (user_aspect.unsqueeze(-1) * self.user_aspect_miu.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        user_aspect_tmp_sigma = (user_aspect.unsqueeze(-1) * self.user_aspect_sigma.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        user_aspect_emb = user_aspect_tmp_miu + 0.01*torch.exp(user_aspect_tmp_sigma / 2) * torch.randn(batch_size, self.dim_num).cuda()

        item_aspect_tmp_miu = (item_aspect.unsqueeze(-1) * self.item_aspect_miu.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        item_aspect_tmp_sigma = (item_aspect.unsqueeze(-1) * self.item_aspect_sigma.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        item_aspect_emb = item_aspect_tmp_miu + 0.01*torch.exp(item_aspect_tmp_sigma / 2) * torch.randn(batch_size, self.dim_num).cuda()

        neg_item_aspect_tmp_miu = (neg_item_aspect.unsqueeze(-1) * self.item_aspect_miu.unsqueeze(0).unsqueeze(0).repeat(batch_size, neg_num, 1, 1)).mean(dim=2)
        neg_item_aspect_tmp_sigma = (neg_item_aspect.unsqueeze(-1) * self.item_aspect_sigma.unsqueeze(0).unsqueeze(0).repeat(batch_size, neg_num, 1, 1)).mean(dim=2)
        neg_item_aspect_emb = neg_item_aspect_tmp_miu + 0.01*torch.exp(neg_item_aspect_tmp_sigma / 2) * torch.randn(batch_size, neg_num, self.dim_num).cuda()

        reverse_item_aspect_tmp_miu = (reverse_item_aspect.unsqueeze(-1) * self.item_aspect_miu.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        reverse_item_aspect_tmp_sigma = (reverse_item_aspect.unsqueeze(-1) * self.item_aspect_sigma.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        reverse_item_aspect_emb = reverse_item_aspect_tmp_miu + 0.01*torch.exp(reverse_item_aspect_tmp_sigma / 2) * torch.randn(batch_size, self.dim_num).cuda()

        reverse_neg_aspect_tmp_miu = (reverse_neg_aspect.unsqueeze(-1) * self.item_aspect_miu.unsqueeze(0).unsqueeze(0).repeat(batch_size, neg_num, 1, 1)).mean(dim=2)
        reverse_neg_aspect_tmp_sigma = (reverse_neg_aspect.unsqueeze(-1) * self.item_aspect_sigma.unsqueeze(0).unsqueeze(0).repeat(batch_size, neg_num, 1, 1)).mean(dim=2)
        reverse_neg_aspect_emb = reverse_neg_aspect_tmp_miu + 0.01*torch.exp(reverse_neg_aspect_tmp_sigma / 2) * torch.randn(batch_size, neg_num, self.dim_num).cuda()

        # list relation-------------

        user_i_relation, item_u_relation = self.func_cal_relation(user_embedding_mlp, item_embedding_mlp, user_i_embedding, item_u_embedding, u_item_list, i_user_list, self.dense_attention, self.dense_attention_user) #pos, uniform: r1, r2
        user_rev_i_relation, rev_item_u_relation = self.func_cal_relation(rev_user_embedding_mlp, rev_item_embedding_mlp, rev_user_i_embedding, rev_item_u_embedding, u_item_list, rev_i_user_list, self.dense_attention2, self.dense_attention_user2) #pos, reversed: r1, r2

        #u-neg -->relation
        user_neg_relation_all, neg_user_relation_all = [], []   #neg, uniform: r1, r2
        for n in range(neg_num):
            neg_u_list = neg_user_list[n]
            negs_u_embedding = self.embedding_user_mf1(neg_u_list)  # [batch, neg, len, emb]
            curr_ui_relation, curr_iu_relation = self.func_cal_relation(user_embedding_mlp, negs_embedding[:, n, :], user_i_embedding, negs_u_embedding, u_item_list, neg_u_list, self.dense_attention, self.dense_attention_user)

            user_neg_relation_all.append(curr_ui_relation.unsqueeze(1))
            neg_user_relation_all.append(curr_iu_relation.unsqueeze(1))
        user_neg_relation_all = torch.cat(user_neg_relation_all, dim=1)
        neg_user_relation_all = torch.cat(neg_user_relation_all, dim=1)

        user_rev_neg_relation_all, rev_neg_user_relation_all = [], []  # neg, reversed: r1, r2
        for n in range(neg_num):
            rev_neg_u_list = rev_neg_user_list[n]
            rev_negs_u_embedding = self.embedding_user_mf2(rev_neg_u_list)  # [batch, neg, len, emb]
            curr_ui_relation, curr_iu_relation = self.func_cal_relation(rev_user_embedding_mlp, rev_negs_embedding_mlp[:, n, :], rev_user_i_embedding, rev_negs_u_embedding, u_item_list, rev_neg_u_list, self.dense_attention2, self.dense_attention_user2)

            user_rev_neg_relation_all.append(curr_ui_relation.unsqueeze(1))
            rev_neg_user_relation_all.append(curr_iu_relation.unsqueeze(1))
        user_rev_neg_relation_all = torch.cat(user_rev_neg_relation_all, dim=1)
        rev_neg_user_relation_all = torch.cat(rev_neg_user_relation_all, dim=1)

        #2 relation ------------------------
        user_i_relation = user_i_relation + user_aspect_emb
        item_u_relation = item_u_relation + item_aspect_emb
        user_rev_i_relation = user_rev_i_relation + user_aspect_emb
        rev_item_u_relation = rev_item_u_relation + reverse_item_aspect_emb
        user_neg_relation_all = user_neg_relation_all + user_aspect_emb.unsqueeze(1)
        neg_user_relation_all = neg_user_relation_all + neg_item_aspect_emb
        user_rev_neg_relation_all = user_rev_neg_relation_all + user_aspect_emb.unsqueeze(1)
        rev_neg_user_relation_all = rev_neg_user_relation_all + reverse_neg_aspect_emb

        # trans uniform ---------------------
        pos_distances = torch.sum(torch.pow(user_embedding_mlp + user_i_relation - item_embedding_mlp, 2), dim=-1)
        neg_distances = torch.sum(torch.pow(user_embedding_mlp.unsqueeze(1).repeat(1, neg_num, 1) + user_neg_relation_all - negs_embedding, 2), dim=-1)
        closest_neg_distances, closest_idx = neg_distances.min(dim=1)

        pos_distances_2 = torch.sum(torch.pow(item_embedding_mlp + item_u_relation - user_embedding_mlp, 2), dim=-1)
        neg_distances_2 = torch.sum(torch.pow(negs_embedding + neg_user_relation_all - user_embedding_mlp.unsqueeze(1).repeat(1, neg_num, 1), 2), dim=-1)
        closest_neg_distances_2, closest_idx_2 = neg_distances_2.min(dim=1)

        loss_per_pair = self.relu(pos_distances - closest_neg_distances + self.margin)
        loss_per_pair_2 = self.relu(pos_distances_2 - closest_neg_distances_2 + self.margin)

        #trans reversed
        rev_pos_distances = torch.sum(torch.pow(rev_user_embedding_mlp + user_rev_i_relation - rev_item_embedding_mlp, 2), dim=-1)
        rev_neg_distances = torch.sum(torch.pow(rev_user_embedding_mlp.unsqueeze(1).repeat(1, neg_num, 1) + user_rev_neg_relation_all - rev_negs_embedding_mlp, 2), dim=-1)
        closest_rev_neg_distances, rev_closest_idx = rev_neg_distances.min(dim=1)

        rev_pos_distances_2 = torch.sum(torch.pow(rev_item_embedding_mlp + rev_item_u_relation - rev_user_embedding_mlp, 2), dim=-1)
        rev_neg_distances_2 = torch.sum(torch.pow(rev_negs_embedding_mlp + rev_neg_user_relation_all - rev_user_embedding_mlp.unsqueeze(1).repeat(1, neg_num, 1), 2), dim=-1)
        closest_rev_neg_distances_2, rev_closest_idx_2 = rev_neg_distances_2.min(dim=1)

        loss_per_pair_rev = self.relu(rev_pos_distances - closest_rev_neg_distances + self.margin)
        loss_per_pair_2_rev = self.relu(rev_pos_distances_2 - closest_rev_neg_distances_2 + self.margin)


        #loss (bi-trans + kl) --------------------------------
        def kl_categorical(p_logit, q_logit):
            p = F.softmax(p_logit, dim=-1)
            _kl = torch.sum(p * (F.log_softmax(p_logit, dim=-1)
                                 - F.log_softmax(q_logit, dim=-1)), 1)
            return torch.mean(_kl)
            #return torch.sum(_kl)
        kl_1 = kl_categorical(user_embedding_mlp + user_i_relation - item_embedding_mlp, rev_user_embedding_mlp + user_rev_i_relation - rev_item_embedding_mlp)
        kl_2 = kl_categorical(item_embedding_mlp + item_u_relation - user_embedding_mlp, rev_item_embedding_mlp + rev_item_u_relation - rev_user_embedding_mlp)

        loss = ((l_alpha * user_alpha_origin) * (loss_per_pair + loss_per_pair_2)).mean(0) + ((1 - user_alpha_origin * l_alpha) * (loss_per_pair_rev + loss_per_pair_2_rev)).mean(0) + (kl_1 + kl_2)

        return loss


    def predict(self, user_indices, item_indices, u_item_list, i_user_list, l_alpha, user_aspect, item_aspect):
        batch_size = item_indices.shape[0]
        
        user_embedding_mlp = self.embedding_user_mlp1(user_indices)
        item_embedding_mlp = self.embedding_item_mlp1(item_indices)
        rev_user_embedding_mlp = self.embedding_user_mlp2(user_indices)
        item_embedding_mlp2 = self.embedding_item_mlp2(item_indices)

        user_i_embedding = self.embedding_item_mf1(u_item_list)  # [batch, len, emb]
        item_u_embedding = self.embedding_user_mf1(i_user_list)
        user_i_embedding2 = self.embedding_item_mf2(u_item_list)  # [batch, len, emb]
        item_u_embedding2 = self.embedding_user_mf2(i_user_list)

        user_aspect_tmp_miu = (user_aspect.unsqueeze(-1) * self.user_aspect_miu.unsqueeze(0)).mean(dim=1)
        user_aspect_tmp_sigma = (user_aspect.unsqueeze(-1) * self.user_aspect_sigma.unsqueeze(0)).mean(dim=1)
        user_aspect_emb = user_aspect_tmp_miu + 0.01*torch.exp(user_aspect_tmp_sigma / 2) * torch.randn(1, self.dim_num).cuda()
        user_aspect_emb = user_aspect_emb.repeat(batch_size, 1)

        item_aspect_tmp_miu = (item_aspect.unsqueeze(-1) * self.item_aspect_miu.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        item_aspect_tmp_sigma = (item_aspect.unsqueeze(-1) * self.item_aspect_sigma.unsqueeze(0).repeat(batch_size, 1, 1)).mean(dim=1)
        item_aspect_emb = item_aspect_tmp_miu + 0.01*torch.exp(item_aspect_tmp_sigma / 2) * torch.randn(batch_size,self.dim_num).cuda()

        user_i_relation, item_u_relation = self.func_cal_relation(user_embedding_mlp.repeat(item_indices.shape[0], 1), item_embedding_mlp, user_i_embedding.repeat(item_indices.shape[0], 1, 1), item_u_embedding, u_item_list, i_user_list, self.dense_attention, self.dense_attention_user)
        user_i_relation2, item_u_relation2 = self.func_cal_relation(rev_user_embedding_mlp.repeat(item_indices.shape[0], 1),  item_embedding_mlp2, user_i_embedding2.repeat(item_indices.shape[0], 1, 1), item_u_embedding2, u_item_list, i_user_list, self.dense_attention2, self.dense_attention_user2)

        user_i_relation = user_i_relation + user_aspect_emb
        user_i_relation2 = user_i_relation2 + user_aspect_emb
        item_u_relation = item_u_relation + item_aspect_emb
        item_u_relation2 = item_u_relation2 + item_aspect_emb


        pos_distances = l_alpha * (-torch.sum(torch.pow(user_embedding_mlp + user_i_relation - item_embedding_mlp, 2), dim=-1) - torch.sum(torch.pow(item_embedding_mlp + item_u_relation - user_embedding_mlp, 2), dim=-1)) \
                        + (1 - l_alpha) * (-torch.sum(torch.pow(rev_user_embedding_mlp + user_i_relation2 - item_embedding_mlp2, 2), dim=-1) - torch.sum(torch.pow(item_embedding_mlp2 + item_u_relation2 - rev_user_embedding_mlp, 2), dim=-1))

        return pos_distances