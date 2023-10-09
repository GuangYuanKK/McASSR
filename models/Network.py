import torch
import torch.nn as nn
import torch.nn.functional as F

import models
from models import register
from models.rcan import  local_enhanced_blcok
from utils import make_coord

import numpy as np


@register('mc_arsr')
class MC_ARSR(nn.Module):

    def __init__(self, encoder_spec, imnet_spec=None, hidden_dim=256):
        super().__init__()
        self.encoder = models.make(encoder_spec)
        self.coef = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.freq = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.ref_conv = nn.Conv2d(self.encoder.out_dim, hidden_dim, 3, padding=1)
        self.phase = nn.Linear(2, hidden_dim // 2, bias=False)
        self.local_enhanced_block = local_enhanced_blcok()
        self.imnet = models.make(imnet_spec, args={'in_dim': hidden_dim})

    def gen_feat(self, inp, ref, ref_hr):
        self.inp = inp
        self.ref = ref
        self.ref_hr = ref_hr

        self.inp_feat_coord = make_coord(inp.shape[-2:], flatten=False) \
            .permute(2, 0, 1) \
            .unsqueeze(0).expand(inp.shape[0], 2, *inp.shape[-2:])

        self.inp_feat = self.encoder(inp, ref)
        self.ref_feat_hr, self.ref_loss =  self.local_enhanced_block(ref_hr)

        self.inp_coeff = self.coef(self.inp_feat)
        self.inp_freqq = self.freq(self.inp_feat)
        self.ref_feat_hr_res = self.ref_conv(self.ref_feat_hr)

        return self.inp_feat, self.ref_feat_hr_res, self.ref_loss

    def query_rgb(self, inp_hr_coord, inp_cell):
        ###### Tar ######
        inp_feat = self.inp_feat
        inp_coef = self.inp_coeff
        inp_freq = self.inp_freqq
        ###### Ref ######
        ref_feat_hr_res = self.ref_feat_hr_res

        vx_lst = [-1, 1]
        vy_lst = [-1, 1]
        eps_shift = 1e-6

        # field radius (global: [-1, 1])
        rx = 2 / inp_feat.shape[-2] / 2
        ry = 2 / inp_feat.shape[-1] / 2

        inp_feat_coord = self.inp_feat_coord

        preds = []
        areas = []

        for vx in vx_lst:
            for vy in vy_lst:
                inp_out, bs, q, rel_coord = self.RIA(inp_feat, inp_cell, inp_hr_coord, inp_feat_coord, inp_coef, inp_freq, ref_feat_hr_res, vx, rx, vy, ry, eps_shift)
                merge = inp_out

                pred = self.imnet(merge.contiguous().view(bs * q, -1)).view(bs, q, -1)
                preds.append(pred)

                area = torch.abs(rel_coord[:, :, 0] * rel_coord[:, :, 1])
                areas.append(area + 1e-9)

        tot_area = torch.stack(areas).sum(dim=0)
        t = areas[0]
        areas[0] = areas[3]
        areas[3] = t
        t = areas[1]
        areas[1] = areas[2]
        areas[2] = t

        ret = 0
        for pred, area in zip(preds, areas):
            ret = ret + pred * (area / tot_area).unsqueeze(-1)
        ret += F.grid_sample(self.inp, inp_hr_coord.flip(-1).unsqueeze(1), mode='bilinear',padding_mode='border', align_corners=False)[:, :, 0, :].permute(0, 2, 1)

        return ret

    def RIA(self, feat, cell, coord, inp_feat_coord, inp_coef, inp_freq, ref_feat_hr_res, vx, rx, vy, ry, eps_shift):
        # Reference-Aware Implicit Attention

        coord_ = coord.clone()
        coord_[:, :, 0] += vx * rx + eps_shift
        coord_[:, :, 1] += vy * ry + eps_shift
        coord_.clamp_(-1 + 1e-6, 1 - 1e-6)

        V_ref_coef = F.grid_sample(
            inp_coef, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) # V
        K_ref_freq = F.grid_sample(
            inp_freq, coord_.flip(-1).unsqueeze(1),
            mode='nearest', align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) # K
        Q_tar_coord = F.grid_sample(
            inp_feat_coord, coord_.flip(-1).unsqueeze(1),
            align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) # Q

        ref_hr_res = F.grid_sample(
            ref_feat_hr_res, coord_.flip(-1).unsqueeze(1),
            align_corners=False)[:, :, 0, :] \
            .permute(0, 2, 1) # R

        Q_rel_coord = coord - Q_tar_coord
        Q_rel_coord[:, :, 0] *= feat.shape[-2]
        Q_rel_coord[:, :, 1] *= feat.shape[-1]

        # prepare cell
        rel_cell = cell.clone()
        rel_cell[:, :, 0] *= feat.shape[-2]
        rel_cell[:, :, 1] *= feat.shape[-1]

        # basis generation
        bs, q = coord.shape[:2]
        K_q_freq = torch.stack(torch.split(K_ref_freq, 2, dim=-1), dim=-1)
        K_q_freq = torch.mul(K_q_freq, Q_rel_coord.unsqueeze(-1))
        K_q_freq = torch.sum(K_q_freq, dim=-2)
        K_q_freq += self.phase(rel_cell.view((bs * q, -1))).view(bs, q, -1)
        K_q_freq = torch.cat((torch.sin(np.pi * K_q_freq), torch.sin(np.pi * K_q_freq)), dim=-1)

        out = torch.mul(V_ref_coef, K_q_freq) + ref_hr_res

        return out, bs, q, Q_rel_coord

    def forward(self, inp, inp_hr_coord, inp_cell, ref, ref_hr):
        self.gen_feat(inp, ref, ref_hr)
        return self.query_rgb(inp_hr_coord, inp_cell), self.ref_loss
