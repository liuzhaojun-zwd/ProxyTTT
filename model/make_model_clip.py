

import torch.nn.functional as F
import torch
import torch.nn as nn
import numpy as np
from .clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import copy
from .MPL import MPLModule
import numpy as np
import re



class ProxyOrthogonalDisentangler(nn.Module):
    def __init__(self, dim, ortho_weight=1.0):
        super().__init__()
        self.dim = dim
        self.ortho_weight = ortho_weight
        # 可学习的正交变换矩阵 Q
        self.q_transform = nn.Linear(dim, dim, bias=False)
        self._init_orthogonal_weights()

    def _init_orthogonal_weights(self):
        # 初始化为近似正交矩阵
        nn.init.orthogonal_(self.q_transform.weight)

    def forward(self, proxy_r, proxy_n, proxy_t):
        """
        proxy_*: [C, D]，每个类别的代理表示
        """

        q_r = self.q_transform(proxy_r)   # [C, D]
        q_n = self.q_transform(proxy_n)

        sim_rn = torch.mean(torch.abs(torch.sum(F.normalize(q_r, dim=-1) * F.normalize(q_n, dim=-1), dim=-1)))
        sim_rt = torch.mean(torch.abs(torch.sum(F.normalize(q_r, dim=-1) * F.normalize(q_t, dim=-1), dim=-1)))
        sim_nt = torch.mean(torch.abs(torch.sum(F.normalize(q_n, dim=-1) * F.normalize(q_t, dim=-1), dim=-1)))
        decouple_loss = (sim_rn + sim_rt + sim_nt)

        q_mat = self.q_transform.weight  # [D, D]
        q_norm = torch.matmul(q_mat, q_mat.T)
        identity = torch.eye(self.dim, device=q_mat.device)
        ortho_reg = F.mse_loss(q_norm, identity)

        return self.ortho_weight * (decouple_loss + ortho_reg)


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)




class build_transformer(nn.Module):
    def __init__(self, num_classes, camera_num, view_num, cfg):
        super(build_transformer, self).__init__()
        self.model_name = cfg.MODEL.NAME
        self.ttt_pth = cfg.TEST.WEIGHT
 
        self.neck = cfg.MODEL.NECK
        self.neck_feat = cfg.TEST.NECK_FEAT
        self.in_planes = 768

        self.miss = cfg.TEST.MISS

        self.cfg = cfg
        self.ol = ProxyOrthogonalDisentangler(dim=768)
    

    
        if self.model_name == 'ViT-B-16':
            self.in_planes = 768
            self.in_planes_proj = 512
        elif self.model_name == 'RN50':
            self.in_planes = 2048
            self.in_planes_proj = 1024
        self.num_classes = num_classes
        self.camera_num = camera_num
        self.view_num = view_num


        
     
        self.classifier = nn.Linear(self.in_planes*3, self.num_classes, bias=False)  # 768*3 Class_num
        self.classifier.apply(weights_init_classifier)
        self.proj_dim = 768
        self.proxy = MPLModule(self.proj_dim, cfg.MODEL.proxy_num, use_frl=True, margin=0.3)  #多模态模态代理
        self.rgb_proxy = MPLModule(768, 1, use_frl=True, margin=0.3)  #rgb模态代理
        self.ni_proxy = MPLModule(768, 1, use_frl=True, margin=0.3)  #nir模态代理
        self.ti_proxy = MPLModule(768, 1, use_frl=True, margin=0.3)  #thermal模态代理
        
   

        self.proj = nn.Linear(768*3, self.proj_dim)

       

        self.fc_r = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.fc_r.apply(weights_init_classifier)

     

        
   

        self.fc_n = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.fc_n.apply(weights_init_classifier)

        self.fc_t = nn.Linear(self.in_planes, self.num_classes, bias=False)
        self.fc_t.apply(weights_init_classifier)

        self.bn_r = nn.BatchNorm1d(self.in_planes)
        self.bn_r.bias.requires_grad_(False)
        self.bn_r.apply(weights_init_kaiming)

        self.bn_n = nn.BatchNorm1d(self.in_planes)
        self.bn_n.bias.requires_grad_(False)
        self.bn_n.apply(weights_init_kaiming)

        self.bn_t = nn.BatchNorm1d(self.in_planes)
        self.bn_t.bias.requires_grad_(False)
        self.bn_t.apply(weights_init_kaiming)
        
        

        self.bottleneck = nn.BatchNorm1d(self.in_planes*3)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)


      
       

        self.h_resolution = int((cfg.INPUT.SIZE_TRAIN[0]-16)//cfg.MODEL.STRIDE_SIZE[0] + 1)
        self.w_resolution = int((cfg.INPUT.SIZE_TRAIN[1]-16)//cfg.MODEL.STRIDE_SIZE[1] + 1)
        self.vision_stride_size = cfg.MODEL.STRIDE_SIZE[0]
        clip_model = load_clip_to_cpu(self.model_name, self.h_resolution, self.w_resolution, self.vision_stride_size)
        clip_model.to("cuda")

        self.image_encoder = clip_model.visual
    
     
    def generate_pseudo_labels_classifier(self, ori_f):
        """
        使用分类器输出生成伪标签
        Args:
            ori_f: 多模态拼接特征，形状为 [B, 768*3]
        Returns:
            pseudo_labels: 分类器预测的伪标签 [B]
        """
        ori_f_bn = self.bottleneck(ori_f)
    
        # 获取分类器输出
        logits = self.classifier(ori_f_bn)  # [B, num_classes]
        probs = F.softmax(logits, dim=1)

        _, pseudo_labels = torch.max(probs, dim=1)    
    
        return pseudo_labels
        

    def forward(self, x, label=None, cam_label= None, view_label=None, TTT = False, misalign=True) :
            layer_r, r_x12, _ = self.image_encoder(x['RGB'], None) 
            layer_n, n_x12, _ = self.image_encoder(x['NI'], None) 
            layer_t, t_x12, _ = self.image_encoder(x['TI'], None) 

            ori_f = torch.cat([r_x12[:,0], n_x12[:,0], t_x12[:,0]], dim=-1)
   
            rgb, ni, ti = [r_x12[:,0], n_x12[:,0], t_x12 [:,0]]
            


            ori_proj = self.proj(ori_f)

            if self.training and TTT == False: 
                out = self.proxy(ori_proj, label)
                loss_mcp = out["total_loss"] 
                rgb_out = self.rgb_proxy(rgb, label)
                loss_r= rgb_out["total_loss"] 
                proxy_r = rgb_out['proxy_feat']
                ni_out = self.ni_proxy(ni, label)
                loss_n = ni_out["total_loss"] 
                proxy_n = ni_out['proxy_feat']
                ti_out = self.ti_proxy(ti, label)
                loss_t = ti_out["total_loss"] 
                proxy_t = ti_out['proxy_feat']
                loss_msp = loss_r +  loss_n + loss_t + self.ol(proxy_r,proxy_n,proxy_t)
                loss = loss_mcp + loss_msp
        
                ori_id = self.classifier(self.bottleneck(ori_f) )

                rgb_id = self.fc_r(self.bn_r(rgb))
                ni_id = self.fc_n(self.bn_n(ni))
                ti_id = self.fc_t(self.bn_t(ti))
                if self.cfg.MODEL.DIRECT:
                    return [ori_id], [ori_f], loss
                else:
                    return [ori_id,rgb_id, ni_id, ti_id], [ori_f,rgb,ni,ti], loss
            if self.training and TTT:
                proxy_r = self.rgb_proxy.get_proxies()  # [training_class, 768]
                proxy_n = self.ni_proxy.get_proxies()   # [training_class, 768]
                proxy_t = self.ti_proxy.get_proxies()   # [training_class, 768]       

                if self.cfg.TEST.MISS == 'R':
                    ori_f =  torch.cat([torch.zeros_like(rgb), ni, ti], dim=-1)
                    proxy_r = torch.zeros_like(proxy_r)
                if self.cfg.TEST.MISS == 'N':
                    ori_f = torch.cat([rgb,torch.zeros_like(ni), ti], dim=-1)
                    proxy_n = torch.zeros_like(proxy_n)
                if self.cfg.TEST.MISS == 'T':
                    ori_f =  torch.cat([rgb, ni, torch.zeros_like(ti)], dim=-1)
                    proxy_t = torch.zeros_like(proxy_t)
                if self.cfg.TEST.MISS == 'RN':
                    ori_f =  torch.cat([torch.zeros_like(rgb), torch.zeros_like(ni), ti], dim=-1)
                    proxy_r = torch.zeros_like(proxy_r)
                    proxy_n = torch.zeros_like(proxy_n)
                if self.cfg.TEST.MISS == 'RT':
                    ori_f =  torch.cat([torch.zeros_like(rgb), ni, torch.zeros_like(ti)], dim=-1)
                    proxy_r = torch.zeros_like(proxy_r)
                    proxy_t = torch.zeros_like(proxy_t)
                if self.cfg.TEST.MISS == 'NT':
                    ori_f =  torch.cat([rgb, torch.zeros_like(ni), torch.zeros_like(ti)], dim=-1)
                    proxy_n = torch.zeros_like(proxy_n)
                    proxy_t = torch.zeros_like(proxy_t)
                
    
                ttt_loss = self.forward_with_PESA(
                    ori_f, proxy_r, proxy_n, proxy_t,entropy=self.cfg.MODEL.entropy)
                return ttt_loss*self.cfg.MODEL.ttt_weight
            if self.training == False and TTT == False:
                if self.cfg.TEST.MISS == 'R':
                    return  torch.cat([ni, ti], dim=-1)
                if self.cfg.TEST.MISS == 'N':
                    return  torch.cat([rgb, ti], dim=-1)
                if self.cfg.TEST.MISS == 'T':
                    return  torch.cat([rgb, ni], dim=-1)
                if self.cfg.TEST.MISS == 'RN':
                    return  ti
                if self.cfg.TEST.MISS == 'RT':
                    return  ni
                if self.cfg.TEST.MISS == 'NT':
                    return  rgb
                return ori_f

    def forward_with_PESA(self, ori_f, proxy_r, proxy_n, proxy_t, 
                            strategy='entropy_based', entropy=0.45):  
        pseudo_labels, confident_mask, max_sims, similarities = PESA(
            ori_f, proxy_r, proxy_n, proxy_t, strategy=strategy, entropy=entropy
        )
        ttt_loss_fn = AdaptiveTTTLoss(
            ce_weight=1.0, 
            info_weight=1.0
        )
        ttt_loss = ttt_loss_fn(ori_f, pseudo_labels, similarities, confident_mask) 
        return ttt_loss


   
    def load_param(self, trained_path):
      
        param_dict = torch.load(trained_path, map_location='cpu')
        
  
        if 'module.' in next(iter(param_dict.keys())):
            param_dict = {k.replace('module.', ''): v for k, v in param_dict.items()}

        proxy_prefixes = {
            'proxy': 'proxy.proxies_',
            'rgb_proxy': 'rgb_proxy.proxies_',
            'ni_proxy': 'ni_proxy.proxies_',
            'ti_proxy': 'ti_proxy.proxies_'
        }
        all_proxy_prefixes = list(proxy_prefixes.values())
        
        submodule_class_ids = {submodule: set() for submodule in proxy_prefixes.keys()}
        for key in param_dict.keys():
            for submodule, prefix in proxy_prefixes.items():
                if key.startswith(prefix):
                    match = re.match(f'{prefix}(\d+)', key)
                    if match:
                        class_id = int(match.group(1))
                        submodule_class_ids[submodule].add(class_id)
                    break
        
        for submodule_name in proxy_prefixes.keys():
            if not hasattr(self, submodule_name):
                continue
            submodule = getattr(self, submodule_name)
            if not hasattr(submodule, '_create_class_entry'):
                continue
            device = next(submodule.parameters()).device if list(submodule.parameters()) else torch.device('cpu')
            for class_id in submodule_class_ids[submodule_name]:
                if class_id not in submodule.member_bank:
                    submodule._create_class_entry(class_id, device=device)
        
        jpl_params = {k: v for k, v in param_dict.items() if any(k.startswith(p) for p in all_proxy_prefixes)}
        other_params = {k: v for k, v in param_dict.items() if not any(k.startswith(p) for p in all_proxy_prefixes)}
        
        temp_state_dict = {k: v for k, v in other_params.items() if k in self.state_dict()}
      
        missing, unexpected = self.load_state_dict(temp_state_dict, strict=False)

        for submodule_name in proxy_prefixes.keys():
            if not hasattr(self, submodule_name):
                continue
            submodule = getattr(self, submodule_name)
            prefix = proxy_prefixes[submodule_name]
            submodule_params = {k.replace(prefix, ''): v for k, v in jpl_params.items() if k.startswith(prefix)}
            if submodule_params:
                submodule.load_state_dict(submodule_params, strict=False)
        
    def load_param_simple(self, trained_path):
        param_dict = torch.load(trained_path, map_location='cpu')
        
        # Handle DataParallel wrapper if present
        if 'module.' in list(param_dict.keys())[0]:
            param_dict = {k.replace('module.', ''): v for k, v in param_dict.items()}
        
        # Load with strict=False to ignore missing/extra keys
        missing_keys, unexpected_keys = self.load_state_dict(param_dict, strict=False)
        
      
        print(f"Successfully loaded model from {trained_path}")
    
    def load_param_finetune(self, model_path):
        param_dict = torch.load(model_path)
        for i in param_dict:
            self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model for finetuning from {}'.format(model_path))


def make_model(cfg, num_class, camera_num, view_num):
    model = build_transformer(num_class, camera_num, view_num, cfg)
    return model

from .clip import clip
def load_clip_to_cpu(backbone_name, h_resolution, w_resolution, vision_stride_size):
    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict(), h_resolution, w_resolution, vision_stride_size)

    return model



class AdaptivePseudoLabelGenerator:
    def __init__(self, entropy=0.45):
     
        self.entropy = entropy
        
    def generate_labels(self, ori_f, proxy_r, proxy_n, proxy_t):
        return self._entropy_based_selection(ori_f, proxy_r, proxy_n, proxy_t)
    

    def _compute_similarities(self, ori_f, proxy_r, proxy_n, proxy_t):
        #  [training_class, 768*3]
        proxy = torch.cat([proxy_r, proxy_n, proxy_t], dim=-1)
        
   
        ori_f_norm = F.normalize(ori_f, p=2, dim=1)
        proxy_norm = F.normalize(proxy, p=2, dim=1)
        
   
        similarities = torch.mm(ori_f_norm, proxy_norm.t())
        max_similarities, pseudo_labels = torch.max(similarities, dim=1)
        
        return similarities, max_similarities, pseudo_labels

    def _entropy_based_selection(self, ori_f, proxy_r, proxy_n, proxy_t):
        similarities, max_sims, pseudo_labels = self._compute_similarities(
            ori_f, proxy_r, proxy_n, proxy_t)
        
        probs = F.softmax(similarities / 0.1, dim=1)  #
        
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
   
        entropy_threshold = torch.quantile(entropy, self.entropy)  
        confident_mask = entropy < entropy_threshold
        
 
        return pseudo_labels, confident_mask, max_sims, similarities

   



def PESA(ori_f, proxy_r, proxy_n, proxy_t, 
                                  strategy='entropy_based', entropy=0.45):
    generator = AdaptivePseudoLabelGenerator(strategy, entropy)
    return generator.generate_labels(ori_f, proxy_r, proxy_n, proxy_t)


class AdaptiveTTTLoss(nn.Module):
    def __init__(self, margin=0.3, ce_weight=1.0, info_weight=1.0, 
                 temperature=0.1):
        super().__init__()
        self.margin = margin
        self.ce_weight = ce_weight
        self.info_weight = info_weight
        self.temperature = temperature
        self.ce_loss = nn.CrossEntropyLoss(reduction='mean')
        
    def forward(self, features, pseudo_labels, similarities, confident_mask):

        if confident_mask.sum() == 0:
            prob = F.softmax(similarities / self.temperature, dim=1)
            entropy_loss = -torch.sum(prob * torch.log(prob + 1e-8), dim=1).mean()
            return 0.1 * entropy_loss
        
        confident_features = features[confident_mask]
        confident_labels = pseudo_labels[confident_mask]
        confident_sims = similarities[confident_mask]
        
        total_loss = 0.0
        

        if self.ce_weight > 0:
            ce_loss = self.ce_loss(confident_sims, confident_labels)
            total_loss += self.ce_weight * ce_loss

        if self.info_weight > 0:
            contrastive_loss = self.compute_contrastive_loss(
                confident_features, confident_labels)
            total_loss += self.info_weight * contrastive_loss
    
        return total_loss
    
    def compute_contrastive_loss(self, features, labels):
        if len(labels.unique()) < 2:
            return torch.tensor(0.0, device=features.device)
        
        features_norm = F.normalize(features, p=2, dim=1)
        sim_matrix = torch.mm(features_norm, features_norm.t())
        labels_expanded = labels.unsqueeze(1)
        pos_mask = (labels_expanded == labels_expanded.t()).float()
        neg_mask = 1 - pos_mask
        

        pos_loss = -torch.log(torch.exp(sim_matrix) / (torch.exp(sim_matrix).sum(dim=1, keepdim=True) + 1e-8))
        pos_loss = (pos_loss * pos_mask).sum() / (pos_mask.sum() + 1e-8)
        
        return pos_loss
    
  