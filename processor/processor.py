import logging
import os
import time
import torch
import torch.nn as nn
from utils.meter import AverageMeter
from utils.metrics import R1_mAP_eval, R1_mAP, R1_mAP_eval_allday
from torch.cuda import amp
import torch.distributed as dist

def count_parameters_detailed(model, logger):
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  
    
    logger.info(f"Trainable parameters:{trainable_params/1e6:.1f}M")
 

def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query, local_rank):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("ProxyTTT.train")
    logger.info('start training')
    _LOCAL_PROCESS_GROUP = None
    if device:
        model.to(local_rank)
        if torch.cuda.device_count() > 1 and cfg.MODEL.DIST_TRAIN:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank],
                                                              find_unused_parameters=True)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATASETS.NAMES == 'AllDay843&AllDay843-G':
        evaluator = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()
    
    scaler = amp.GradScaler()

    best_index = {'mAP': 0, "Rank-1": 0, 'Rank-5': 0, 'Rank-10': 0}
    
    # train
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        evaluator.reset()
        scheduler.step(epoch)
        count_parameters_detailed(model,logger)
        model.train()

        for n_iter, (img, vid, target_cam, target_view, _) in enumerate(train_loader):
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = {'RGB': img['RGB'].to(device),
                   'NI': img['NI'].to(device),
                   'TI': img['TI'].to(device)}
            target = vid.to(device)
            target_cam = target_cam.to(device)
            target_view = target_view.to(device)
            loss = 0
        

        
            with amp.autocast(enabled=True):
                score, feat, l = model(img, target, cam_label=target_cam, view_label=target_view, TTT = False)     
                loss += loss_fn(score, feat, target, target_cam)     
                loss += l
                

            scaler.scale(loss).backward()
         
            scaler.step(optimizer)
            scaler.update()

        
                 

            if 'center' in cfg.MODEL.METRIC_LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.SOLVER.CENTER_LOSS_WEIGHT)
                scaler.step(optimizer_center)
                scaler.update()
            if isinstance(score, list):
                acc = (score[0].max(1)[1] == target).float().mean()
            else:
                acc = (score.max(1)[1] == target).float().mean()

            loss_meter.update(loss.item(), img['RGB'].shape[0])
            acc_meter.update(acc, 1)

            torch.cuda.synchronize()
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, scheduler._get_lr(epoch)[0]))

        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)

        ttt_v = {'RGBNT100'}


        # TTT phrase
        if cfg.DATASETS.NAMES in ttt_v and epoch % cfg.TEST.TTT_epoch == 0:
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        
                img = {'RGB': img['RGB'].to(device),
                        'NI': img['NI'].to(device),
                        'TI': img['TI'].to(device)}
                camids = camids.to(device)
                scenceids = target_view
                target_view = target_view.to(device)
         
     
                with amp.autocast(enabled=True):
                    loss_ttt = model(img, cam_label=camids, view_label=target_view, TTT = True)
                  
    
                scaler.scale(loss_ttt).backward()
             
                scaler.step(optimizer)
                scaler.update()
    
           
     
                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("TTTEpoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(val_loader),
                                        loss_ttt.item(),  scheduler._get_lr(epoch)[0]))
    
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        elif cfg.DATASETS.NAMES not in ttt_v and epoch >= cfg.TEST.TTT_epoch:
            for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                        
                img = {'RGB': img['RGB'].to(device),
                        'NI': img['NI'].to(device),
                        'TI': img['TI'].to(device)}
                camids = camids.to(device)
                scenceids = target_view
                target_view = target_view.to(device)
         
     
                with amp.autocast(enabled=True):
                    loss_ttt = model(img, cam_label=camids, view_label=target_view, TTT = True)
                  
    
                scaler.scale(loss_ttt).backward()
             
                scaler.step(optimizer)
                scaler.update()
    
           
     
                torch.cuda.synchronize()
                if (n_iter + 1) % log_period == 0:
                    logger.info("TTT_Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Base Lr: {:.2e}"
                                .format(epoch, (n_iter + 1), len(val_loader),
                                        loss_ttt.item(),  scheduler._get_lr(epoch)[0]))
            if cfg.MODEL.DIST_TRAIN:
                pass
            else:
                logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                            .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
    
    
        
    
       
           
        if epoch % eval_period == 0:
            if epoch >= 1:
                model.eval()
                for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
                    with torch.no_grad():
                        img = {'RGB': img['RGB'].to(device),
                                'NI': img['NI'].to(device),
                                'TI': img['TI'].to(device)}
                        camids = camids.to(device)
                        scenceids = target_view
                        target_view = target_view.to(device)
                        feat = model(img, cam_label=camids, view_label=target_view)
                        if cfg.DATASETS.NAMES == "MSVR310":
                            evaluator.update((feat, vid, camid, scenceids, _))

                        elif cfg.DATASETS.NAMES == 'AllDay843&AllDay843-G':
                          
                            evaluator.update((feat, vid, camid, scenceids))
                        else:
                            evaluator.update((feat, vid, camid))
                cmc, mAP, _, _, _, _, _ = evaluator.compute()
                logger.info("Validation Results - Epoch: {}".format(epoch))
                logger.info("mAP: {:.1%}".format(mAP))
                for r in [1, 5, 10]:
                    logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

                if mAP >= best_index['mAP']:
                    if(mAP == best_index['mAP'] and cmc[0] >  best_index['Rank-1'] ):# 第一个等于 第二个超过了才更新
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'train_best.pth'))
                    else:
                        best_index['mAP'] = mAP
                        best_index['Rank-1'] = cmc[0]
                        best_index['Rank-5'] = cmc[4]
                        best_index['Rank-10'] = cmc[9]
                        torch.save(model.state_dict(),os.path.join(cfg.OUTPUT_DIR, cfg.MODEL.NAME + 'train_best.pth'))
                         
                logger.info("Best mAP: {:.1%}".format(best_index['mAP']))
                logger.info("Best Rank-1: {:.1%}".format(best_index['Rank-1']))
                logger.info("Best Rank-5: {:.1%}".format(best_index['Rank-5']))
                logger.info("Best Rank-10: {:.1%}".format(best_index['Rank-10']))
                torch.cuda.empty_cache()


def do_test(cfg,
                 model,
                 val_loader,
                 num_query,  
                 optimizer,
                 optimizer_center,
                 scheduler):
    device = "cuda"
    logger = logging.getLogger("transreid.test")
    logger.info("Enter test")

    if cfg.DATASETS.NAMES == "MSVR310":
        evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    elif cfg.DATASETS.NAMES == 'AllDay843&AllDay843-G':
        evaluator = R1_mAP_eval_allday(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    else:
        evaluator = R1_mAP_eval(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)
    evaluator.reset()

    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)


    model.eval()
    img_path_list = []

    for n_iter, (img, vid, camid, camids, target_view, _) in enumerate(val_loader):
        with torch.no_grad():
            img = {'RGB': img['RGB'].to(device),
                    'NI': img['NI'].to(device),
                    'TI': img['TI'].to(device)}
            camids = camids.to(device)
            scenceids = target_view
            target_view = target_view.to(device)
            feat = model(img, cam_label=camids, view_label=target_view)
            if cfg.DATASETS.NAMES == "MSVR310":
                evaluator.update((feat, vid, camid, scenceids, _))

            elif cfg.DATASETS.NAMES == 'AllDay843&AllDay843-G':
              
                evaluator.update((feat, vid, camid, scenceids))
            else:
                evaluator.update((feat, vid, camid))

    cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Validation Results ")
    logger.info("mAP: {:.1%}".format(mAP))
    for r in [1, 5, 10]:
        logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
    return cmc[0], cmc[4]