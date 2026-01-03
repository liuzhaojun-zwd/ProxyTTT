import os
from config import cfg
import argparse
from datasets import make_dataloader
from model import make_model
from processor import do_test
from utils.logger import setup_logger
from loss import make_loss
from solver import make_optimizer
from solver.scheduler_factory import create_scheduler

if __name__ == "__main__":
    # 解析命令行参数, --config_file：指定配置文件的路径。
    parser = argparse.ArgumentParser(description="ReID Baseline Training")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    # opts：可以通过命令行进一步修改配置文件中的配置选项。
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()


    # 如果提供了配置文件路径，则从文件中加载配置  命令行参数会覆盖扩展cfg 内容
    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    # 创建输出目录并设置日志记录：
    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    logger = setup_logger("ProxyTTT", output_dir, if_train=False)
    logger.info(args)

    if args.config_file != "":
        logger.info("Loaded configuration file {}".format(args.config_file))
        with open(args.config_file, 'r') as cf:
            config_str = "\n" + cf.read()
            logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID

    # 创建数据加载器和模型实例：
    train_loader, train_loader_normal, val_loader, num_query, num_classes, camera_num, view_num = make_dataloader(cfg)

    loss_func, center_criterion = make_loss(cfg, num_classes=num_classes)
    model = make_model(cfg, num_class=num_classes, camera_num=camera_num, view_num = view_num)
    model.load_param(cfg.TEST.WEIGHT)
    
    optimizer, optimizer_center = make_optimizer(cfg, model, center_criterion)



    scheduler = create_scheduler(cfg, optimizer)

  
    

   
    do_test(cfg,
             model,
             val_loader,
             num_query, 
             optimizer,
             optimizer_center,
             scheduler)

