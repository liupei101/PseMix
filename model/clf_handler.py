import os
import os.path as osp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.dataloader import default_collate
from types import SimpleNamespace
import wandb

from .model_utils import load_model, general_init_weight
from loss.utils import load_loss, loss_reg_l1
from eval.utils import load_evaluator
from dataset.utils import prepare_clf_dataset
from optim import create_optimizer

from utils.func import seed_everything, parse_str_dims, print_metrics
from utils.func import add_prefix_to_filename, rename_keys
from utils.func import fetch_kws, print_config, EarlyStopping
from utils.func import seed_generator, seed_worker
from utils.io import read_datasplit_npz, read_maxt_from_table
from utils.io import save_prediction_clf, save_prediction_mixclf
from utils.core import PseudoBag, augment_bag, remix_bag, generate_pseudo_bags
from utils.core import PseudoBag_Kmeans, PseudoBag_Random, mixup_bag


class ClfHandler(object):
    """
    Handling the initialization, training, and testing 
    of general MIL-based classification models.
    """
    def __init__(self, cfg):
        # check args
        assert cfg['task'] == 'clf', 'Task must be clf.'

        torch.cuda.set_device(cfg['cuda_id'])
        seed_everything(cfg['seed'])

        # path setup
        if cfg['test']:
            if cfg['test_mask_ratio'] is None:
                cfg['test_save_path'] = cfg['test_save_path'].format(cfg['data_split_seed'])
            else:
                cfg['test_save_path'] = cfg['test_save_path'].format(cfg['test_mask_ratio'], cfg['data_split_seed'])
            cfg['test_load_path'] = cfg['test_load_path'].format(cfg['data_split_seed'])
            if not osp.exists(cfg['test_save_path']):
                os.makedirs(cfg['test_save_path'])
            run_name = cfg['test_save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['test_load_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['test_load_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['test_save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['test_save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['test_save_path'], 'print_config.txt')
            # wandb writter
            self.writer = wandb.init(project=cfg['test_wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)
        else:
            if not osp.exists(cfg['save_path']):
                os.makedirs(cfg['save_path'])
            run_name = cfg['save_path'].split('/')[-1]
            self.last_ckpt_path = osp.join(cfg['save_path'], 'model-last.pth')
            self.best_ckpt_path = osp.join(cfg['save_path'], 'model-best.pth')
            self.last_metrics_path = osp.join(cfg['save_path'], 'metrics-last.txt')
            self.best_metrics_path = osp.join(cfg['save_path'], 'metrics-best.txt')
            self.config_path = osp.join(cfg['save_path'], 'print_config.txt')
            # wandb writter
            self.writer = wandb.init(project=cfg['wandb_prj'], name=run_name, dir=cfg['wandb_dir'], config=cfg, reinit=True)

        # model setup
        dims = parse_str_dims(cfg['net_dims'])
        self.net = load_model(
            cfg['task'], cfg['backbone'], dims, 
            drop_rate=cfg['drop_rate'], use_feat_proj=cfg['use_feat_proj']
        ).cuda()
        if cfg['init_wt']:
            self.net.apply(general_init_weight)

        # loss setup
        cfg['loss_active_mixup'] = False if cfg['mixup_type'] not in ['psebmix', 'insmix'] else True
        kws_loss = fetch_kws(cfg, prefix='loss')
        self.loss = load_loss(cfg['task'], **kws_loss)
        if kws_loss['bce']:
            assert dims[-1] == 2, "conflit between the configs 'bce_loss' and 'net_dims'."
        else:
            assert dims[-1] > 2, "conflit between the configs 'bce_loss' and 'net_dims'."
        
        # optimizer and lr_scheduler
        cfg_optimizer = SimpleNamespace(opt=cfg['opt_name'], weight_decay=cfg['opt_weight_decay'], lr=cfg['opt_lr'], 
            opt_eps=None, opt_betas=None, momentum=None)
        self.optimizer = create_optimizer(cfg_optimizer, self.net)
        # LR scheduler
        kws_lrs = fetch_kws(cfg, prefix='lrs')
        self.steplr = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', 
            factor=kws_lrs['factor'], patience=kws_lrs['patience'], verbose=True)

        # evaluator
        self.evaluator = load_evaluator(cfg['task'], binary_clf=cfg['loss_bce'])
        if cfg['loss_bce']: # binary
            self.metrics_list = ['auc', 'loss', 'acc', 'acc@mid', 'acc_best', 'recall', 'precision', 'f1_score', 'ece', 'mce']
        else: # multi-class
            self.metrics_list = ['auc', 'loss', 'acc', 'macro_f1_score', 'micro_f1_score']
        self.ret_metrics = ['auc', 'loss']

        self.task = cfg['task']
        self.bin_clf = cfg['loss_bce']
        self.backbone = cfg['backbone']
        self.uid = dict()
        self.pseb_ind = dict()
        self.instance_score = dict()
        self.cfg = cfg
        print_config(cfg, print_to_path=self.config_path)

    def exec(self):
        print('[exec] setting: task = {}, backbone = {}.'.format(self.task, self.backbone))
        
        # Prepare data spliting 
        if "{}" in self.cfg['data_split_path']:
            if 'data_split_fold' in self.cfg:
                path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'], self.cfg['data_split_fold'])
            else:
                path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        else:
            path_split = self.cfg['data_split_path']
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        print('[exec] finished reading patient IDs from {}'.format(path_split))

        # Prepare datasets 
        train_set  = prepare_clf_dataset(pids_train, self.cfg, ratio_sampling=self.cfg['data_sampling_ratio'])
        self.uid.update({'train': train_set.uid})
        if 'data_corrupt_label' in self.cfg and self.cfg['data_corrupt_label'] is not None:
            assert self.cfg['data_corrupt_label'] > 1e-7 and self.cfg['data_corrupt_label'] <= 1.0
            train_set.corrupt_labels(self.cfg['data_corrupt_label'])
        val_set    = prepare_clf_dataset(pids_val, self.cfg)
        self.uid.update({'validation': val_set.uid})
        train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=True,  worker_init_fn=seed_worker, collate_fn=default_collate
        )
        val_loader = DataLoader(val_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
        )
        if pids_test is not None:
            test_set    = prepare_clf_dataset(pids_test, self.cfg)
            self.uid.update({'test': test_set.uid})
            test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
                generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                shuffle=False, worker_init_fn=seed_worker, collate_fn=default_collate
            )
        else:
            test_set    = None 
            test_loader = None

        run_name = 'train'
        # Train
        if 'force_to_skip_training' in self.cfg and self.cfg['force_to_skip_training']:
            print("[warning] your training is skipped...")
        else:
            val_name = 'validation'
            val_loaders = {'validation': val_loader, 'test': test_loader}
            if 'eval_training_loader_per_epoch' in self.cfg and self.cfg['eval_training_loader_per_epoch']:
                train_loader_for_eval = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
                    generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
                    shuffle=False,  worker_init_fn=seed_worker, collate_fn=default_collate
                )
                val_loaders['eval-train'] = train_loader_for_eval
            self._run_training(self.cfg['epochs'], train_loader, 'train', val_loaders=val_loaders, val_name=val_name, 
                measure_training_set=True, save_ckpt=True, early_stop=True, run_name=run_name)

        # Evals using the best ckpt
        train_set.resume_labels()
        train_loader = DataLoader(train_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=False,  worker_init_fn=seed_worker, collate_fn=default_collate
        )
        evals_loader = {'train': train_loader, 'validation': val_loader, 'test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', run_name=run_name, if_print=True)
        
        return metrics

    def exec_test(self):
        print('[exec] test under task = {}, backbone = {}.'.format(self.task, self.backbone))
        mode_name = 'test_mode'
        
        # Prepare datasets 
        path_split = self.cfg['data_split_path'].format(self.cfg['data_split_seed'])
        pids_train, pids_val, pids_test = read_datasplit_npz(path_split)
        if self.cfg['test_path'] == 'train':
            pids = pids_train
        elif self.cfg['test_path'] == 'val':
            pids = pids_val
        elif self.cfg['test_path'] == 'test':
            pids = pids_test
        else:
            pass
        print('[exec] test patient IDs from {}'.format(self.cfg['test_path']))

        # Prepare datasets 
        test_set = prepare_clf_dataset(pids, self.cfg, ratio_mask=self.cfg['test_mask_ratio'])
        self.uid.update({'exec-test': test_set.uid})

        if self.cfg['test_in_between']:
            test_in_between_data = True
            shuffle = True
            if_print = False
        else:
            test_in_between_data = False
            shuffle = False
            if_print = True

        test_loader = DataLoader(test_set, batch_size=self.cfg['batch_size'], 
            generator=seed_generator(self.cfg['seed']), num_workers=self.cfg['num_workers'], 
            shuffle=shuffle, worker_init_fn=seed_worker, collate_fn=default_collate
        )

        # Evals
        evals_loader = {'exec-test': test_loader}
        metrics = self._eval_all(evals_loader, ckpt_type='best', if_print=if_print, test_mode=True, 
            test_mode_name=mode_name, test_in_between_data=test_in_between_data)
        return metrics

    def _run_training(self, epochs, train_loader, name_loader, val_loaders=None, val_name=None, 
        measure_training_set=True, save_ckpt=True, early_stop=False, run_name='train', **kws):
        """Traing model.

        Args:
            epochs (int): Epochs to run.
            train_loader ('DataLoader'): DatasetLoader of training set.
            name_loader (string): name of train_loader, used for infering patient IDs.
            val_loaders (dict): A dict like {'val': loader1, 'test': loader2}, which gives the datasets
                to evaluate at each epoch.
            val_name (string): The dataset used to perform early stopping and optimal model saving.
            measure_training_set (bool): If measure training set at each epoch.
            save_ckpt (bool): If save models.
            early_stop (bool): If early stopping according to validation loss.
            run_name (string): Name of this training, which would be used as the prefixed name of ckpt files.
        """
        # setup early_stopping
        if early_stop and self.cfg['es_patience'] is not None:
            self.early_stop = EarlyStopping(warmup=self.cfg['es_warmup'], patience=self.cfg['es_patience'], 
                start_epoch=self.cfg['es_start_epoch'], verbose=self.cfg['es_verbose'])
        else:
            self.early_stop = None

        if val_name is not None and self.early_stop is not None:
            assert val_name in val_loaders.keys(), "Not specify a dataloader to enable early stopping."
            print("[{}] {} epochs w early stopping on {}.".format(run_name, epochs, val_name))
        else:
            print("[{}] {} epochs w/o early stopping.".format(run_name, epochs))
        
        # iterative training
        last_epoch = -1
        for epoch in range(epochs):
            last_epoch = epoch + 1
            train_cltor = self._train_each_epoch(train_loader, name_loader)
            cur_name = name_loader

            if measure_training_set:
                for k_cltor, v_cltor in train_cltor.items():
                    self._eval_and_print(v_cltor, name=cur_name+'/'+k_cltor, at_epoch=epoch+1)

            # val/test
            early_stopping_metrics = None
            if val_loaders is not None:
                for k in val_loaders.keys():
                    if val_loaders[k] is None:
                        continue
                    val_cltor = self.test_model(self.net, val_loaders[k], loader_name=k)
                    for k_cltor, v_cltor in val_cltor.items():
                        met_auc, met_loss = self._eval_and_print(v_cltor, name=k+'/'+k_cltor, at_epoch=epoch+1)
                        if k == val_name and k_cltor == 'pred':
                            early_stopping_metrics = met_auc if self.cfg['monitor_metrics'] == 'auc' else met_loss
            
            # early_stop using VAL_METRICS
            if early_stopping_metrics is not None and self.early_stop is not None:
                self.steplr.step(early_stopping_metrics)
                self.early_stop(epoch, early_stopping_metrics)
                if self.early_stop.save_ckpt():
                    self.save_model(epoch+1, ckpt_type='best', run_name=run_name)
                    print("[train] {} best model saved at epoch {}".format(run_name, epoch+1))
                if self.early_stop.stop():
                    break
        
        if save_ckpt:
            self.save_model(last_epoch, ckpt_type='last', run_name=run_name) # save models and optimizers
            print("[train] {} last model saved at epoch {}".format(run_name, last_epoch))

    def _train_each_epoch(self, train_loader, name_loader):
        print("[train] train one epoch using train_loader={}".format(name_loader))
        self.net.train()
        bp_every_batch = self.cfg['bp_every_batch']
        all_pred, all_gt = [], []

        idx_collector, x_collector, ext_x_collector, y_collector = [], [], [], []
        i_batch = 0
        for data_idx, data_x, data_y in train_loader:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            i_batch += 1
            # 1. read data (mini-batch)
            data_input = data_x[0] # only use the first item
            data_label = data_y[0]

            data_input = data_input.cuda()
            data_label = data_label.cuda()

            x_collector.append(data_input)
            y_collector.append(data_label)
            idx_collector.append(data_idx)
            # For ReMix, data_x[1] = Tensor of [n_batch, n_cluster, n_shift_vector, dim_feat]
            ext_x_collector.append(data_x[1].squeeze(0).numpy()) 

            # in a mini-batch
            if i_batch % bp_every_batch == 0:
                # 2. data augmentation
                if self.cfg['mixup_type'] == 'psebmix' or self.cfg['mixup_type'] == 'pseudo-bag':
                    pseb_ind_collector = self._collect_pseb_ind(name_loader, idx_collector, x_collector)
                else:
                    pseb_ind_collector = None

                if self.cfg['mixup_type'] == 'remix':
                    # It follows ReMix's implementation
                    x_mixed_collector, y_a_collector = remix_bag(
                        x_collector, y_collector,
                        mode='joint',
                        semantic_shifts=ext_x_collector,
                    )
                    y_b_collector, lam = y_a_collector, 1.
                elif self.cfg['mixup_type'] == 'mixup':
                    # It follows Mixup. Instance number aligned by random cropping the larger
                    x_mixed_collector, y_a_collector, y_b_collector, lam, _ = mixup_bag(
                        x_collector, y_collector, alpha=self.cfg['mixup_alpha'],
                    )
                elif self.cfg['mixup_type'] == 'rankmix':
                    # It follows RankMix. Instance number aligned by ranking instances and then cropping the larger
                    if len(self.instance_score) <= 0:
                        score_collector = None
                    else:
                        score_collector = [self.instance_score[_idx.item()] for _idx in idx_collector]
                    x_mixed_collector, y_a_collector, y_b_collector, lam, _ = mixup_bag(
                        x_collector, y_collector, scores=score_collector, alpha=self.cfg['mixup_alpha'],
                    )
                elif self.cfg['mixup_type'] == 'pseudo-bag':
                    # follows ProtoDiv's implementation
                    x_mixed_collector = generate_pseudo_bags(
                        x_collector,
                        n_pseb=self.cfg['pseb_n'],
                        ind_pseb=pseb_ind_collector,
                    )
                    lam = 1.
                    y_a_collector = y_collector
                    y_b_collector = y_collector
                else:
                    x_mixed_collector, y_a_collector, y_b_collector, lam, _ = augment_bag(
                        x_collector, y_collector,
                        alpha=self.cfg['mixup_alpha'],
                        method=self.cfg['mixup_type'],
                        mixup_lam_from=self.cfg['mixup_lam_from'],
                        psebmix_n=self.cfg['pseb_n'],
                        psebmix_ind=pseb_ind_collector, 
                        psebmix_prob=self.cfg['pseb_mixup_prob']
                    )

                # 3. update network
                cur_pred = self._update_network(i_batch, x_mixed_collector, y_a_collector, y_b_collector, lam)
                all_pred.append(cur_pred)
                all_gt.append(torch.cat(y_collector, dim=0).detach().cpu())

                # 4. reset mini-batch
                idx_collector, x_collector, ext_x_collector, y_collector = [], [], [], []
                torch.cuda.empty_cache()

        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0).squeeze(1) # [B, ]

        train_cltor = dict()
        train_cltor['pred'] = {'y': all_gt, 'y_hat': all_pred}

        return train_cltor

    def _update_network(self, i_batch, xs, ys_a, ys_b, lam):
        """
        Update network using one batch data
        """
        n_sample = len(xs)
        y_hat = []

        for i in range(n_sample):
            # [B, num_cls], [B, N]
            logit_bag = self.net(xs[i])
            y_hat.append(logit_bag)

        # 3.1 zero gradients buffer
        self.optimizer.zero_grad()

        # 3.2 loss
        # loss of bag clf
        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
        bag_label_a = torch.cat(ys_a, dim=0).squeeze(-1) # [B, ]
        bag_label_b = torch.cat(ys_b, dim=0).squeeze(-1) # [B, ]
        # prepare lam
        if isinstance(lam, list):
            assert len(lam) == len(bag_preds)
            lam = torch.FloatTensor(lam * bag_preds.shape[1]).view(bag_preds.shape[1], -1).permute(1, 0).cuda()
        else:
            lam = torch.full_like(bag_preds, lam)
        clf_loss = self.loss(bag_preds, bag_label_a, lam) + self.loss(bag_preds, bag_label_b, 1. - lam)
        print("[training one epoch] {}-th batch: clf_loss = {:.6f}".format(i_batch, clf_loss.item()))
        wandb.log({'train_batch/clf_loss': clf_loss.item()})

        # 3.3 backward gradients and update networks
        clf_loss.backward()
        self.optimizer.step()

        val_preds = bag_preds.detach().cpu()
        return val_preds

    def _eval_all(self, evals_loader, ckpt_type='best', run_name='train', task='bag_clf', if_print=True,
        test_mode=False, test_mode_name='test_mode', test_in_between_data=False):
        """
        test_mode = True only if run self.exec_test(), indicating a test mode.
        test_in_between_data: if testing in between data.
        """
        if test_mode:
            print('[warning] you are in test mode now.')
            ckpt_run_name = 'train'
            wandb_group_name = test_mode_name
            metrics_path_name = test_mode_name
            csv_prefix_name = test_mode_name
            save_pred_path = self.cfg['test_save_path']
        else:
            ckpt_run_name = run_name
            wandb_group_name = run_name
            metrics_path_name = run_name
            csv_prefix_name = run_name
            save_pred_path = self.cfg['save_path']
        
        if ckpt_type == 'best':
            ckpt_path = add_prefix_to_filename(self.best_ckpt_path, ckpt_run_name)
            wandb_group = 'bestckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.best_metrics_path, metrics_path_name)
            csv_name = '{}_{}_best'.format(task, csv_prefix_name)
        elif ckpt_type == 'last':
            ckpt_path = add_prefix_to_filename(self.last_ckpt_path, ckpt_run_name)
            wandb_group = 'lastckpt/{}'.format(wandb_group_name)
            print_path = add_prefix_to_filename(self.last_metrics_path, metrics_path_name)
            csv_name = '{}_{}_last'.format(task, csv_prefix_name)
        else:
            pass

        metrics = dict()
        for k, loader in evals_loader.items():
            if loader is None:
                continue
            if test_in_between_data:
                print('[info] testing in-between {} data...'.format(k))
                res_mix = self.test_model_with_in_between_data(self.net, loader, loader_name=k, ckpt_path=ckpt_path)
                if self.cfg['save_prediction']:
                    path_save_pred = osp.join(save_pred_path, '{}_MixBagClf_pred_{}.csv'.format(csv_name, k))
                    res_mix['mix_idx_a'] = self._get_unique_id('exec-test', res_mix['mix_idx_a'])
                    res_mix['mix_idx_b'] = self._get_unique_id('exec-test', res_mix['mix_idx_b'])
                    save_prediction_mixclf(res_mix, path_save_pred)
            else:
                cltor = self.test_model(self.net, loader, loader_name=k, ckpt_path=ckpt_path)
                metrics[k] = []
                for k_cltor, v_cltor in cltor.items():
                    auc, loss = self._eval_and_print(v_cltor, name='{}/{}/{}'.format(wandb_group, k, k_cltor))
                    metrics[k].append(('auc_'+k_cltor, auc))
                    metrics[k].append(('loss_'+k_cltor, loss))

                used_cltor = cltor['pred']
                if self.cfg['save_prediction']:
                    path_save_pred = osp.join(save_pred_path, '{}_BagClf_pred_{}.csv'.format(csv_name, k))
                    uids = self._get_unique_id(k, used_cltor['idx'])
                    save_prediction_clf(uids, used_cltor['y'], used_cltor['y_hat'], path_save_pred, binary=self.bin_clf)

        if if_print:
            print_metrics(metrics, print_to_path=print_path)

        return metrics

    def _eval_and_print(self, cltor, name='', ret_metrics=None, at_epoch=None):
        if ret_metrics is None:
            ret_metrics = self.ret_metrics
        eval_metrics = self.metrics_list
        eval_results = self.evaluator.compute(cltor, eval_metrics)
        eval_results = rename_keys(eval_results, name, sep='/')

        print("[{}] At epoch {}:".format(name, at_epoch), end=' ')
        print(' '.join(['{}={:.6f},'.format(k, v) for k, v in eval_results.items()]))
        wandb.log(eval_results)

        return [eval_results[name+'/'+k] for k in ret_metrics]

    def _get_unique_id(self, k, idxs, concat=None):
        if k not in self.uid:
            raise KeyError('Key {} not found in `uid`'.format(k))
        uids = self.uid[k]
        idxs = idxs.squeeze().tolist()
        if concat is None:
            return [uids[i] for i in idxs]
        else:
            return [uids[v] + "-" + str(concat[i].item()) for i, v in enumerate(idxs)]

    def _collect_pseb_ind(self, k, idxs, Xs):
        cur_pseb_ind = []
        if self.cfg['pseb_gene_once']:
            if k not in self.uid:
                raise KeyError('Key {} not found in `uid`'.format(k))
            for i, batch_id in enumerate(idxs):
                uid = self.uid[k][batch_id]
                if uid not in self.pseb_ind:
                    if 'pseb_dividing' not in self.cfg or self.cfg['pseb_dividing'] == 'proto':
                        bag = PseudoBag(self.cfg['pseb_n'], self.cfg['pseb_l'], self.cfg['pseb_proto'], 
                            self.cfg['pseb_pheno_cut'], self.cfg['pseb_iter_tuning'])
                    elif self.cfg['pseb_dividing'] == 'kmeans':
                        bag = PseudoBag_Kmeans(self.cfg['pseb_n'], self.cfg['pseb_l'])
                    elif self.cfg['pseb_dividing'] == 'random':
                        bag = PseudoBag_Random(self.cfg['pseb_n'])
                    else:
                        pass
                    self.pseb_ind[uid] = bag.divide(Xs[i])
                cur_pseb_ind.append(self.pseb_ind[uid])
        else:
            for i, batch_id in enumerate(idxs):
                bag = PseudoBag(self.cfg['pseb_n'], self.cfg['pseb_l'], self.cfg['pseb_proto'], 
                    self.cfg['pseb_pheno_cut'], self.cfg['pseb_iter_tuning'])
                pseb = bag.divide(Xs[i])
                cur_pseb_ind.append(pseb)

        return cur_pseb_ind

    def test_model(self, model, loader, loader_name=None, ckpt_path=None):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'])
        model.eval()

        all_idx, all_pred, all_gt = [], [], []
        for data_idx, data_x, data_y in loader:
            # data_x = (feats, coords) | data_y = (label_slide, label_patch)
            X = data_x[0].cuda() 
            data_label = data_y[0] 
            if self.cfg['mixup_type'] == 'rankmix' and loader_name == 'eval-train':
                with torch.no_grad():
                    logit_bag, attn = model(X, ret_with_attn=True)
                self.instance_score[data_idx.item()] = attn
            else:
                with torch.no_grad():
                    logit_bag = model(X)
            all_gt.append(data_label)
            all_pred.append(logit_bag.detach().cpu())
            all_idx.append(data_idx)
        
        all_pred = torch.cat(all_pred, dim=0) # [B, num_cls]
        all_gt = torch.cat(all_gt, dim=0).squeeze() # [B, ]
        all_idx = torch.cat(all_idx, dim=0).squeeze() # [B, ]

        cltor = dict()
        cltor['pred'] = {'y': all_gt, 'y_hat': all_pred, 'idx': all_idx}

        return cltor

    def test_model_with_in_between_data(self, model, loader, loader_name=None, ckpt_path=None):
        if ckpt_path is not None:
            net_ckpt = torch.load(ckpt_path)
            model.load_state_dict(net_ckpt['model'])
        model.eval()
        
        N_DATA, TEST_EPOCH, TEST_BATCH = len(loader), 30, 16
        all_mix_idx, all_mix_lam, all_mix_y, all_mix_y_hat, all_mix_loss = [], [], [], [], []
        for ITH_EPOCH in range(TEST_EPOCH):
            i_batch, batch_idx, batch_x, batch_y = 0, [], [], []
            for data_idx, data_x, data_y in loader:
                i_batch += 1
                # data_x = (feats, coords) | data_y = (label_slide, label_patch)
                X = data_x[0].cuda() 
                data_label = data_y[0].cuda()

                batch_idx.append(data_idx)
                batch_x.append(X)
                batch_y.append(data_label)

                if i_batch % TEST_BATCH == 0 or i_batch == N_DATA:
                    pseb_ind = self._collect_pseb_ind('exec-test', batch_idx, batch_x)

                    x_mixed, y_a, y_b, lam, idx_b = augment_bag(
                        batch_x, batch_y,
                        alpha=1.0,
                        method='psebmix',
                        mixup_lam_from='content',
                        psebmix_n=self.cfg['pseb_n'],
                        psebmix_ind=pseb_ind, 
                        psebmix_prob=1.1
                    )

                    num_batch = len(x_mixed)
                    with torch.no_grad():
                        y_hat = []
                        for X in x_mixed:
                            logit_bag = model(X)
                            y_hat.append(logit_bag)

                        bag_preds = torch.cat(y_hat, dim=0) # [B, num_cls]
                        bag_label_a = torch.cat(y_a, dim=0).squeeze(-1) # [B, ]
                        bag_label_b = torch.cat(y_b, dim=0).squeeze(-1) # [B, ]
                        # prepare lam
                        if not isinstance(lam, list): # a float
                            lam = [lam] * num_batch
                        assert len(lam) == num_batch
                        new_lam = torch.FloatTensor(lam * bag_preds.shape[1]).view(bag_preds.shape[1], -1).permute(1, 0).cuda()
                        clf_loss = self.loss(bag_preds, bag_label_a, new_lam, ret_mean=False) + self.loss(bag_preds, bag_label_b, 1. - new_lam, ret_mean=False)
                        if len(clf_loss.shape) > 1 and clf_loss.shape[1] > 1:
                            clf_loss = clf_loss.mean(dim=-1) 
                        clf_loss_mean = clf_loss.mean()
                        print("[test in-between data at {}-th epoch] {}-th batch: clf_loss = {:.6f}".format(ITH_EPOCH, i_batch, clf_loss_mean.item()))
                        wandb.log({'test_batch/clf_loss': clf_loss_mean.item()})
                        
                        # collect the data used in mixup
                        for j in range(num_batch):
                            all_mix_idx.append((batch_idx[j], batch_idx[idx_b[j]]))
                            all_mix_lam.append(lam[j])
                            all_mix_y.append((y_a[j].item(), y_b[j].item()))
                            all_mix_y_hat.append(y_hat[j].detach().cpu())
                            assert y_b[j].item() == y_a[idx_b[j]].item()
                            all_mix_loss.append(clf_loss[j].item())

                        # reset mini-batch
                        batch_idx, batch_x, batch_y = [], [], []
                        torch.cuda.empty_cache()

        res = {'mix_idx_a': torch.cat([x[0] for x in all_mix_idx], dim=0), 'mix_idx_b': torch.cat([x[1] for x in all_mix_idx], dim=0),
           'mix_y_a': [x[0] for x in all_mix_y], 'mix_y_b': [x[1] for x in all_mix_y],
           'mix_lam': all_mix_lam, 'mix_y_hat': torch.cat(all_mix_y_hat, dim=0), 'mix_loss': all_mix_loss}
        return res

    def _get_state_dict(self, epoch):
        return {
            'epoch': epoch,
            'model': self.net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }

    def save_model(self, epoch, ckpt_type='best', run_name='train'):
        net_ckpt_dict = self._get_state_dict(epoch)
        if ckpt_type == 'last':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            torch.save(net_ckpt_dict, add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))

    def resume_model(self, ckpt_type='best', run_name='train'):
        if ckpt_type == 'last':
            net_ckpt = torch.load(add_prefix_to_filename(self.last_ckpt_path, prefix=run_name))
        elif ckpt_type == 'best':
            net_ckpt = torch.load(add_prefix_to_filename(self.best_ckpt_path, prefix=run_name))
        else:
            raise KeyError("Expected best or last for `ckpt_type`, but got {}.".format(ckpt_type))
        self.net.load_state_dict(net_ckpt['model']) 
        self.optimizer.load_state_dict(net_ckpt['optimizer']) 
        print('[model] resume the network from {}_{} at epoch {}...'.format(ckpt_type, run_name, net_ckpt['epoch']))

