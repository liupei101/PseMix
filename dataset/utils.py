from dataset.PatchWSI import WSIPatchClf
from dataset.PatchWSI import WSIProtoPatchClf


def prepare_clf_dataset(patient_ids:list, cfg, **kws):
    """
    Interface for preparing slide-level classification dataset

    patient_ids: a list including all patient IDs.
    cfg: a dict where 'path_patch', 'path_table', and 'feat_format' are included.
    """
    path_patch = cfg['path_patch']
    path_table = cfg['path_table']
    feat_format = cfg['feat_format']
    if 'path_label' in kws:
        path_label = kws['path_label']
    else:
        path_label = None
    if 'ratio_sampling' in kws:
        ratio_sampling = kws['ratio_sampling']
    else:
        ratio_sampling = None
    if 'ratio_mask' in kws:
        if cfg['test']: # only used in a test mode
            ratio_mask = kws['ratio_mask']
        else:
            ratio_mask = None
    else:
        ratio_mask = None
    if 'filter_slide' in kws:
        if_remove_slide = kws['filter_slide']
    else:
        if_remove_slide = None

    if cfg['mixup_type'] == 'remix':
        dataset = WSIProtoPatchClf(patient_ids, path_patch, path_table)
    else:
        dataset = WSIPatchClf(
            patient_ids, path_patch, path_table, path_label=path_label, read_format=feat_format, ratio_sampling=ratio_sampling, 
            ratio_mask=ratio_mask, coord_path=cfg['path_coord'], filter_slide=if_remove_slide
        )
    return dataset
