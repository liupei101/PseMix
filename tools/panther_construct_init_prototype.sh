WORK_DIR=/path/to/PseMix
DIR_TO_CONCH_FEAT=/path/to/CONCH_feats/pt_files

cd ${WORK_DIR}

echo "Constructing initial prototypes with KMeans..."

for k in {0..3}; do
    CUDA_VISIBLE_DEVICES=0 python panther_prototype.py \
        --mode kmeans \
        --data_source ${DIR_TO_CONCH_FEAT} \
        --save_dir ./data_split/tcga_brca \
        --label_path ./data_split/tcga_brca/TCGA_BRCA_path_full_subtype.csv \
        --split_path ./data_split/tcga_brca/TCGA_BRCA-fold{}.npz \
        --split_fold ${k} \
        --split_names train \
        --in_dim 512 \
        --n_proto_patches 1000000 \
        --n_proto 8 \
        --n_init 5 \
        --seed 1 \
        --num_workers 10
done