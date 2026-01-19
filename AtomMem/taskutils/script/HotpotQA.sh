python ./AtomMem/taskutils/script/processing_multiquery.py \
    --data_file hotpotqa_train.json \
    --data_size 20000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/hotpotqa_train_multiquery.parquet

python ./AtomMem/taskutils/script/processing_multiquery.py \
    --data_file hotpotqa_dev.json \
    --data_size 1000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/hotpotqa_dev_multiquery.parquet