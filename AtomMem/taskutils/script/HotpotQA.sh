python ./taskutils/script/processing_multiquery.py \
    --data_file ./taskutils/memory_data/hotpotqa_train.json \
    --data_size 20000 \
    --doc_num 200 \
    --output ./taskutils/memory_data/hotpotqa_train_multiquery.parquet

python ./taskutils/script/processing_multiquery.py \
    --data_file ./taskutils/memory_data/hotpotqa_dev.json \
    --data_size 600 \
    --doc_num 200 \
    --output ./taskutils/memory_data/hotpotqa_dev_multiquery.parquet
