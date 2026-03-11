python ./taskutils/script/processing_multiquery.py \
    --data_file 2wikimultihop_train.json \
    --data_size 20000 \
    --doc_num 200 \
    --output ./taskutils/memory_data/2wiki_train_multiquery.parquet

python ./taskutils/script/processing_multiquery.py \
    --data_file 2wikimultihop_dev.json \
    --data_size 600 \
    --doc_num 200 \
    --output ./taskutils/memory_data/2wiki_dev_multiquery.parquet
