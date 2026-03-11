python ./taskutils/script/processing_multiquery_musique.py \
    --data_file ./taskutils/memory_data/musique_ans_v1.0_train.jsonl \
    --data_size 20000 \
    --doc_num 200 \
    --output ./taskutils/memory_data/musique_train_multiquery.parquet

python ./taskutils/script/processing_multiquery_musique.py \
    --data_file ./taskutils/memory_data/musique_ans_v1.0_dev.jsonl \
    --data_size 600 \
    --doc_num 200 \
    --output ./taskutils/memory_data/musique_dev_multiquery.parquet
