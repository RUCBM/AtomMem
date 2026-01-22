python ./AtomMem/taskutils/script/processing_multiquery_musique.py \
    --data_file ./AtomMem/taskutils/memory_data/musique_ans_v1.0_train.jsonl \
    --data_size 20000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/musique_train_multiquery.parquet

python ./AtomMem/taskutils/script/processing_multiquery_musique.py \
    --data_file ./AtomMem/taskutils/memory_data/musique_ans_v1.0_dev.jsonl \
    --data_size 1000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/musique_dev_multiquery.parquet