python ./AtomMem/taskutils/script/processing_multiquery_musique.py \
    --data_file 2wikimultihot_train.json \
    --data_size 20000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/2wiki_train_multiquery.parquet

python ./AtomMem/taskutils/script/processing_multiquery_musique.py \
    --data_file 2wikimultihot_dev.json \
    --data_size 1000 \
    --doc_num 200 \
    --output ./AtomMem/taskutils/memory_data/2wiki_dev_multiquery.parquet