python main.py main --model='GRU' \
    --embedding_dim 100 --embedding_path "./data/wordvec_embsize_100_window_4_mincount_1.bin" \
    --attn_head 1 --device=0 --data_path='./data/new_train_data.csv'

# python main.py main --model='Fasttext' --attn_head 1 --device=0 --data_path='./data/new_train_data.csv'