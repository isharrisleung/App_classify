python main.py main --model='Transformer_TextCNN' \
    --embedding_dim 100 --vocab_size 2320 --embedding_path "./data/wordvec_embsize_100_window_4_mincount_2.bin" \
    --attn_head 1 --device=0 --data_path='./data/wordvec_embsize_100_window_4_mincount_2.csv'

# python main.py main --model='Fasttext' --attn_head 1 --device=0 --data_path='./data/new_train_data.csv'