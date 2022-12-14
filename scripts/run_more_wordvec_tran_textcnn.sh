python main.py main --model='Transformer_TextCNN' \
    --embedding_dim 100 --vocab_size 2776 --embedding_path "./data/more_data_wordvec_embsize_100_window_4_mincount_1.bin" \
    --attn_head 1 --device=0 --data_path='./data/more_data.csv' \
    --fold 5 \
    --name_max_text_len 15 --desc_max_text_len 85 \
    --transformer_layer_num 1

# python main.py main --model='Fasttext' --attn_head 1 --device=0 --data_path='./data/new_train_data.csv'