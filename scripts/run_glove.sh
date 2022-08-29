python main.py main --model='Transformer_TextCNN' \
    --embedding_dim 100 --vocab_size 1904 --embedding_path "./data/glove_word2vec_100_4.bin" \
    --attn_head 1 --device=0 --data_path='./data/glove_word2vec_100_4.csv' \
    --binary False

# python main.py main --model='Fasttext' --attn_head 1 --device=0 --data_path='./data/new_train_data.csv'