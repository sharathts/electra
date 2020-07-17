horovodrun -np 8 python run_pretrain.py \
    --model_name='test00' \
    --data_dir='pretraining_data/' \
    --pretrain_tfrecords='pretraining_data/pretrain_tfrecords/pretrain_data*' \
    --num_train_steps=766000 \
    --num_warmup_steps=10000 \
    --disc_weight=50.0 \
    --generator_hidden_size=0.3333333 \
    --learning_rate=2e-4 \
    --train_batch_size=16 \
    --max_seq_length=512 \
    --save_checkpoints_steps=1000