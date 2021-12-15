export NCCL_DEBUG=INFO
export NCCL_DEBUG_SUBSYS=ALL

log_path=log_d4_nccl.log
res_path=joined_d4.json

python -m torch.distributed.launch --nproc_per_node 4 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 8898 pretrain_bert.py --num-layers 24 --hidden-size 1024 --num-attention-heads 16 --micro-batch-size 4 --global-batch-size 16 --seq-length 512 --max-position-embeddings 512 --train-iters 1000000 --save examples/bert/checkpoints/bert_345m --load examples/bert/checkpoints/bert_345m --data-path examples/bert/dataset/my_bert_text_sentence --vocab-file examples/bert/bert-large-uncased-vocab.txt --data-impl mmap --split 949,50,1 --distributed-backend nccl --lr 0.0001 --lr-decay-style linear --min-lr 1.0e-5 --lr-decay-iters 990000 --weight-decay 1e-2 --clip-grad 1.0 --lr-warmup-fraction .01 --log-interval 100 --save-interval 10000 --eval-interval 1000 --eval-iters 10 --fp16 > ${log_path}
echo "finished training profile"


python lt_scripts/join_log/nccl_pf_joiner.py -p log/bert35_4t4_p2 -n ${log_path} --iters 15 -o ${res_path}
