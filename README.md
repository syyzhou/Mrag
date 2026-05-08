python supervised-fine-tune-qlora-two-router-llama2-7b.py  --model_name_or_path ./Qwen2.5-3B --bf16=True --output_dir ./output/train_multiview --use_flash_attn=False --dataset ./rag/multiview_enriched_qa/train_qa_multiview.json --low_rank_training True --num_train_epochs 1  --per_device_train_batch_size 1     --per_device_eval_batch_size 2       --evaluation_strategy "no"     --gradient_accumulation_steps 8 --save_strategy "no"     --save_steps 100     --learning_rate 2e-5     --weight_decay 0.0     --warmup_steps 20     --lr_scheduler_type "constant_with_warmup"     --logging_steps 1     --logging_dir ./logs  --deepspeed "ds_configs/stage2.json"  --tf32 True 
--use_trajectory_routing True     --trajectory_embedding_path ./datasets/nyc/preprocessed/trajectory_embeddings.pt     --trajectory_fusion_mode gate     --share_traj_projector False  --router1_use_shared_expert True   --router1_shared_expert_weight 1.0
<!-- --model_max_length 8192 -->


eval 导入的model5/8，生成长度4k 8k 32k，数据集，model_path，回答几个答案
python eval_two_router-llama2-7b.py --model_path ./Qwen2.5-3B --dataset_name nyc --output_dir ./output/train_multiview --test_file "./rag/multiview_enriched_qa/test_qa_multiview.txt"  --trajectory_embedding_path ./datasets/nyc/preprocessed/test_embeddings.pt


conda env create -f environment.yml