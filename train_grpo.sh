export CUDA_VISIBLE_DEVICES=6,7

DATA_DIR=/data/liangjh/verl/data/gsm8k
MODEL_DIR=/data/liangjh/model_set/Qwen2.5-0.5B-Instruct

experiment="Qwen2.5-0.5B-Instruct-gsm8k-grpo"
tensorboard_dir="./output/${experiment}/tensorboard"
output_dir="./output/${experiment}"
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

cp $0 $output_dir

TENSORBOARD_DIR=$tensorboard_dir PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=grpo \
 data.train_files=$DATA_DIR/train.parquet \
 data.val_files=$DATA_DIR/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=512 \
 data.max_response_length=256 \
 actor_rollout_ref.model.path=$MODEL_DIR \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.free_cache_engine=False \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.rollout.n=5 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 actor_rollout_ref.actor.use_kl_loss=True \
 actor_rollout_ref.actor.kl_loss_coef=0.001 \
 actor_rollout_ref.actor.kl_loss_type=low_var_kl \
 actor_rollout_ref.actor.entropy_coeff=0 \
 actor_rollout_ref.model.enable_gradient_checkpointing=True \
 algorithm.use_kl_in_reward=False \
 trainer.logger=['console','tensorboard'] \
 trainer.critic_warmup=0 \
 trainer.default_local_dir=$output_dir \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=50 \
 trainer.test_freq=10 \
 trainer.total_epochs=10 \
 2>&1 | tee "./output/${experiment}/${experiment}.log" &