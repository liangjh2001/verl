export CUDA_VISIBLE_DEVICES=3,4

DATA_DIR=/data/liangjh/verl/data/countdown_qwen_instruct_new
MODEL_DIR=/data/liangjh/model_set/Qwen2.5-0.5B-Instruct

experiment="Qwen2.5-0.5B-Instruct-countdown-ppo-reasoning"
tensorboard_dir="./output/${experiment}/tensorboard"
output_dir="./output/${experiment}"
if [ ! -d $output_dir ]; then
  mkdir -p $output_dir
fi

cp $0 $output_dir

TENSORBOARD_DIR=$tensorboard_dir PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
 algorithm.adv_estimator=gae \
 data.train_files=$DATA_DIR/train.parquet \
 data.val_files=$DATA_DIR/test.parquet \
 data.train_batch_size=256 \
 data.max_prompt_length=256 \
 data.max_response_length=1024 \
 actor_rollout_ref.model.path=$MODEL_DIR \
 actor_rollout_ref.rollout.enforce_eager=False \
 actor_rollout_ref.rollout.free_cache_engine=False \
 actor_rollout_ref.actor.optim.lr=1e-6 \
 actor_rollout_ref.actor.ppo_mini_batch_size=64 \
 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
 actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
 actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
 critic.optim.lr=1e-5 \
 critic.model.path=$MODEL_DIR \
 critic.model.enable_gradient_checkpointing=True \
 critic.ppo_micro_batch_size_per_gpu=1 \
 algorithm.kl_ctrl.kl_coef=0.001 \
 trainer.logger=['console','tensorboard'] \
 trainer.default_local_dir=$output_dir \
 trainer.val_before_train=False \
 trainer.default_hdfs_dir=null \
 trainer.n_gpus_per_node=2 \
 trainer.nnodes=1 \
 trainer.save_freq=1000 \
 trainer.test_freq=10 \
 trainer.total_epochs=5 \
 2>&1 | tee "./output/${experiment}/${experiment}.log" &