ISAACLAB_CONDA_ENV ?= xr_isaaclab
ISAACLAB_RUN = conda run -n $(ISAACLAB_CONDA_ENV)

g1_pp_record:
	./isaaclab.sh -p scripts/tools/record_demos.py \
	--device cpu \
	--task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
	--teleop_device handtracking \
	--dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
	--num_demos 10 \
	--enable_pinocchio

g1_pp_replay:
	./isaaclab.sh -p scripts/tools/replay_demos.py \
	--device cpu \
	--task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
	--dataset_file ./datasets/dataset_g1_locomanip.hdf5 \
	--enable_pinocchio

g1_pp_annotation:
	./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
	--device cpu \
	--task Isaac-Locomanipulation-G1-Abs-Mimic-v0 \
	--input_file ./datasets/dataset_g1_locomanip.hdf5 \
	--output_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
	--enable_pinocchio

g1_pp_generation:
	./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
	--device cpu \
	--headless \
	--num_envs 20 \
	--generation_num_trials 1000 \
	--enable_pinocchio \
	--input_file ./datasets/dataset_annotated_g1_locomanip.hdf5 \
	--output_file ./datasets/generated_dataset_g1_locomanip.hdf5

g1_pp_train:
	$(ISAACLAB_RUN) ./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
	--task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 --algo bc \
	--normalize_training_actions \
	--dataset ./datasets/generated_dataset_g1_locomanip.hdf5

g1_pp_play:
	$(ISAACLAB_RUN) ./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
	--device cpu \
	--enable_pinocchio \
	--task Isaac-PickPlace-Locomanipulation-G1-Abs-v0 \
	--num_rollouts 10 \
	--horizon 400 \
	--norm_factor_min -1.7463293075561523 \
	--norm_factor_max 1.7463293075561523 \
	--checkpoint logs/robomimic/Isaac-PickPlace-Locomanipulation-G1-Abs-v0/bc_rnn_low_dim_g1/20251201153000/models/model_epoch_2000.pth

.PHONY: g1_pp_record g1_pp_replay g1_pp_annotation g1_pp_generation g1_pp_train g1_pp_play g1_pw_record g1_pw_replay g1_pw_annotation g1_pw_verify g1_pw_generation g1_pw_ds_replay g1_pw_train


DATE ?= 1223_1
TASK_NAME ?= Isaac-PourWater-G1-InspireFTP-Abs-v0
# TASK_NAME ?= Isaac-PickPlace-G1-InspireFTP-Abs-v0
TASK_MIMIC_NAME ?= Isaac-PourWater-G1-InspireFTP-Abs-Mimic-v0

CURRENT_DATE ?= 1223_1
RECORD_G1_TASK_DATASET ?= ./datasets/$(TASK_NAME)/dataset_$(DATE).hdf5
ANNOTATE_G1_TASK_DATASET ?= ./datasets/$(TASK_NAME)/dataset_annotated_$(CURRENT_DATE).hdf5
GENERATED_G1_TASK_DATASET ?= ./datasets/$(TASK_NAME)/generated_dataset_$(CURRENT_DATE).hdf5
GENERATED_G1_TASK_FAILED_DATASET ?= ./datasets/$(TASK_NAME)/generated_dataset_$(CURRENT_DATE)_failed.hdf5


g1_task_record:
	./isaaclab.sh -p scripts/tools/record_demos.py \
	--device cpu \
	--task $(TASK_NAME) \
	--teleop_device handtracking \
	--dataset_file $(RECORD_G1_TASK_DATASET) \
	--num_demos 10 \
	--enable_pinocchio

g1_task_replay:
	./isaaclab.sh -p scripts/tools/replay_demos.py \
	--device cpu \
	--task $(TASK_NAME) \
	--dataset_file $(RECORD_G1_TASK_DATASET) \
	--enable_pinocchio

g1_task_annotation:
	./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/annotate_demos.py \
	--device cpu \
	--task $(TASK_MIMIC_NAME) \
	--input_file $(RECORD_G1_TASK_DATASET) \
	--output_file $(ANNOTATE_G1_TASK_DATASET) \
	--enable_pinocchio \
	--auto

g1_task_verify:
	python scripts/tools/verify_annotated_data.py \
	--dataset $(ANNOTATE_G1_TASK_DATASET) \
	--required_subtasks grasp_bottle move

g1_task_generation:
	./isaaclab.sh -p scripts/imitation_learning/isaaclab_mimic/generate_dataset.py \
	--device cpu \
	--num_envs 20 \
	--generation_num_trials 1000 \
	--enable_pinocchio \
	--input_file $(ANNOTATE_G1_TASK_DATASET) \
	--output_file $(GENERATED_G1_TASK_DATASET)

g1_task_ds_replay:
	./isaaclab.sh -p scripts/tools/replay_demos.py \
	--device cpu \
	--task $(TASK_NAME) \
	--dataset_file $(GENERATED_G1_TASK_DATASET) \
	--enable_pinocchio

g1_task_ds_failed_replay:
	./isaaclab.sh -p scripts/tools/replay_demos.py \
	--device cpu \
	--task $(TASK_NAME) \
	--dataset_file $(GENERATED_G1_TASK_FAILED_DATASET) \
	--enable_pinocchio

g1_task_train:
	./isaaclab.sh -p scripts/imitation_learning/robomimic/train.py \
	--task $(TASK_NAME) \
	--algo bc \
	--normalize_training_actions \
	--dataset $(GENERATED_G1_TASK_DATASET)

g1_task_play:
	./isaaclab.sh -p scripts/imitation_learning/robomimic/play.py \
	--device cuda \
	--enable_pinocchio \
	--task $(TASK_NAME) \
	--num_rollouts 10 \
	--horizon 400 \
	--norm_factor_min -0.5991024971008301 \
	--norm_factor_max 1.651545524597168 \
	--checkpoint logs/robomimic/$(TASK_NAME)/bc_rnn_low_dim_gr1t2/20251223135618/models/model_epoch_2000.pth

convert_glb_to_usd:
	python scripts/tools/batch_convert_glb.py \
	--input_dir /home/ubuntu22/sourcecode/unitree_sim_isaaclab/models/glb \
	--output_dir /home/ubuntu22/sourcecode/unitree_sim_isaaclab/models/usd2 \
	--collision-approximation none \
	--headless

mesh:
	python scripts/tools/mesh2usd.py \
		--folders ./assets/glb \
		--max-models 1000000 \
		--load-materials \
		--dist-folder ./assets/usd

.PHONY: g1_task_data_remove g1_task_record g1_task_replay g1_task_annotation g1_task_verify g1_task_generation g1_task_ds_replay g1_task_ds_failed_replay g1_task_train g1_task_play convert_glb_to_usd