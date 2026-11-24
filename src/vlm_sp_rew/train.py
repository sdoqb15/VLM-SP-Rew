"""Train the paper ObjectNav learner on a frozen scene/goal/start manifest."""
from vlm_sp_rew.training_utils import (build_arg_parser, setup_gpu_env, setup_random_seed,
    print_training_config, run_training, per_episode_callback)
from vlm_sp_rew.environment import ObjectNavEnv


def main():
    parser = build_arg_parser()
    args = parser.parse_args()
    if not args.episode_manifest:
        parser.error("--episode-manifest is required: prepare and freeze the benchmark episodes first")
    if not args.controller_config or not args.run_record_root:
        parser.error("--controller-config and --run-record-root are required")
    if args.max_episode_steps != 200:
        parser.error("The paper experiment horizon is 200 actions")
    if args.steps <= 0 or args.checkpoint_every <= 0:
        parser.error("Positive steps and checkpoint interval required")
    if args.split != "train":
        parser.error("Training consumes only train episodes; use evaluate for dev/test")
    from vlm_sp_rew.reward import load_episode_bank
    bank=load_episode_bank(args.episode_manifest,args.data_root)
    if len(bank["goal_vocabulary"])!=1:
        parser.error("Each RGB-only fitted policy has one fixed goal category")
    setup_gpu_env(platform=args.jax_platform)
    setup_random_seed(args.seed)
    print_training_config(args)
    run_training(args, ObjectNavEnv, per_episode_callback)


if __name__ == "__main__":
    main()
