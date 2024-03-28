python main_multi_round.py --config-name=config_cifar ++model.init_args.loss="mae" ++logger.init_args.name="cifar_mae" 
python main_multi_round.py --config-name=config_cifar ++model.init_args.loss="nce_rce" ++logger.init_args.name="cifar_nce_rce" 
python main_multi_round.py --config-name=config_cifar ++model.init_args.loss="anl_ce" ++logger.init_args.name="cifar_anl_ce"