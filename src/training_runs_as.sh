## MAR 30 2025 AS CE runs with fixed mild/normal labels
python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_logit_mmfix ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## MAR 26 2025 AS CE runs with fixed mild/normal labels, training on everything to do error analysis
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_allsplits ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## FEB 12 2025 AS Evidential runs with evidential pseudolabel
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_logit ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## FEB 04 2025 AS Evidential runs with evidential pseudolabel
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ev_nocalib ++model.init_args.loss=evidential ++pseudo_calibrate=false ++pseudo_method=evidence

# ## JAN 27 2025 AS CE runs with avg_pred mode
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_avgpred