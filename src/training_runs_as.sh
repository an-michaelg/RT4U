## APR 10 AS CE MIL runs with 0.5 coeff (half MIL half normal)
python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_mil_005 ++trainer.max_epochs=10
#python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_mil_05 ++trainer.max_epochs=15

# ## APR 09 AS CE MIL runs testing mechanics
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_mil_05 ++trainer.max_epochs=10

# ## MAR 30 2025 AS CE runs with fixed mild/normal labels
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_logit_mmfix ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## MAR 26 2025 AS CE runs with fixed mild/normal labels, training on everything to do error analysis
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_allsplits ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## FEB 12 2025 AS Evidential runs with evidential pseudolabel
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_logit ++model.init_args.loss=ce ++pseudo_calibrate=true ++pseudo_method=avg_logit

# ## FEB 04 2025 AS Evidential runs with evidential pseudolabel
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ev_nocalib ++model.init_args.loss=evidential ++pseudo_calibrate=false ++pseudo_method=evidence

# ## JAN 27 2025 AS CE runs with avg_pred mode
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_avgpred