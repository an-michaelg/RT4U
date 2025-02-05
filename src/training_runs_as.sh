## FEB 04 2025 AS Evidential runs with evidential pseudolabel
python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ev_nocalib ++model.init_args.loss=evidential ++pseudo_calibrate=false ++pseudo_method=evidence

# ## JAN 27 2025 AS CE runs with avg_pred mode
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_avgpred