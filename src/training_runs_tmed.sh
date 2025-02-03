#python main_multi_round.py --config-name=config_tmed ++model.init_args.loss="nce_rce" ++logger.init_args.name="tmed_as_nce_rce" 

## FEB 03 2025 TMED Evidential run with avg_pred_mode as a placeholder
python main_multi_round.py --config-name=config_tmed ++model.init_args.loss=evidential ++logger.init_args.name=tmed_as_ev_avgpred

# ## JAN 28 2025 TMED CE runs with avg_pred mode
# python main_multi_round.py --config-name=config_tmed ++logger.init_args.name=tmed_as_ce