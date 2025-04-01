## FEB 12 2025 TMED Evidential run with evidence mode and calibration
python main_multi_round.py --config-name=config_tmed ++model.init_args.loss=ce ++logger.init_args.name=tmed_ce_logit ++pseudo_calibrate=true ++pseudo_method=avg_logit

### FEB 05 2025 TMED Evidential run with evidence mode and calibration
#python main_multi_round.py --config-name=config_tmed ++model.init_args.loss=evidential ++logger.init_args.name=tmed_ev_nocalib ++pseudo_calibrate=false ++pseudo_method=evidence

# ## FEB 03 2025 TMED Evidential run with avg_pred_mode as a placeholder
# python main_multi_round.py --config-name=config_tmed ++model.init_args.loss=evidential ++logger.init_args.name=tmed_as_ev_avgpred

# ## JAN 28 2025 TMED CE runs with avg_pred mode
# python main_multi_round.py --config-name=config_tmed ++logger.init_args.name=tmed_as_ce