## APR 21 AS no attn guidance run with logit calibration because I don't know what is real anymore
python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_logit_newcsv ++trainer.max_epochs=15 ++model.init_args.attn_guiding_coeff=0.0 ++pseudo_method=avg_logit ++pseudo_calibrate=true

# ## APR 16 AS attn guidance bumping up higher
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_l2_05 ++trainer.max_epochs=15 ++model.init_args.attn_guiding_coeff=0.5

# ## APR 16 AS attn guidance with L2 loss this time around
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_l2_02 ++trainer.max_epochs=15

## APR 13 AS CE runs with attn guidance shortcut run (starting from round 1)
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_logit ++trainer.max_epochs=15 ++pseudo_calibrate=true ++pseudo_method=avg_logit #++start_from_round=1 ++external_pseudo_file="../logs/as_ce_atg_test/round0/pseudo.csv"

## APR 12 AS CE runs with attn guidance
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_01 ++trainer.max_epochs=15 ++model.init_args.attn_guiding_coeff=0.1
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_test ++trainer.max_epochs=15

# ## APR 12 AS CE runs with attn guidance testing mechanics
# python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_atg_test ++trainer.max_epochs=2

# ## APR 10 AS CE MIL runs with 0.5 coeff (half MIL half normal)
#python main_multi_round.py --config-name=config_as ++logger.init_args.name=as_ce_mil_005 ++trainer.max_epochs=10
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