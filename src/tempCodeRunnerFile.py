    # if cfg.TEST.MODEL_FILE:
    #     logger.info('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
    #     state = torch.load(cfg.TEST.MODEL_FILE)
    #     if 'best_state_dict' in state.keys():
    #         state = state['best_state_dict']
    #     state = model_key_helper(state)
    #     model.load_state_dict(state)
    # else:
    #     model_state_file = os.path.join(
    #         final_output_dir, 'final_state.pth'
    #     )
    #     logger.info('=> loading model from {}'.format(model_state_file))
    #     model.load_state_dict(model_key_helper(torch.load(model_state_file)))