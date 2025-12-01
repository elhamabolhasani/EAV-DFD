import os
from .model_config import model_config


class TsConfig:
    teacher_model = 'MiniEavNet'
    student_model = 'MiniEavNet'  # ['EavNet', 'MiniEavNet']

    # teacher_pretrained_model_path = it will be best model path of model_config file
    model_root_path = '/models/teacher_model_fakeavceleb/'


    run_name = 'mini_b24_v20_400ne_mean_clip_prob_lr0.0001_optimizer(adam0)_scheduler(step_lr0.8-15)_and_(aug_4)_w3.0_1_c_loss_0.005'
    info_name = model_config.run_name
    classifier_type = 'split'
    model_name = 'checkpoint_step000000100.pth'
    teacher_pretrained_model_path = os.path.join(model_root_path, run_name, model_name)

    # config files
    config_file_path = '/models/teacher_student_model/model_config.py'
    ts_config_file_path = '/models/teacher_student_model/teacher_student_config.py'

    config_file_name = 'model_config.py'
    ts_config_file_name = 'teacher_student_config.py'

    #  model dirs
    models_root = '/models/'
    checkpoint_path = models_root + 'student_model/'
    tensorboard_path = '/student_model_tensorboard/'
    model_checkpoint_path = None  # for more training

    if teacher_model == student_model == 'MiniEavNet':
        kind = 'MM_' + info_name
    elif teacher_model == student_model == 'EavNet':
        kind = 'EE_' + info_name
    else:
        kind = 'EM_' + info_name

    # teacher student loss parameters
    ce_loss_flag = True
    ce_loss_var = 0.25
    joint_loss_flag = True
    adapt_mse_loss_flag = True
    adapt_av_loss_flag = True
    adapt_av_loss_var = 1
    kl_output_loss_flag = False
    dynamic_weight = False

    ce_loss = 'bce_weighted'  # bce, bce_weighted
    loss_fn = 'bce_with_logits'  # bce_with_logits, bce_with_logits_weighted
    # train class num : fake->  tensor([14829.])   real->  tensor([5498.])
    loss_weight_a_v = 1
    loss_weight_av = 3
    loss_weight_ce = 5


    loss_function = 'transdistill_layer_selection_clip'
    loss_config = 'transdistill_layer_select_f3_clip_f3_av4_va1_T25'
    loss_function_name = "models.teacher_student_model.adaptation_loss.{}".format(loss_function)
    mse_loss_function = 'models.teacher_student_model.adaptation_loss.mse_loss'
    kl_loss_function = 'models.teacher_student_model.adaptation_loss.KD'
    loss_subconfig = dict(zip(['fus', 'av', 'va'], [3, 4, 1]))
    alpha = 1
    temperature = 25
    mse_temperature = 2
    kl_temperature = 2

    loss_fn_name = ''
    if ce_loss_flag:
        loss_fn_name += '_ce(' + str(ce_loss_var) + ')'

    if ce_loss == 'bce_weighted':
        loss_fn_name += '_(' + str(loss_weight_ce) + ')'

    if joint_loss_flag:
        loss_fn_name += '_jl'

    if adapt_mse_loss_flag:
        loss_fn_name += '_mse'
    if adapt_av_loss_flag:
        loss_fn_name += '_dav'
    if kl_output_loss_flag:
        loss_fn_name += '_kl'
    if loss_fn == 'bce_with_logits_weighted':
        loss_fn_name += '_w(av_' + str(loss_weight_av) + ', _a_v_' + str(loss_weight_a_v) + ')'
    if dynamic_weight:
        loss_fn_name += '_dw'
    
    if adapt_av_loss_var != 1:
        loss_fn_name += str(adapt_av_loss_var)

    tensorboard_dir = os.path.join(tensorboard_path, run_name, kind + loss_fn_name)
    checkpoint_dir = os.path.join(checkpoint_path, run_name, kind + loss_fn_name)


ts_config = TsConfig()
