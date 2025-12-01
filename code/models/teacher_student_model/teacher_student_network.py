import torch
import torch.nn as nn
import os
from models.ensemble_model.ensemble_model import EavNet, MiniEavNet
from models.teacher_student_model.model_config import model_config


# class Teacher(nn.Module):
#     def __init__(self, ts_model_config):
#         super(Teacher, self).__init__()
#         if ts_model_config.teacher_model == 'EavNet':
#             self.teacher = EavNet(eav_model_config)
#         else:
#             self.teacher = None
#             print('unable to set teacher network ........................')
#
#         checkpoint = torch.load(os.path.join(ts_model_config.teacher_pretrained_model_path))
#         self.teacher.load_state_dict(checkpoint["state_dict"])
#
#     def get_end_classifier(self):
#         return self.teacher.get_end_classifier()
#
#     def forward(self, vid_v, vid_av, mels, aud_a):
#         v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden = self.teacher(vid_v, vid_av, mels, aud_a)
#         return v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden
#
#
# class Student(nn.Module):
#     def __init__(self, ts_model_config):
#         super(Student, self).__init__()
#         if ts_model_config.student_model == 'EavNet':
#             self.student = EavNet(eav_model_config)
#             # checkpoint = torch.load(os.path.join(ts_model_config.teacher_pretrained_model_path))
#             # self.teacher.load_state_dict(checkpoint["state_dict"])
#         elif ts_model_config.student_model == 'MiniEavNet':
#             self.student = MiniEavNet(distill_eav_model_config)
#         else:
#             self.student = None
#             print('unable to set student network ........................')
#
#     def get_end_classifier(self):
#         return self.student.get_end_classifier()
#
#     def forward(self, vid_v, vid_av, mels, aud_a):
#         v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden = self.student(vid_v, vid_av, mels, aud_a)
#         return v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden


class TeacherStudentNetwork(nn.Module):
    def __init__(self, ts_model_config):
        super(TeacherStudentNetwork, self).__init__()
        # teacher network
        if ts_model_config.teacher_model == 'EavNet' and ts_model_config.teacher_model == ts_model_config.student_model:
            self.teacher = EavNet(model_config)
            self.student = EavNet(model_config)
            self.teacher, self.student = self.set_network_weights(ts_model_config)


        elif ts_model_config.teacher_model == 'MiniEavNet' and ts_model_config.teacher_model == ts_model_config.student_model:
            self.teacher = MiniEavNet(model_config)
            self.student = MiniEavNet(model_config)
            self.teacher, self.student = self.set_network_weights(ts_model_config)

        elif ts_model_config.teacher_model == 'EavNet' and ts_model_config.student_model == 'MiniEavNet':
            self.teacher = EavNet(model_config)
            self.student = MiniEavNet(model_config)
            self.teacher, self.student = self.set_network_weights(ts_model_config)
        else:
            self.teacher = None
            print('unable to set teacher network ........................')

        # freeze teacher model
        for param in self.teacher.parameters():
            param.requires_grad = False



    def set_network_weights(self, ts_model_config):
        if ts_model_config.teacher_model == ts_model_config.student_model:
            checkpoint = torch.load(os.path.join(ts_model_config.teacher_pretrained_model_path), weights_only=False)
            self.teacher.load_state_dict(checkpoint["state_dict"])
            self.student.load_state_dict(checkpoint["state_dict"])
        else:
            checkpoint = torch.load(os.path.join(ts_model_config.teacher_pretrained_model_path), weights_only=False)
            self.teacher.load_state_dict(checkpoint["state_dict"])

        return self.teacher, self.student

    def set_student_weights(self, student_weights):
        self.student.load_state_dict(student_weights)
    
    def get_student_weights(self):
        return self.student.state_dict()

    def forward(self, vid_v, vid_av, mels, aud_a):
        with torch.no_grad():
            t_v_outputs, t_av_outputs, t_a_outputs, t_v_hidden, t_feature_list, t_a_hidden, t_av_hidden = self.teacher(vid_v, vid_av, mels, aud_a)

        
        s_v_outputs, s_av_outputs, s_a_outputs, s_v_hidden, s_feature_list, s_a_hidden, s_av_hidden = self.student(vid_v, vid_av, mels, aud_a)

        return (
            t_v_outputs, t_av_outputs, t_a_outputs, t_v_hidden, t_feature_list, t_a_hidden, t_av_hidden,
            s_v_outputs, s_av_outputs, s_a_outputs, s_v_hidden, s_feature_list, s_a_hidden, s_av_hidden
        )
