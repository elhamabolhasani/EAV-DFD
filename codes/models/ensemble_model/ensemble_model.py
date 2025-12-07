from torch import nn
from models.audio_model.audio_model import HubertModel
from models.mcx_model.visual_models import API_Net
from models.vocalist.voca_model.model import SyncTransformer as LargeSyncTransformer
from models.vocalist.voca_model.student_model import SyncTransformer as DistillSyncTransformer
import os
import torch
import numpy as np
import torchvision.transforms as transforms

os.environ['CURL_CA_BUNDLE'] = ''


class EavNet(nn.Module):
    def __init__(self, model_config):
        super(EavNet, self).__init__()

        self.model_config = model_config
        # fake = 0 , real = 1
        self.num_classes = 2
        self.av_model = LargeSyncTransformer()
        checkpoint = torch.load(model_config.pretrained_av_path)
        self.av_model.load_state_dict(checkpoint["state_dict"])
        if model_config.visual_model_type == 'vivit':
            self.v_model = ViViT(
                image_size=model_config.v_img_size,  # image size
                frames=model_config.v_context,  # number of frames
                image_patch_size=model_config.image_patch_size,  # image patch size
                frame_patch_size=model_config.frame_patch_size,  # frame patch size
                num_classes=model_config.num_classes,
                dim=model_config.dim,
                spatial_depth=model_config.spatial_depth,  # depth of the spatial transformer
                temporal_depth=model_config.temporal_depth,  # depth of the temporal transformer
                heads=model_config.heads,
                mlp_dim=model_config.mlp_dim,
                dropout=model_config.dropout,
                emb_dropout=model_config.emb_dropout
            )
        else:
            self.v_model = API_Net(num_classes=model_config.num_classes,
                                   model_name=model_config.model_name,
                                   weight_init=model_config.weight_init,
                                   )
            self.v_model.conv = nn.DataParallel(self.v_model.conv, device_ids=[model_config.device_id])
            checkpoint = torch.load(model_config.pretrained_v_path)
            self.v_model.load_state_dict(checkpoint['state_dict'])
            if hasattr(model_config, 'freeze') and model_config.freeze == True:
                for param in self.v_model.conv.parameters():
                    param.requires_grad = False

        self.a_model = HubertModel(model_config.audio_model_checkpoint_path, self.num_classes)
        if hasattr(model_config, 'audio_pretrained_model_path'):
            checkpoint = torch.load(model_config.audio_pretrained_model_path)
            self.a_model.load_state_dict(checkpoint["state_dict"])
        if hasattr(model_config, 'freeze') and model_config.freeze == True:
            self.a_model.freeze_feature_encoder()

        if self.model_config.classifier_type == 'split':
            if self.model_config.visual_model_type == 'vivit':
                self.v_classifier = nn.Sequential(
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 1)
                )
            else:
                self.v_classifier = nn.Sequential(
                    nn.Linear(2048, 1024),
                    nn.Tanh(),
                    nn.Linear(1024, 1)
                )

            self.a_classifier = nn.Sequential(
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )

        else:
            if self.model_config.visual_model_type == 'vivit':
                self.end_classifier = nn.Sequential(
                    # need to set first dim
                    nn.Linear(model_config.dim + 256 + 512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 1)
                )
            else:
                self.end_classifier = nn.Sequential(
                    # need to set first dim
                    nn.Linear(model_config.dim + 256 + 2048, 1024),
                    nn.Tanh(),
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 1)
                )
        if hasattr(model_config, 'contrastive_loss') and model_config.contrastive_loss is True:
            self.projection_layer_av = nn.Sequential(
                nn.Linear(in_features= 640 * self.model_config.v_context, out_features=200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(in_features=200, out_features=200),
                nn.BatchNorm1d(200)
            )
            self.projection_layer_va = nn.Sequential(
                nn.Linear(in_features= 200 * self.model_config.v_context, out_features=200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(in_features=200, out_features=200),
                nn.BatchNorm1d(200)
            )

    def get_end_classifier(self):
        return self.end_classifier

    def calculate_loss_embeddings(self, feature_list):
        out_av = feature_list['AV_Trans']['av_emb'].permute(1, 2, 0)
        out_av = out_av.reshape([out_av.shape[0], out_av.shape[1] * out_av.shape[2]])
        out_av = self.projection_layer_av(out_av)
        out_va = feature_list['AV_Trans']['va_emb'].permute(1, 2, 0)
        out_va = out_va.reshape([out_va.shape[0], out_va.shape[1] * out_va.shape[2]])
        out_va = self.projection_layer_va(out_va)
        feature_list['contrastive_loss'] = {'av_embedding': out_av, 'va_embedding': out_va}
        return feature_list

    def forward(self, vid_v, vid_av, mels, aud_a):
        _, v_hidden = self.v_model(vid_v.clone())
        _, a_hidden = self.a_model(aud_a)
        av_outputs, av_hidden, feature_list = self.av_model(vid_av.clone().detach(), mels.clone().detach())

        if hasattr(self.model_config, 'contrastive_loss') and self.model_config.contrastive_loss is True:
            feature_list = self.calculate_loss_embeddings(feature_list)

        if self.model_config.classifier_type == 'split':
            v_outputs = self.v_classifier(v_hidden)
            a_outputs = self.a_classifier(a_hidden)
            return v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden
        else:
            outputs = self.end_classifier(torch.cat([v_hidden, a_hidden, av_hidden], dim=1))
            return [], outputs.squeeze(-1), [], v_hidden, feature_list, a_hidden, av_hidden


class MiniEavNet(nn.Module):
    def __init__(self, model_config):
        super(MiniEavNet, self).__init__()

        self.model_config = model_config
        # fake = 0 , real = 1
        self.num_classes = 2
        self.av_model = DistillSyncTransformer(d_model=200)

        checkpoint = torch.load(model_config.pretrained_av_path)
        self.av_model.load_state_dict(checkpoint["state_dict"])

        if hasattr(model_config, 'simple_av') and model_config.simple_av is True:
            # We'll calculate the actual input dimension dynamically in forward pass
            # since the exact shapes may vary
            self.av_model = nn.Sequential(
                nn.Linear(281600, 128),  # Placeholder - will be updated in forward
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 200),  # Output: 200-dimensional features (same as transformer)
                nn.Tanh(),
                nn.Linear(200, 1)
            )
            # Flag to indicate we need to update the first layer
            self.av_model_initialized = False

        if model_config.visual_model_type == 'vivit':
            self.v_model = ViViT(
                image_size=model_config.v_img_size,  # image size
                frames=model_config.v_context,  # number of frames
                image_patch_size=model_config.image_patch_size,  # image patch size
                frame_patch_size=model_config.frame_patch_size,  # frame patch size
                num_classes=model_config.num_classes,
                dim=model_config.dim,
                spatial_depth=model_config.spatial_depth,  # depth of the spatial transformer
                temporal_depth=model_config.temporal_depth,  # depth of the temporal transformer
                heads=model_config.heads,
                mlp_dim=model_config.mlp_dim,
                dropout=model_config.dropout,
                emb_dropout=model_config.emb_dropout
            )
            # TODO : loading visual model pretrained path
            # checkpoint = torch.load(model_config.pretrained_v_path)
            # self.v_model.load_state_dict(checkpoint["state_dict"])
        else:
            self.v_model = API_Net(num_classes=model_config.num_classes,
                                   model_name=model_config.model_name,
                                   weight_init=model_config.weight_init,
                                   )
            self.v_model.conv = nn.DataParallel(self.v_model.conv, device_ids=[model_config.device_id])
            checkpoint = torch.load(model_config.pretrained_v_path)
            self.v_model.load_state_dict(checkpoint['state_dict'])
            if hasattr(model_config, 'freeze') and model_config.freeze is True:
                for param in self.v_model.conv.parameters():
                    param.requires_grad = False

        self.a_model = HubertModel(model_config.audio_model_checkpoint_path, self.num_classes)
        if hasattr(model_config, 'audio_pretrained_distill_model_path'):
            checkpoint = torch.load(model_config.audio_pretrained_distill_model_path)
            self.a_model.load_state_dict(checkpoint["state_dict"])
        if hasattr(model_config, 'freeze') and model_config.freeze is True:
            self.a_model.freeze_feature_encoder()

        if self.model_config.classifier_type == 'split':
            if model_config.visual_model_type == 'vivit':
                self.v_classifier = nn.Sequential(
                    nn.Linear(model_config.dim, 512),
                    nn.Tanh(),
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 512),
                    nn.Tanh(),
                    nn.Linear(512, 1)
                )
            else:
                self.v_classifier = nn.Sequential(
                    nn.Linear(2048, 512),
                    nn.Tanh(),
                    nn.Linear(512, 1)
                )
            self.a_classifier = nn.Sequential(
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 256),
                nn.Tanh(),
                nn.Linear(256, 1)
            )
        elif self.model_config.classifier_type == 'joint':
            if model_config.visual_model_type == 'vivit':
                self.end_classifier = nn.Sequential(
                    nn.Linear(model_config.dim + 256 + 200, 512),
                    nn.Tanh(),
                    nn.Linear(512, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1)
                )
            else:
                self.end_classifier = nn.Sequential(
                    nn.Linear(2048 + 256 + 200, 512),
                    nn.Tanh(),
                    nn.Linear(512, 256),
                    nn.Tanh(),
                    nn.Linear(256, 1)
                )
        else:
            v_dim = model_config.dim if model_config.visual_model_type == 'vivit' else 2048
            a_dim = 256
            av_dim = 200
            hidden_dim = 256

            self.v_proj = nn.Linear(v_dim, hidden_dim)
            self.a_proj = nn.Linear(a_dim, hidden_dim)
            self.av_proj = nn.Linear(av_dim, hidden_dim)

            # Simple multi-head self-attention over the three modality tokens
            self.attn = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=4, batch_first=True)

            # Output head after attention-pooled token
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.Tanh(),
                nn.Linear(hidden_dim, 1)
            )
        if self.model_config.contrastive_loss is True:

            self.projection_layer_av = nn.Sequential(
               nn.Linear(in_features= 640 * self.model_config.v_context, out_features=200),
               nn.BatchNorm1d(200),
               nn.ReLU(),
               nn.Linear(in_features=200, out_features=200),
               nn.BatchNorm1d(200)
           )
            self.projection_layer_va = nn.Sequential(
                nn.Linear(in_features= 200 * self.model_config.v_context, out_features=200),
                nn.BatchNorm1d(200),
                nn.ReLU(),
                nn.Linear(in_features=200, out_features=200),
                nn.BatchNorm1d(200)
            )

    def get_end_classifier(self):
        return self.end_classifier

    def calculate_loss_embeddings(self, feature_list):
        out_av = feature_list['AV_Trans']['av_emb'].permute(1, 2, 0)
        out_av = out_av.reshape([out_av.shape[0], out_av.shape[1] * out_av.shape[2]])
        out_av = self.projection_layer_av(out_av)
        out_va = feature_list['AV_Trans']['va_emb'].permute(1, 2, 0)
        out_va = out_va.reshape([out_va.shape[0], out_va.shape[1] * out_va.shape[2]])
        out_va = self.projection_layer_va(out_va)
        feature_list['contrastive_loss'] = {'av_embedding': out_av, 'va_embedding': out_va}
        return feature_list

    def forward(self, vid_v, vid_av, mels, aud_a):
        _, v_hidden = self.v_model(vid_v.clone())
        _, a_hidden = self.a_model(aud_a)
        
        # Handle simple MLP case differently
        if hasattr(self.model_config, 'simple_av') and self.model_config.simple_av is True:
            # Flatten vid_av: [batch, frames, height, width] -> [batch, frames*height*width]
            batch_size = vid_av.shape[0]
            vid_av_flat = vid_av.reshape(batch_size, -1)
            
            # Flatten mels: [batch, 1, 80, 80] -> [batch, 1*80*80]
            mels_flat = mels.reshape(batch_size, -1)
            
            # Concatenate vid_av and mels along feature dimension
            combined_input = torch.cat([vid_av_flat, mels_flat], dim=1)
            
            # Pass through the simple MLP
            av_outputs = self.av_model(combined_input)
            av_outputs = av_outputs.squeeze(-1)  
            av_hidden = av_outputs  # For simple MLP, hidden is the same as outputs
            feature_list = {}  # Empty feature list for simple MLP
            
            # Store concatenated features for t-SNE visualization
            if not hasattr(self, 'tsne_features'):
                self.tsne_features = []
                self.tsne_labels = []
            
            # Concatenate hidden features: [v_hidden, av_hidden, a_hidden]
            # Note: v_hidden and a_hidden come from other models, av_hidden is from simple MLP
            # We'll store them separately and concatenate later when all are available
            self.current_av_hidden = av_hidden
        else:
            # Original transformer case
            av_outputs, av_hidden, feature_list = self.av_model(vid_av.clone().detach(), mels.clone().detach())

        if self.model_config.contrastive_loss is True:
            feature_list = self.calculate_loss_embeddings(feature_list)

        if self.model_config.classifier_type == 'split':
            v_outputs = self.v_classifier(v_hidden)
            a_outputs = self.a_classifier(a_hidden)
            return v_outputs, av_outputs, a_outputs, v_hidden, feature_list, a_hidden, av_hidden
        elif self.model_config.classifier_type == 'joint':
            outputs = self.end_classifier(torch.cat([v_hidden, a_hidden, av_hidden], dim=1))
            return np.array([]), outputs.squeeze(-1), np.array([]), v_hidden, feature_list, a_hidden, av_hidden
        else:
            # Project to common space and build 3-token sequence [B, 3, H]
            v_tok = self.v_proj(v_hidden).unsqueeze(1)
            a_tok = self.a_proj(a_hidden).unsqueeze(1)
            av_tok = self.av_proj(av_hidden).unsqueeze(1)
            tokens = torch.cat([v_tok, a_tok, av_tok], dim=1)

            # Self-attention over tokens; use mean pooling over sequence dimension
            attn_out, _ = self.attn(tokens, tokens, tokens)
            pooled = attn_out.mean(dim=1)

            outputs = self.head(pooled)
            return np.array([]), outputs.squeeze(-1), np.array([]), v_hidden, feature_list, a_hidden, av_hidden

