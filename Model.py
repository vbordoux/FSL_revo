import torch.nn as nn
import torch.nn.functional as F
import torch
import json
from torchaudio.models import wav2vec2_model

def conv_block(in_channels,out_channels):

    return nn.Sequential(
        nn.Conv2d(in_channels,out_channels,3,padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

class ProtoNet(nn.Module):
    def __init__(self):
        super(ProtoNet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(1,64),
            conv_block(64,64),
            conv_block(64,64),
            conv_block(64,64)
        )
    def forward(self,x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)
        x = self.encoder(x)
        x = nn.MaxPool2d(2)(x)
        
        return x.view(x.size(0),-1) # -1 infer the dimension given that the other one is provided

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, drop_rate=0.0, drop_block=False, block_size=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.num_batches_tracked = 0
        self.drop_block = drop_block
        self.block_size = block_size
        

    def forward(self, x):
        self.num_batches_tracked += 1

        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.maxpool(out)
        out = F.dropout(out, p=self.drop_rate, training=self.training, inplace=True)
        
                

        return out


class ResNet(nn.Module):

    def __init__(self, block=BasicBlock, keep_prob=1.0, avg_pool=True, drop_rate=0.1, dropblock_size=5):
        self.inplanes = 1
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        self.layer4 = self._make_layer(block, 64, stride=2, drop_rate=drop_rate, drop_block=True, block_size=dropblock_size)
        if avg_pool:
            self.avgpool = nn.AvgPool2d(5, stride=1)
        self.keep_prob = keep_prob
        self.keep_avg_pool = avg_pool
        self.dropout = nn.Dropout(p=1 - self.keep_prob, inplace=False)
        self.drop_rate = drop_rate
        self.pool = nn.AdaptiveAvgPool2d((4, 2))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride=1, drop_rate=0.0, drop_block=False, block_size=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, drop_rate, drop_block, block_size))
        self.inplanes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        (num_samples,seq_len,mel_bins) = x.shape
        x = x.view(-1,1,seq_len,mel_bins)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        #x = self.layer4(x)
        
        x = self.pool(x)
        
        x = x.view(x.size(0), -1)
        
        return x

class AvesClassifier(nn.Module):
    """ Uses AVES Hubert to embed sounds and classify """
    def __init__(self, config_path, model_path, trainable, embedding_dim=768, sr=48000):
        super().__init__()
        # reference: https://pytorch.org/audio/stable/_modules/torchaudio/models/wav2vec2/utils/import_fairseq.html
        self.config = self.load_config(config_path)
        self.model = wav2vec2_model(**self.config, aux_num_out=None)
        self.model.load_state_dict(torch.load(model_path))
        # Freeze the AVES network
        self.trainable = trainable
        self.freeze_embedding_weights()
        # We will only train the classifier head
        # self.classifier_head = nn.Linear(in_features=embedding_dim, out_features=n_classes)
        self.audio_sr = sr        

    def load_config(self, config_path):
        with open(config_path, 'r') as ff:
            obj = json.load(ff)
        return obj

    def forward(self, sig):
        """
        Input
          sig (Tensor): (batch, time)
        Returns
          mean_embedding (Tensor): (batch, output_dim)
          logits (Tensor): (batch, n_classes)
        """
        # extract_feature in the sorchaudio version will output all 12 layers' output, -1 to select the final one
        out = self.model.extract_features(sig)[0][-1]
        mean_embedding = out.mean(dim=1) #over time
        # logits = self.model.classifier_head(mean_embedding)
        # return mean_embedding, logits
        return mean_embedding
    
    # Code to use while initially setting up the model
    def freeze_embedding_weights(self):
        """ Freeze weights in AVES embeddings for classification """
        # The convolutional layers should never be trainable
        self.model.feature_extractor.requires_grad_(False)
        self.model.feature_extractor.eval()
        # The transformers are optionally trainable
        for param in self.model.encoder.parameters():
            param.requires_grad = self.trainable
        if not self.trainable:
            # We also set layers without params (like dropout) to eval mode, so they do not change
            self.model.encoder.eval()


    # Code to use during training loop, to switch between eval and train mode
    def set_eval_aves(self):
        """ Set AVES-based classifier to eval mode. Takes into account whether we are training transformers """
        # self.classifier_head.eval()
        self.model.encoder.eval()

    def set_train_aves(self):
        """ Set AVES-based classifier to train mode. Takes into account whether we are training transformers """
        # Always train the classifier head
        # self.classifier_head.train()
        # Optionally train the transformer of the model
        if self.trainable:
            self.model.encoder.train()