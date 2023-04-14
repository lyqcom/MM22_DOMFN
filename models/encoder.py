import mindspore
import mindspore.nn as nn
import mindspore.ops as ops


# Fully Convolutional Network
class Fcn(nn.Cell):
    def __init__(self,
                 input_dim,
                 feature_dim,
                 num_labels,
                ):
        '''
        :param input_dim:
        :param feature_dim:
        :param num_labels:
        '''
        super(Fcn, self).__init__()

        self.input_dim = input_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.average_logit = None
        self.logit = None

        self.feature_layer = nn.SequentialCell(
            nn.Dense(in_channels=self.input_dim, 
                     out_channels=512, 
                     activation = nn.ReLU()),
            nn.Dense(in_channels=512, 
                     out_channels=256, 
                     activation = nn.ReLU()),
            nn.Dense(in_channels=256, 
                     out_channels=self.feature_dim, 
                     activation = nn.ReLU())
        )
        self.prediction_layer = nn.Dense(
                                in_channels=self.feature_dim, 
                                out_channels=self.num_labels, 
                                activation = nn.Sigmoid()
                                )
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        
    def set_averLogit(self, average_logit):
        self.average_logit = average_logit
    
    def set_eta(self, eta = None):
        self.eta = eta

    def construct(self, batch = None):
        if(batch is not None and len(batch) == 2):
            feature = self.feature_layer(batch[0])
            logit = self.prediction_layer(feature)
            celoss = self.criterion(logit, batch[1])
            self.celoss = celoss
            self.logit = logit
            return {
                'feature': feature,
                'logit': logit,
                'celoss': celoss,
            }
        else:
            _, penalty = ops.ArgMaxWithValue(axis=1)((self.logit-self.average_logit)**2)
            loss = 0.5 * self.celoss * self.celoss - self.eta * penalty
            return {
                'penalty': penalty,
                'loss': loss,
            }



class DOMFN(nn.Cell):
    def __init__(self,
                 text_hidden_dim=None,
                 vision_hidden_dim=None,
                 audio_hidden_dim=None,
                 feature_dim=None,
                 num_labels=None,
                 fusion='concat'):
        '''
        :param text_hidden_dim:
        :param vision_hidden_dim:
        :param audio_hidden_dim:
        :param feature_dim:
        :param num_labels:
        :param fusion:
        '''

        super(DOMFN, self).__init__()
        self.text_hidden_dim = text_hidden_dim
        self.vision_hidden_dim = vision_hidden_dim
        self.audio_hidden_dim = audio_hidden_dim
        self.feature_dim = feature_dim
        self.num_labels = num_labels
        self.fusion = fusion
        
        self.text_encoder= Fcn(self.text_hidden_dim, self.feature_dim, self.num_labels)
        self.vision_encoder = Fcn(self.vision_hidden_dim, self.feature_dim, self.num_labels)
        self.audio_encoder = Fcn(self.audio_hidden_dim, self.feature_dim, self.num_labels)

        self.multi_encoder = nn.SequentialCell(
            nn.Dense(in_channels=(3 if self.fusion == 'concat' else 1)*self.feature_dim, 
                     out_channels=64, 
                     activation = nn.ReLU()),
            nn.Dense(in_channels=64, 
                     out_channels=self.num_labels, 
                     activation = nn.Sigmoid())
        )
        self.criterion = nn.SoftmaxCrossEntropyWithLogits(sparse=True, reduction='mean')
        

    def construct(self,
                batch):
        text_embeds = batch[0]
        vision_embeds = batch[1]
        audio_embeds = batch[2]
        labels = batch[3]
        eta = batch[4]
        self.text_encoder.set_eta(eta)
        self.vision_encoder.set_eta(eta)
        self.audio_encoder.set_eta(eta)
        text_out = self.text_encoder([text_embeds, labels])
        vision_out = self.vision_encoder([vision_embeds, labels])
        audio_out = self.audio_encoder([audio_embeds, labels])
        text_feat = text_out['feature']
        vision_feat = vision_out['feature']
        audio_feat = audio_out['feature']
        text_logit = text_out['logit']
        vision_logit = vision_out['logit']
        audio_logit = audio_out['logit']
        text_celoss = text_out['celoss']
        vision_celoss = vision_out['celoss']
        audio_celoss = audio_out['celoss']
        multi_logit = None

        if self.fusion == 'concat':
            multi_feat = ops.concat([text_feat, vision_feat, audio_feat], axis=1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'mean':
            multi_feat = ops.ReduceMean()(ops.stack([text_feat, vision_feat, audio_feat], axis=1), 1)
            multi_logit = self.multi_encoder(multi_feat)
        elif self.fusion == 'max':
            _, multi_feat = ops.ArgMaxWithValue(axis=1)(ops.stack([text_feat, vision_feat, audio_feat], axis=1))
            multi_logit = self.multi_encoder(multi_feat)

        multi_celoss = self.criterion(multi_logit, labels)
        average_logit = ops.ReduceMean()(ops.stack([text_logit, vision_logit, audio_logit], axis=1), 1)
        self.text_encoder.set_averLogit(average_logit)
        self.vision_encoder.set_averLogit(average_logit)
        self.audio_encoder.set_averLogit(average_logit)
        text_out = self.text_encoder([])
        vision_out = self.vision_encoder([])
        audio_out = self.audio_encoder([])
        text_penalty = text_out['penalty']
        vision_penalty = vision_out['penalty']
        audio_penalty = audio_out['penalty']
        text_loss = text_out['loss']
        vision_loss = vision_out['loss']
        audio_loss = audio_out['loss']

        multi_loss = text_loss + vision_loss + audio_loss + multi_celoss

        
        return {
            'text_logit': text_logit,
            'vision_logit': vision_logit,
            'audio_logit': audio_logit,
            'multi_logit': multi_logit,

            'text_celoss': text_celoss,
            'vision_celoss': vision_celoss,
            'audio_celoss': audio_celoss,

            'text_penalty': text_penalty,
            'vision_penalty': vision_penalty,
            'audio_penalty': audio_penalty,

            'text_loss': text_loss,
            'vision_loss': vision_loss,
            'audio_loss': audio_loss,
            'multi_loss': multi_loss
        }
