from tqdm import tqdm
import os
import mindspore as ms
import mindspore.nn as nn
import mindspore.ops as ops
from scipy import stats
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, mean_absolute_error
# from torch.utils.data import DataLoader
from mindspore import ReduceLROnPlateau
from mindspore.common import ParameterTuple
from mindspore.ops.composite import GradOperation

TEXT_IDX = 0
VISION_IDX = 1
AUDIO_IDX = 2
LABEL_IDX = 3


class MyLoss(nn.LossBase):
    def __init__(self, target_loss = None, reduction='mean'):
        super().__init__(reduction)
        self.target_loss = target_loss

    def construct(self, outdict, label = None):
        return outdict[self.target_loss]


class MyWithLossCell(nn.Cell):
   def __init__(self, backbone, loss_fn):
       super(MyWithLossCell, self).__init__(auto_prefix=False)
       self._backbone = backbone
       self._loss_fn = loss_fn

   def construct(self):
       out = self._backbone()
       return self._loss_fn(out)

class DomfnTrainer(object):
    def __init__(self, config, model):
        self.config = config

        self.model = model

        #10.22
        if config.use_cuda:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target = 'GPU')
        else:
            ms.set_context(mode=ms.PYNATIVE_MODE, device_target = 'CPU')

        self.text_model = self.model.text_encoder
        self.vision_model = self.model.vision_encoder
        self.audio_model = self.model.audio_encoder
        self.multi_model = self.model.multi_encoder

        self.text_optim  = nn.Adam( params=list(self.text_model.trainable_params()),
                                    learning_rate=self.config.pre_lr,
                                    weight_decay=self.config.weight_decay_text)
        self.vision_optim= nn.Adam( params=list(self.vision_model.trainable_params()),
                                    learning_rate=self.config.text_ft_lr,
                                    weight_decay=self.config.weight_decay_vision)
        self.audio_optim = nn.Adam( params=list(self.audio_model.trainable_params()),
                                    learning_rate=self.config.pre_lr,
                                    weight_decay=self.config.weight_decay_audio)
        self.multi_optim = nn.Adam( params=list(self.multi_model.trainable_params()),
                                    learning_rate=self.config.multi_lr,
                                    weight_decay=self.config.weight_decay_multi)

        #10.22
        self.text_pretrainer = nn.WithGradCell(self.text_model, MyLoss('celoss'))
        self.vision_pretrainer = nn.WithGradCell(self.vision_model, MyLoss('celoss'))
        self.audio_pretrainer = nn.WithGradCell(self.audio_model, MyLoss('celoss'))

        self.text_trainer = nn.TrainOneStepCell(MyWithLossCell(self.text_model, MyLoss('loss')), self.text_optim)
        self.vision_trainer = nn.TrainOneStepCell(MyWithLossCell(self.vision_model, MyLoss('loss')), self.vision_optim)
        self.audio_trainer = nn.TrainOneStepCell(MyWithLossCell(self.audio_model, MyLoss('loss')), self.audio_optim)

        self.multi_trainer = nn.WithGradCell(self.model, MyLoss('multi_loss'))


    def train(self, train_loader):
        self.model.set_train()
        # self.model.set_grad()
        train_tqdm = tqdm(train_loader)
        all_out = []
        all_label = []
        all_loss = []
        if self.config.is_pretrain:
            self.text_optim.learning_rate.set_data(self.config.text_ft_lr) 
            self.vision_optim.learning_rate.set_data(self.config.vision_ft_lr)
            self.audio_optim.learning_rate.set_data(self.config.audio_ft_lr)

        multi_cnt = 0
        uni_cnt = 0
        for batch in train_tqdm:
            labels = batch[LABEL_IDX]
            output = self.model([*batch, self.config.eta])

            text_p = output['text_penalty'] / self.config.tau
            vision_p = output['vision_penalty'] / self.config.tau
            audio_p = output['audio_penalty'] / self.config.tau

            _ = None
            if text_p < self.config.gamma and vision_p < self.config.gamma and audio_p < self.config.gamma:

                multi_cnt += 1
                loss = output['multi_loss']
                all_grads = self.multi_trainer([*batch, self.config.eta], _)
                text_grads = all_grads[:8]
                vision_grads = all_grads[8:16]
                audio_grads = all_grads[16:24]
                multi_grads = all_grads[24:28]
                text_success = self.text_optim(text_grads)
                vision_success = self.vision_optim(vision_grads)
                audio_success = self.audio_optim(audio_grads)
                multi_success = self.multi_optim(multi_grads)

                out = output['multi_logit']
            else:
                
                loss1 = output['text_loss'] + output['vision_loss'] + output['audio_loss']
                uni_cnt += 1

                loss = self.text_trainer() + self.vision_trainer() + self.audio_trainer()
                text_pred, text_conf = ops.ArgMaxWithValue(axis=1)(output['text_logit'])
                vision_pred, vision_conf = ops.ArgMaxWithValue(axis=1)(output['vision_logit'])
                audio_pred, audio_conf = ops.ArgMaxWithValue(axis=1)(output['audio_logit'])

                if text_conf > vision_conf and text_conf > audio_conf:
                    out = output['text_logit']
                elif vision_conf > text_conf and vision_conf > audio_conf:
                    out = output['vision_logit']
                elif audio_conf > text_conf and audio_conf > vision_conf:
                    out = output['audio_logit']

                if len(out.shape) == 1:
                    out = ops.expand_dims(out, axis=0)

            
            all_out += out.asnumpy().tolist()
            all_label += labels.asnumpy().tolist()
            all_loss.append(loss.asnumpy().item(0))
            train_tqdm.set_description('Loss: {}, text_p: {}, vision_p: {}, audio_p: {}, muiti: {}, uni: {}'.format(
                np.mean(all_loss), text_p.item(0), vision_p.item(0), audio_p.item(0), multi_cnt, uni_cnt))
        labels = np.array(all_label).reshape(-1)
        one_hot_targets = np.eye(self.config.num_label)[labels]
        mae = mean_absolute_error(one_hot_targets, all_out)
        return np.mean(all_loss), mae

    def pre_train(self, train_loader):
        self.model.set_train()
        
        train_tqdm = tqdm(train_loader)
        all_text_loss = []
        all_vision_loss = []
        all_audio_loss = []
        for batch in train_tqdm:
            _ = None
            text_grads = self.text_pretrainer([batch[TEXT_IDX], batch[LABEL_IDX]], _)
            vision_grads = self.vision_pretrainer([batch[VISION_IDX], batch[LABEL_IDX]], _)
            audio_grads = self.audio_pretrainer([batch[AUDIO_IDX], batch[LABEL_IDX]], _)
            text_success = self.text_optim(text_grads)
            vision_success = self.vision_optim(vision_grads)
            audio_success = self.audio_optim(audio_grads)

            all_text_loss.append(self.text_model.celoss.asnumpy().item(0))
            all_vision_loss.append(self.vision_model.celoss.asnumpy().item(0))
            all_audio_loss.append(self.audio_model.celoss.asnumpy().item(0))

        return np.mean(all_text_loss), np.mean(all_vision_loss), np.mean(all_audio_loss)

    def evaluate(self, model, valid_loader):
        model.set_train(False)
        # model.set_grad(False)
        all_out = []
        all_label = []
        all_loss = []
        for batch in valid_loader:
            labels = batch[LABEL_IDX]
            output = model([*batch, self.config.eta])
            loss =  output['multi_loss']
            text_p = output['text_penalty'] / self.config.tau
            vision_p = output['vision_penalty'] / self.config.tau
            audio_p = output['audio_penalty'] / self.config.tau
            _, text_conf = ops.ArgMaxWithValue(axis=1)(output['text_logit'])
            _, vision_conf = ops.ArgMaxWithValue(axis=1)(output['vision_logit'])
            _, audio_conf = ops.ArgMaxWithValue(axis=1)(output['audio_logit'])

            if text_p < self.config.gamma and vision_p < self.config.gamma and audio_p < self.config.gamma:
                out = output['multi_logit']
            elif text_conf > vision_conf and text_conf > audio_conf:
                out = output['text_logit']
            elif vision_conf > text_conf and vision_conf > audio_conf:
                out = output['vision_logit']
            elif audio_conf > vision_conf and audio_conf > vision_conf:
                out = output['audio_logit']
            all_out += out.asnumpy().tolist()
            all_label += labels.asnumpy().tolist()
            all_loss.append(loss.asnumpy().item(0))
        labels = np.array(all_label).reshape(-1)
        one_hot_targets = np.eye(self.config.num_label)[labels]
        mae = mean_absolute_error(one_hot_targets, all_out)
        return np.mean(all_loss), mae, all_out, all_label, one_hot_targets

    def test(self, model, test_loader, choose_threshold=0.5):

        loss, mae, all_out, all_label, one_hot_targets = self.evaluate(model, test_loader)
        corr = np.mean(np.corrcoef(all_out, one_hot_targets))

        predict = np.argmax(all_out, axis=1)
        acc = accuracy_score(all_label, predict)
        precision = precision_score(all_label, predict, average='macro')
        recall = recall_score(all_label, predict, average='macro')
        f1 = f1_score(all_label, predict, average='macro')
        return {'mae': mae,
                'corr': corr,
                'acc': acc,
                'precision': precision,
                'recall': recall,
                'f1': f1}

    def save(self, model_path):
        ms.save_checkpoint(self.model, model_path)
