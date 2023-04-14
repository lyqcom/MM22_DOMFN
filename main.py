import sys
sys.path.append('..')
from datasets.mosi_dataset import MosiDataset
from models.encoder import DOMFN
from engines.mosi_trainer import *
from config.mosi_config import *
import pickle as pkl
import numpy as np
from mindspore.dataset import GeneratorDataset

def setup_seed(seed):
    ms.set_seed(seed)


def main(config):
    data_path = config.data_dir
    data = pkl.load(open(data_path, 'rb'))
    train_data = data['train']
    valid_data = data['valid']
    test_data = data['test']
    
    train_dataset = MosiDataset(**train_data)
    valid_dataset = MosiDataset(**valid_data)
    test_dataset = MosiDataset(**test_data)
    model = DOMFN(config.text_dim, config.vision_dim, config.audio_dim,
                 config.feature_dim, config.num_label, config.fusion)
    trainer = DomfnTrainer(config, model)
    model_path = config.model_path + 'checkpoint_' + config.version + '.pt'
    col_names = ['vision_embeds', 'text_embeds', 'audio_embeds', 'labels']

    if config.is_pretrain:
        train_loader = GeneratorDataset(train_dataset, column_names=col_names, shuffle=True).batch(batch_size=config.pre_batch)
        for epoch in range(config.pre_epoch):
            text_loss, vision_loss, audio_loss = trainer.pre_train(train_loader)

    train_loader = GeneratorDataset(train_dataset, column_names=col_names, shuffle=True).batch(batch_size=config.batch)
    valid_loader = GeneratorDataset(valid_dataset, column_names=col_names, shuffle=True).batch(batch_size=config.batch)
    test_loader = GeneratorDataset(test_dataset, column_names=col_names, shuffle=True).batch(batch_size=config.batch)
    best_mae = 10e5
    for epoch in range(config.epoch):
        train_loss, train_mae = trainer.train(train_loader)
        evaluate_loss, evaluate_mae, _, _, _ = trainer.evaluate(trainer.model, valid_loader)
        if evaluate_mae > best_mae:
            best_mae = evaluate_mae
            trainer.save(model_path)

    best_model = ms.load_checkpoint(model_path)
    best_results = trainer.test(best_model, test_loader)
    print('best results:', best_results)


if __name__ == '__main__':
    config = get_args()
    setup_seed(config.seed)
    main(config)








