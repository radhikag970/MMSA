import logging
from tqdm import tqdm
import torch
import torch.nn as nn
from utils import MetricsTop, dict_to_str
from transformers import BertTokenizer

logger = logging.getLogger('MMSA')

class CENET:
    def __init__(self, args):
        self.args = args
        # make sure defaults exist
        self.args.setdefault('max_grad_norm', 2)
        self.args.setdefault('adam_epsilon', 1e-8)
        self.args.setdefault('weight_decay', 0.0)
        self.args.setdefault('KeyEval', 'Loss')
        self.metrics = MetricsTop(args['train_mode']).getMetics(args['dataset_name'])
        self.tokenizer = BertTokenizer.from_pretrained(args['pretrained'])
        self.criterion = nn.L1Loss()

    def do_train(self, model, dataloader, return_epoch_results=False):
        # get all parameters
        param_optimizer = list(model.named_parameters())
        no_decay = ["bias", "LayerNorm.weight"]
        ce_only = model.bert.encoder.CE
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in param_optimizer
                    if not any(nd in n for nd in no_decay) and not n.startswith('bert.encoder.CE')
                ],
                "weight_decay": self.args['weight_decay'],
            },
            {
                "params": ce_only.parameters(),
                "lr": self.args['learning_rate'],
                "weight_decay": self.args['weight_decay'],
            },
            {
                "params": [
                    p for n, p in param_optimizer
                    if any(nd in n for nd in no_decay) and not n.startswith('bert.encoder.CE')
                ],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.Adam(
            optimizer_grouped_parameters,
            lr=self.args['learning_rate'],
            eps=self.args['adam_epsilon']
        )

        best_valid = float('inf') if self.args['KeyEval']=='Loss' else -float('inf')
        best_epoch, epoch = 0, 0

        while True:
            epoch += 1
            model.train()
            total_loss = 0.0
            all_pred, all_true = [], []

            for batch in tqdm(dataloader['train'], desc=f"Epoch {epoch}"):
                vision = batch['vision'].to(self.args['device'])
                audio  = batch['audio'].to(self.args['device'])
                text   = batch['text'].to(self.args['device'])
                labels = batch['labels']['M'].to(self.args['device']).view(-1,1)

                optimizer.zero_grad()
                logits = model(text, audio, vision)[0]
                loss = self.criterion(logits, labels)
                loss.backward()

                # gradient clipping
                if self.args['max_grad_norm']>0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), self.args['max_grad_norm'])
                optimizer.step()

                total_loss += loss.item()
                all_pred.append(logits.detach().cpu())
                all_true.append(labels.detach().cpu())

            avg_train_loss = total_loss / len(dataloader['train'])
            train_pred = torch.cat(all_pred); train_true = torch.cat(all_true)
            train_metrics = self.metrics(train_pred, train_true)
            logger.info(f"Epoch {epoch} TRAIN loss={avg_train_loss:.4f} metrics={dict_to_str(train_metrics)}")

            # validation
            val_metrics = self.do_test(model, dataloader['valid'], mode="VAL")
            score = val_metrics[self.args['KeyEval']]
            is_better = (score < best_valid) if self.args['KeyEval']=='Loss' else (score > best_valid)
            if is_better:
                best_valid, best_epoch = score, epoch
                # save best
                torch.save(model.cpu().state_dict(), self.args['model_save_path'])
                model.to(self.args['device'])

            # early stop
            if epoch - best_epoch >= self.args['early_stop']:
                return

    def do_test(self, model, dataloader, mode="VAL"):
        model.eval()
        total_loss = 0.0
        all_pred, all_true = [], []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc=mode):
                vision = batch['vision'].to(self.args['device'])
                audio  = batch['audio'].to(self.args['device'])
                text   = batch['text'].to(self.args['device'])
                labels = batch['labels']['M'].to(self.args['device']).view(-1,1)

                logits = model(text, audio, vision)[0]
                loss = self.criterion(logits, labels)
                total_loss += loss.item()

                all_pred.append(logits.cpu())
                all_true.append(labels.cpu())

        avg_loss = total_loss / len(dataloader)
        preds = torch.cat(all_pred); trues = torch.cat(all_true)
        metrics = self.metrics(preds, trues)
        metrics['Loss'] = round(avg_loss, 4)
        logger.info(f"{mode} loss={metrics['Loss']} metrics={dict_to_str(metrics)}")
        return metrics
