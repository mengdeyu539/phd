import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.multiprocessing as mp
from transformers import T5ForConditionalGeneration, T5Tokenizer, AutoModel, AutoTokenizer
import numpy as np
from typing import Dict, List, Tuple, Optional
import os
import json
from tqdm import tqdm
import random
from sklearn.metrics import accuracy_score
import logging
from datetime import datetime, timedelta
import argparse
# ⭐ 新增导入
import wandb

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
except ImportError:
    print("请安装nltk: pip install nltk")
    import sys

    sys.exit(1)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ===================== 多卡训练工具函数 =====================

def setup_distributed(rank, world_size):
    """初始化分布式训练环境"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    # ⭐ 增加NCCL超时时间（默认10分钟，增加到30分钟）
    os.environ['NCCL_TIMEOUT'] = '1800'
    os.environ['NCCL_BLOCKING_WAIT'] = '1'  # 让NCCL错误更容易调试

    # ⭐ 使用timedelta设置超时
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=1800)  # 30分钟超时
    )
    torch.cuda.set_device(rank)

    if rank == 0:
        logger.info(f"✅ 分布式环境初始化完成 (超时时间: 1800秒)")


def cleanup_distributed():
    """清理分布式训练环境"""
    dist.destroy_process_group()


def is_main_process(rank):
    """判断是否为主进程"""
    return rank == 0


# ===================== 数据加载 =====================

def load_text_data(file_path):
    texts = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                texts.append(line)
    return texts


def load_style_datasets(data_dir):
    datasets = {}

    train_human = load_text_data(os.path.join(data_dir, 'train.human'))
    train_ai = load_text_data(os.path.join(data_dir, 'train.ais'))
    datasets['train'] = {
        'texts': train_human + train_ai,
        'labels': [0] * len(train_human) + [1] * len(train_ai),
        'human_texts': train_human,
        'ai_texts': train_ai
    }

    dev_human = load_text_data(os.path.join(data_dir, 'dev.human'))
    dev_ai = load_text_data(os.path.join(data_dir, 'dev.ais'))
    datasets['dev'] = {
        'texts': dev_human + dev_ai,
        'labels': [0] * len(dev_human) + [1] * len(dev_ai),
        'human_texts': dev_human,
        'ai_texts': dev_ai
    }

    test_human = load_text_data(os.path.join(data_dir, 'test.human'))
    test_ai = load_text_data(os.path.join(data_dir, 'test.ais'))
    datasets['test'] = {
        'texts': test_human + test_ai,
        'labels': [0] * len(test_human) + [1] * len(test_ai)
    }

    logger.info("数据集统计:")
    for split, data in datasets.items():
        h_count = data['labels'].count(0)
        a_count = data['labels'].count(1)
        logger.info(f"{split}: {len(data['texts'])} (Human: {h_count}, AI: {a_count})")

    return datasets


class StyleTransferDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=384):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        if label == 0:
            input_text = f"transfer to ai style: {text}"
            target_style = 1
        else:
            input_text = f"transfer to human style: {text}"
            target_style = 0

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        targets = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        original_length = (targets['input_ids'][0] != self.tokenizer.pad_token_id).sum().item()

        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'target_ids': targets['input_ids'].squeeze(),
            'target_attention_mask': targets['attention_mask'].squeeze(),
            'style_label': torch.tensor(label, dtype=torch.long),
            'target_style': torch.tensor(target_style, dtype=torch.long),
            'original_length': torch.tensor(original_length, dtype=torch.long),
            'text': text
        }


# ===================== 判别器 =====================

class StyleDiscriminator(nn.Module):
    def __init__(self, hidden_dim=512, style_dim=256, dropout=0.3):
        super().__init__()

        self.embedding = nn.Embedding(32100, 256)
        self.lstm = nn.LSTM(256, hidden_dim, num_layers=2,
                            batch_first=True, bidirectional=True, dropout=dropout)

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, style_dim),
            nn.LayerNorm(style_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim, style_dim // 2),
            nn.LayerNorm(style_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(style_dim // 2, 2)
        )

    def forward(self, input_ids, attention_mask=None):
        embedded = self.embedding(input_ids)
        lstm_out, (h_n, c_n) = self.lstm(embedded)

        forward_hidden = h_n[-2, :, :]
        backward_hidden = h_n[-1, :, :]
        hidden = torch.cat([forward_hidden, backward_hidden], dim=1)

        logits = self.classifier(hidden)
        return logits


# ===================== 生成器 =====================

class T5StyleGenerator(nn.Module):
    def __init__(self, model_name='t5-small', device='cuda'):
        super().__init__()
        self.device = device
        self.t5 = T5ForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = T5Tokenizer.from_pretrained(model_name)

    def forward_with_teacher_forcing(self, input_ids, attention_mask, labels):
        outputs = self.t5(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        return outputs

    def generate_text(self, input_ids, attention_mask, target_length=None, max_length=384):
        if target_length is not None:
            min_len = max(10, int(target_length * 0.8))
            max_len = min(max_length, int(target_length * 1.2))
        else:
            min_len = 20
            max_len = max_length

        outputs = self.t5.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=max_len,
            min_length=min_len,
            num_beams=5,
            no_repeat_ngram_size=3,
            early_stopping=True,
            temperature=0.9,
            top_k=50,
            top_p=0.95,
            do_sample=True,
            repetition_penalty=1.2,
            length_penalty=1.2
        )
        return outputs

    def generate_with_style(self, text, target_style_id, preserve_length=True):
        self.eval()

        if target_style_id == 0:
            input_text = f"transfer to human style: {text}"
        else:
            input_text = f"transfer to ai style: {text}"

        inputs = self.tokenizer(
            input_text,
            return_tensors='pt',
            max_length=384,
            truncation=True
        ).to(self.device)

        target_length = None
        if preserve_length:
            original_tokens = self.tokenizer.encode(text, add_special_tokens=False)
            target_length = len(original_tokens)

        with torch.no_grad():
            outputs = self.generate_text(
                inputs['input_ids'],
                inputs['attention_mask'],
                target_length=target_length
            )

        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text


# ===================== 评估工具类 =====================

class EvaluationMetrics:
    def __init__(self, device='cuda'):
        self.device = device
        logger.info("加载BERT模型用于语义相似度计算...")
        try:
            self.semantic_model = AutoModel.from_pretrained('bert-base-uncased').to(device)
            self.semantic_tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            self.semantic_model.eval()
        except Exception as e:
            logger.warning(f"加载BERT失败: {e}，将使用简化的相似度计算")
            self.semantic_model = None
            self.semantic_tokenizer = None

        self.smoothing = SmoothingFunction()

    def get_bert_embedding(self, texts):
        if self.semantic_model is None:
            return None

        inputs = self.semantic_tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors='pt'
        ).to(self.device)

        with torch.no_grad():
            outputs = self.semantic_model(**inputs)
            embeddings = outputs.last_hidden_state[:, 0, :]

        return embeddings

    def compute_semantic_similarity(self, texts1, texts2):
        if self.semantic_model is None:
            similarities = []
            for t1, t2 in zip(texts1, texts2):
                words1 = set(t1.lower().split())
                words2 = set(t2.lower().split())
                if len(words1) == 0 or len(words2) == 0:
                    sim = 0.0
                else:
                    sim = len(words1 & words2) / len(words1 | words2)
                similarities.append(sim)
            return np.array(similarities)

        embeddings1 = self.get_bert_embedding(texts1)
        embeddings2 = self.get_bert_embedding(texts2)
        similarities = F.cosine_similarity(embeddings1, embeddings2, dim=1)

        return similarities.cpu().numpy()

    def compute_bleu(self, reference, hypothesis):
        reference_tokens = reference.lower().split()
        hypothesis_tokens = hypothesis.lower().split()

        try:
            score = sentence_bleu(
                [reference_tokens],
                hypothesis_tokens,
                smoothing_function=self.smoothing.method1
            )
        except:
            score = 0.0

        return score

    def compute_batch_bleu(self, references, hypotheses):
        scores = []
        for ref, hyp in zip(references, hypotheses):
            score = self.compute_bleu(ref, hyp)
            scores.append(score)
        return scores

    def evaluate_style_transfer(self, discriminator, tokenizer, texts, target_styles, device):
        discriminator.eval()

        inputs = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=384,
            return_tensors='pt'
        ).to(device)

        with torch.no_grad():
            logits = discriminator(inputs['input_ids'])
            predictions = logits.argmax(dim=1).cpu().numpy()

        target_styles = np.array(target_styles)
        success_rate = (predictions == target_styles).mean()

        return success_rate, predictions


# ===================== 改进的训练器（带WandB和多卡支持，优化AI→Human） =====================

class ImprovedAdversarialTrainer:
    def __init__(self, generator, discriminator, train_loader, dev_loader, config, dev_data=None, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size
        self.is_main = is_main_process(rank)

        # ⭐ 使用DDP包装模型
        self.generator = generator.to(rank)
        self.discriminator = discriminator.to(rank)

        if world_size > 1:
            self.generator = DDP(generator, device_ids=[rank], find_unused_parameters=True)
            self.discriminator = DDP(discriminator, device_ids=[rank])

        # ⭐ 创建便捷属性来访问实际模型
        if world_size > 1:
            self.generator_model = self.generator.module
            self.discriminator_model = self.discriminator.module
        else:
            self.generator_model = self.generator
            self.discriminator_model = self.discriminator

        self.train_loader = train_loader
        self.dev_loader = dev_loader
        self.config = config
        self.device = rank
        self.dev_data = dev_data

        # 评估工具（只在主进程创建）
        if self.is_main:
            self.evaluator = EvaluationMetrics(device=rank)

        # ⭐ 初始化WandB（只在主进程）
        if config.get('use_wandb', True) and self.is_main:
            wandb.init(
                project=config.get('wandb_project', 'style-transfer'),
                name=config.get('wandb_run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=config,
                tags=['unsupervised', 'style-transfer', 't5', f'{world_size}gpus', 'ai2human-focused']
            )
            # 监控模型
            wandb.watch(self.generator_model, log='all', log_freq=100)
            wandb.watch(self.discriminator_model, log='all', log_freq=100)

        # 优化器
        self.g_optimizer = torch.optim.AdamW(
            self.generator.parameters(),
            lr=config['g_learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.5, 0.999)
        )

        self.d_optimizer = torch.optim.AdamW(
            self.discriminator.parameters(),
            lr=config['d_learning_rate'],
            weight_decay=config['weight_decay'],
            betas=(0.5, 0.999)
        )

        # 学习率调度器
        self.g_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.g_optimizer, T_max=len(train_loader) * config['num_epochs']
        )
        self.d_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.d_optimizer, T_max=len(train_loader) * config['num_epochs']
        )

        # ⭐ 获取tokenizer
        self.tokenizer = self.generator_model.tokenizer

        self.ce_loss = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)

        # ⭐ 追踪最佳指标 - 重点关注AI→Human
        self.best_metrics = {
            'semantic_similarity': 0.0,
            'bleu_score': 0.0,
            'transfer_success_rate': 0.0,
            'ai_to_human_semantic_sim': 0.0,
            'ai_to_human_bleu': 0.0,
            'ai_to_human_success_rate': 0.0,
            'human_to_ai_success_rate': 0.0,
            'g_loss': float('inf')
        }
        self.patience_counter = 0

        # ⭐ Checkpoint目录
        if self.is_main:
            self.checkpoint_dir = os.path.join(config['output_dir'], 'checkpoints')
            os.makedirs(self.checkpoint_dir, exist_ok=True)

    def compute_length_loss(self, generated_ids, target_lengths):
        pad_token_id = self.tokenizer.pad_token_id
        gen_lengths = (generated_ids != pad_token_id).sum(dim=1).float()
        target_lengths = target_lengths.float()
        length_diff = torch.abs(gen_lengths - target_lengths)
        length_loss = length_diff.mean()
        return length_loss

    def compute_semantic_loss(self, original_texts, generated_texts):
        if not self.is_main:
            return torch.tensor(0.0, device=self.device)

        similarities = self.evaluator.compute_semantic_similarity(
            original_texts,
            generated_texts
        )
        similarities = torch.tensor(similarities, dtype=torch.float32).to(self.device)
        semantic_loss = (1.0 - similarities).mean()
        return semantic_loss

    def train_discriminator(self, real_human_texts, real_ai_texts, fake_human_texts, fake_ai_texts):
        self.discriminator.train()
        self.d_optimizer.zero_grad()

        real_human_inputs = self.tokenizer(
            real_human_texts, padding=True, truncation=True,
            max_length=384, return_tensors='pt'
        ).to(self.device)

        real_ai_inputs = self.tokenizer(
            real_ai_texts, padding=True, truncation=True,
            max_length=384, return_tensors='pt'
        ).to(self.device)

        fake_human_inputs = self.tokenizer(
            fake_human_texts, padding=True, truncation=True,
            max_length=384, return_tensors='pt'
        ).to(self.device)

        fake_ai_inputs = self.tokenizer(
            fake_ai_texts, padding=True, truncation=True,
            max_length=384, return_tensors='pt'
        ).to(self.device)

        batch_size = len(real_human_texts)

        real_human_logits = self.discriminator(real_human_inputs['input_ids'])
        real_human_labels = torch.zeros(batch_size, dtype=torch.long).to(self.device)
        loss_real_human = self.ce_loss(real_human_logits, real_human_labels)

        real_ai_logits = self.discriminator(real_ai_inputs['input_ids'])
        real_ai_labels = torch.ones(batch_size, dtype=torch.long).to(self.device)
        loss_real_ai = self.ce_loss(real_ai_logits, real_ai_labels)

        fake_human_logits = self.discriminator(fake_human_inputs['input_ids'])
        loss_fake_human = self.ce_loss(fake_human_logits, real_human_labels)

        fake_ai_logits = self.discriminator(fake_ai_inputs['input_ids'])
        loss_fake_ai = self.ce_loss(fake_ai_logits, real_ai_labels)

        d_loss = loss_real_human + loss_real_ai + loss_fake_human + loss_fake_ai

        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), 1.0)
        self.d_optimizer.step()
        self.d_scheduler.step()

        with torch.no_grad():
            real_human_acc = (real_human_logits.argmax(1) == real_human_labels).float().mean()
            real_ai_acc = (real_ai_logits.argmax(1) == real_ai_labels).float().mean()

        return d_loss.item(), real_human_acc.item(), real_ai_acc.item()

    def train_generator(self, batch):
        self.generator.train()
        self.g_optimizer.zero_grad()

        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        target_style = batch['target_style'].to(self.device)
        original_lengths = batch['original_length'].to(self.device)
        texts = batch['text']

        # ⭐ Teacher forcing
        outputs = self.generator_model.forward_with_teacher_forcing(
            input_ids, attention_mask, target_ids
        )
        reconstruction_loss = outputs.loss

        # 生成文本
        with torch.no_grad():
            generated_ids_list = []
            for i in range(len(texts)):
                single_input = input_ids[i:i + 1]
                single_mask = attention_mask[i:i + 1]
                target_len = original_lengths[i].item()

                gen_ids = self.generator_model.generate_text(
                    single_input, single_mask, target_length=target_len
                )
                generated_ids_list.append(gen_ids[0])

            max_gen_len = max(len(ids) for ids in generated_ids_list)
            generated_ids = torch.zeros(
                len(generated_ids_list), max_gen_len, dtype=torch.long
            ).to(self.device)

            for i, ids in enumerate(generated_ids_list):
                generated_ids[i, :len(ids)] = ids

        generated_texts = [
            self.tokenizer.decode(ids, skip_special_tokens=True)
            for ids in generated_ids
        ]

        # 损失计算
        length_loss = self.compute_length_loss(generated_ids, original_lengths)
        semantic_loss = self.compute_semantic_loss(list(texts), generated_texts)

        gen_inputs = self.tokenizer(
            generated_texts, padding=True, truncation=True,
            max_length=384, return_tensors='pt'
        ).to(self.device)

        fake_logits = self.discriminator(gen_inputs['input_ids'])
        adv_loss = self.ce_loss(fake_logits, target_style)

        # 总损失
        g_loss = (
                self.config['lambda_reconstruction'] * reconstruction_loss +
                self.config['lambda_adv'] * adv_loss +
                self.config['lambda_length'] * length_loss +
                self.config['lambda_semantic'] * semantic_loss
        )

        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.generator.parameters(), 1.0)
        self.g_optimizer.step()
        self.g_scheduler.step()

        return {
            'g_loss': g_loss.item(),
            'reconstruction_loss': reconstruction_loss.item(),
            'adv_loss': adv_loss.item(),
            'length_loss': length_loss.item(),
            'semantic_loss': semantic_loss.item()
        }

    def train_epoch(self, epoch):
        total_g_loss = 0
        total_d_loss = 0
        total_length_loss = 0
        total_semantic_loss = 0
        total_reconstruction_loss = 0
        total_adv_loss = 0
        total_d_acc_human = 0
        total_d_acc_ai = 0

        n_d_steps = 0

        # ⭐ 设置sampler的epoch（用于shuffle）
        if self.world_size > 1:
            self.train_loader.sampler.set_epoch(epoch)

        # ⭐ 只在主进程显示进度条
        if self.is_main:
            progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = self.train_loader

        for batch_idx, batch in enumerate(progress_bar):
            human_mask = batch['style_label'] == 0
            ai_mask = batch['style_label'] == 1

            if human_mask.sum() == 0 or ai_mask.sum() == 0:
                continue

            human_texts = [batch['text'][i] for i in range(len(batch['text'])) if human_mask[i]]
            ai_texts = [batch['text'][i] for i in range(len(batch['text'])) if ai_mask[i]]

            # 生成假样本
            self.generator.eval()
            with torch.no_grad():
                # ⭐ AI→Human: 生成更多样本（增加到12个，原来是8个）
                fake_human_texts = []
                for text in ai_texts[:min(len(ai_texts), 12)]:
                    fake_human = self.generator_model.generate_with_style(text, 0, preserve_length=True)
                    fake_human_texts.append(fake_human)

                # Human→AI: 保持原样本数量
                fake_ai_texts = []
                for text in human_texts[:min(len(human_texts), 8)]:
                    fake_ai = self.generator_model.generate_with_style(text, 1, preserve_length=True)
                    fake_ai_texts.append(fake_ai)

            # 训练判别器
            if batch_idx % self.config['d_train_interval'] == 0 and len(fake_human_texts) > 0:
                min_len = min(len(human_texts), len(ai_texts), len(fake_human_texts), len(fake_ai_texts))
                if min_len > 0:
                    d_loss, d_acc_h, d_acc_a = self.train_discriminator(
                        human_texts[:min_len], ai_texts[:min_len],
                        fake_human_texts[:min_len], fake_ai_texts[:min_len]
                    )
                    total_d_loss += d_loss
                    total_d_acc_human += d_acc_h
                    total_d_acc_ai += d_acc_a
                    n_d_steps += 1

            # 训练生成器
            g_losses = self.train_generator(batch)
            total_g_loss += g_losses['g_loss']
            total_length_loss += g_losses['length_loss']
            total_semantic_loss += g_losses['semantic_loss']
            total_reconstruction_loss += g_losses['reconstruction_loss']
            total_adv_loss += g_losses['adv_loss']

            # ⭐ 实时记录到WandB（只在主进程）
            if self.config.get('use_wandb', True) and self.is_main and batch_idx % 10 == 0:
                wandb.log({
                    'train/batch_g_loss': g_losses['g_loss'],
                    'train/batch_semantic_loss': g_losses['semantic_loss'],
                    'train/batch_length_loss': g_losses['length_loss'],
                    'train/batch_reconstruction_loss': g_losses['reconstruction_loss'],
                    'train/batch_adv_loss': g_losses['adv_loss'],
                    'train/g_lr': self.g_scheduler.get_last_lr()[0],
                    'train/d_lr': self.d_scheduler.get_last_lr()[0],
                })

            if self.is_main:
                progress_bar.set_postfix({
                    'G': f"{g_losses['g_loss']:.3f}",
                    'Sem': f"{g_losses['semantic_loss']:.3f}",
                    'Len': f"{g_losses['length_loss']:.3f}",
                    'D': f"{d_loss if 'd_loss' in locals() else 0:.3f}"
                })

        n_batches = len(self.train_loader)
        metrics = {
            'g_loss': total_g_loss / n_batches,
            'd_loss': total_d_loss / max(n_d_steps, 1),
            'd_acc_human': total_d_acc_human / max(n_d_steps, 1),
            'd_acc_ai': total_d_acc_ai / max(n_d_steps, 1),
            'length_loss': total_length_loss / n_batches,
            'semantic_loss': total_semantic_loss / n_batches,
            'reconstruction_loss': total_reconstruction_loss / n_batches,
            'adv_loss': total_adv_loss / n_batches
        }

        return metrics

    def evaluate_dev_set(self, epoch):
        """⭐ 完整评估验证集并保存原文和生成文本（只在主进程执行）"""
        if not self.is_main or self.dev_data is None:
            return {}

        logger.info(f"\n📊 评估Epoch {epoch}...")
        self.generator.eval()

        # ⭐ 创建保存目录
        save_dir = os.path.join(self.config['output_dir'], f'epoch_{epoch}_results')
        os.makedirs(save_dir, exist_ok=True)

        # ===== AI → Human =====
        ai_texts = self.dev_data.get('ai_texts', [])
        ai_originals = []
        ai_generated = []
        ai_results = []

        for i, text in enumerate(tqdm(ai_texts, desc="AI→Human", leave=False)):
            generated = self.generator_model.generate_with_style(text, 0, preserve_length=True)
            ai_originals.append(text)
            ai_generated.append(generated)

            semantic_sim = self.evaluator.compute_semantic_similarity([text], [generated])[0]
            bleu = self.evaluator.compute_bleu(text, generated)

            ai_results.append({
                'index': i,
                'original_style': 'AI',
                'target_style': 'Human',
                'original_text': text,
                'generated_text': generated,
                'original_word_count': len(text.split()),
                'generated_word_count': len(generated.split()),
                'length_ratio': len(generated.split()) / max(len(text.split()), 1),
                'semantic_similarity': float(semantic_sim),
                'bleu_score': float(bleu)
            })

        ai_semantic_sims = self.evaluator.compute_semantic_similarity(ai_originals, ai_generated)
        ai_bleu_scores = self.evaluator.compute_batch_bleu(ai_originals, ai_generated)
        ai_success_rate, ai_predictions = self.evaluator.evaluate_style_transfer(
            self.discriminator, self.tokenizer,
            ai_generated, [0] * len(ai_generated), self.device
        )

        for i, result in enumerate(ai_results):
            result['predicted_style'] = int(ai_predictions[i])
            result['transfer_success'] = bool(ai_predictions[i] == 0)

        with open(os.path.join(save_dir, 'ai_to_human.jsonl'), 'w', encoding='utf-8') as f:
            for item in ai_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # ===== Human → AI =====
        human_texts = self.dev_data.get('human_texts', [])
        human_originals = []
        human_generated = []
        human_results = []

        for i, text in enumerate(tqdm(human_texts, desc="Human→AI", leave=False)):
            generated = self.generator_model.generate_with_style(text, 1, preserve_length=True)
            human_originals.append(text)
            human_generated.append(generated)

            semantic_sim = self.evaluator.compute_semantic_similarity([text], [generated])[0]
            bleu = self.evaluator.compute_bleu(text, generated)

            human_results.append({
                'index': i,
                'original_style': 'Human',
                'target_style': 'AI',
                'original_text': text,
                'generated_text': generated,
                'original_word_count': len(text.split()),
                'generated_word_count': len(generated.split()),
                'length_ratio': len(generated.split()) / max(len(text.split()), 1),
                'semantic_similarity': float(semantic_sim),
                'bleu_score': float(bleu)
            })

        human_semantic_sims = self.evaluator.compute_semantic_similarity(human_originals, human_generated)
        human_bleu_scores = self.evaluator.compute_batch_bleu(human_originals, human_generated)
        human_success_rate, human_predictions = self.evaluator.evaluate_style_transfer(
            self.discriminator, self.tokenizer,
            human_generated, [1] * len(human_generated), self.device
        )

        for i, result in enumerate(human_results):
            result['predicted_style'] = int(human_predictions[i])
            result['transfer_success'] = bool(human_predictions[i] == 1)

        with open(os.path.join(save_dir, 'human_to_ai.jsonl'), 'w', encoding='utf-8') as f:
            for item in human_results:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

        # ===== 汇总评估指标 =====
        eval_metrics = {
            'semantic_similarity': float(np.mean(list(ai_semantic_sims) + list(human_semantic_sims))),
            'bleu_score': float(np.mean(ai_bleu_scores + human_bleu_scores)),
            'transfer_success_rate': float((ai_success_rate + human_success_rate) / 2),
            'ai_to_human_semantic_sim': float(np.mean(ai_semantic_sims)),
            'ai_to_human_bleu': float(np.mean(ai_bleu_scores)),
            'ai_to_human_success_rate': float(ai_success_rate),
            'human_to_ai_semantic_sim': float(np.mean(human_semantic_sims)),
            'human_to_ai_bleu': float(np.mean(human_bleu_scores)),
            'human_to_ai_success_rate': float(human_success_rate),
            'avg_length_ratio': float(np.mean([r['length_ratio'] for r in ai_results + human_results]))
        }

        stats = {
            'epoch': epoch,
            'total_samples': len(ai_results) + len(human_results),
            'ai_to_human_count': len(ai_results),
            'human_to_ai_count': len(human_results),
            **eval_metrics,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }

        with open(os.path.join(save_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)

        logger.info(f"💾 评估结果已保存: {save_dir}")
        logger.info(f"   - AI→Human: {len(ai_results)}个样本")
        logger.info(f"   - Human→AI: {len(human_results)}个样本")

        return eval_metrics

    def save_checkpoint(self, epoch, metrics, is_best=False):
        """⭐ 保存checkpoint（只在主进程执行）"""
        if not self.is_main:
            return

        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': self.generator_model.state_dict(),
            'discriminator_state_dict': self.discriminator_model.state_dict(),
            'g_optimizer_state_dict': self.g_optimizer.state_dict(),
            'd_optimizer_state_dict': self.d_optimizer.state_dict(),
            'g_scheduler_state_dict': self.g_scheduler.state_dict(),
            'd_scheduler_state_dict': self.d_scheduler.state_dict(),
            'metrics': metrics,
            'config': self.config
        }

        epoch_path = os.path.join(self.checkpoint_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, epoch_path)
        logger.info(f"💾 Checkpoint已保存: {epoch_path}")

        if is_best:
            best_path = os.path.join(self.checkpoint_dir, 'best_model.pt')
            torch.save(checkpoint, best_path)
            logger.info(f"🌟 最佳模型已保存: {best_path}")

            if self.config.get('use_wandb', True):
                wandb.save(best_path)

    def train(self):
        """⭐ 完整训练流程（修复NCCL超时问题）"""
        if self.is_main:
            logger.info("🚀 开始训练...")
            logger.info(f"📊 使用 {self.world_size} 个GPU")
            logger.info(f"🎯 训练目标：重点优化 AI → Human 风格转换")

        for epoch in range(1, self.config['num_epochs'] + 1):
            # 训练一个epoch
            train_metrics = self.train_epoch(epoch)

            # ⭐ 评估验证集（只在主进程）
            eval_metrics = {}
            if self.is_main:
                eval_metrics = self.evaluate_dev_set(epoch)

            # ⭐ 广播评估指标到所有进程（广播操作自带同步）
            if self.world_size > 1:
                if self.is_main:
                    metrics_list = [
                        eval_metrics.get('semantic_similarity', 0),
                        eval_metrics.get('bleu_score', 0),
                        eval_metrics.get('transfer_success_rate', 0),
                        eval_metrics.get('ai_to_human_semantic_sim', 0),
                        eval_metrics.get('ai_to_human_bleu', 0),
                        eval_metrics.get('ai_to_human_success_rate', 0),
                        eval_metrics.get('human_to_ai_success_rate', 0),
                    ]
                else:
                    metrics_list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

                metrics_tensor = torch.tensor(metrics_list, device=self.device)
                dist.broadcast(metrics_tensor, src=0)  # 广播操作自动同步所有进程

                if not self.is_main:
                    eval_metrics = {
                        'semantic_similarity': float(metrics_tensor[0]),
                        'bleu_score': float(metrics_tensor[1]),
                        'transfer_success_rate': float(metrics_tensor[2]),
                        'ai_to_human_semantic_sim': float(metrics_tensor[3]),
                        'ai_to_human_bleu': float(metrics_tensor[4]),
                        'ai_to_human_success_rate': float(metrics_tensor[5]),
                        'human_to_ai_success_rate': float(metrics_tensor[6]),
                    }

            all_metrics = {**train_metrics, **eval_metrics}

            # ⭐ 记录到WandB（只在主进程）
            if self.config.get('use_wandb', True) and self.is_main:
                wandb.log({
                    'epoch': epoch,
                    'train/g_loss': train_metrics['g_loss'],
                    'train/d_loss': train_metrics['d_loss'],
                    'train/semantic_loss': train_metrics['semantic_loss'],
                    'train/length_loss': train_metrics['length_loss'],
                    'train/reconstruction_loss': train_metrics['reconstruction_loss'],
                    'train/adv_loss': train_metrics['adv_loss'],
                    'train/d_acc_human': train_metrics['d_acc_human'],
                    'train/d_acc_ai': train_metrics['d_acc_ai'],
                    'eval/semantic_similarity': eval_metrics.get('semantic_similarity', 0),
                    'eval/bleu_score': eval_metrics.get('bleu_score', 0),
                    'eval/transfer_success_rate': eval_metrics.get('transfer_success_rate', 0),
                    'eval/avg_length_ratio': eval_metrics.get('avg_length_ratio', 0),
                    'eval/ai_to_human_semantic_sim': eval_metrics.get('ai_to_human_semantic_sim', 0),
                    'eval/ai_to_human_bleu': eval_metrics.get('ai_to_human_bleu', 0),
                    'eval/ai_to_human_success_rate': eval_metrics.get('ai_to_human_success_rate', 0),
                    'eval/human_to_ai_semantic_sim': eval_metrics.get('human_to_ai_semantic_sim', 0),
                    'eval/human_to_ai_bleu': eval_metrics.get('human_to_ai_bleu', 0),
                    'eval/human_to_ai_success_rate': eval_metrics.get('human_to_ai_success_rate', 0),
                })

            # 打印结果（只在主进程）
            if self.is_main:
                logger.info(f"\n{'=' * 80}")
                logger.info(f"📈 Epoch {epoch}/{self.config['num_epochs']}")
                logger.info(f"{'=' * 80}")
                logger.info(f"训练 - G Loss: {train_metrics['g_loss']:.4f} | D Loss: {train_metrics['d_loss']:.4f}")
                logger.info(
                    f"     - Semantic: {train_metrics['semantic_loss']:.4f} | Length: {train_metrics['length_loss']:.4f}")
                logger.info(
                    f"     - Reconstruction: {train_metrics['reconstruction_loss']:.4f} | Adv: {train_metrics['adv_loss']:.4f}")
                logger.info(f"评估 - 整体语义相似度: {eval_metrics.get('semantic_similarity', 0):.3f}")
                logger.info(f"     - 整体BLEU: {eval_metrics.get('bleu_score', 0):.3f}")
                logger.info(f"     - 整体转换成功率: {eval_metrics.get('transfer_success_rate', 0):.1%}")
                logger.info(f"     ─────────────────────────────────────")
                logger.info(f"     ⭐ AI→Human 语义相似度: {eval_metrics.get('ai_to_human_semantic_sim', 0):.3f}")
                logger.info(f"     ⭐ AI→Human BLEU: {eval_metrics.get('ai_to_human_bleu', 0):.3f}")
                logger.info(f"     ⭐ AI→Human 成功率: {eval_metrics.get('ai_to_human_success_rate', 0):.1%}")
                logger.info(f"     ─────────────────────────────────────")
                logger.info(f"     - Human→AI 成功率: {eval_metrics.get('human_to_ai_success_rate', 0):.1%}")
                logger.info(f"     - 平均长度比例: {eval_metrics.get('avg_length_ratio', 0):.1%}")
                logger.info(f"{'=' * 80}\n")

            # ⭐ 判断是否为最佳模型
            is_best = False
            current_score = (
                    eval_metrics.get('ai_to_human_semantic_sim', 0) * 0.3 +
                    eval_metrics.get('ai_to_human_bleu', 0) * 0.2 +
                    eval_metrics.get('ai_to_human_success_rate', 0) * 0.4 +
                    eval_metrics.get('human_to_ai_success_rate', 0) * 0.1
            )

            best_score = (
                    self.best_metrics.get('ai_to_human_semantic_sim', 0) * 0.3 +
                    self.best_metrics.get('ai_to_human_bleu', 0) * 0.2 +
                    self.best_metrics.get('ai_to_human_success_rate', 0) * 0.4 +
                    self.best_metrics.get('human_to_ai_success_rate', 0) * 0.1
            )

            if current_score > best_score:
                is_best = True
                self.best_metrics.update(eval_metrics)
                self.patience_counter = 0
                if self.is_main:
                    logger.info(f"🌟 新的最佳模型! (综合评分: {current_score:.4f})")
            else:
                self.patience_counter += 1
                if self.is_main:
                    logger.info(
                        f"⏳ Patience: {self.patience_counter}/{self.config['patience']} (当前: {current_score:.4f}, 最佳: {best_score:.4f})")

            # ⭐ 保存checkpoint（只在主进程）
            self.save_checkpoint(epoch, all_metrics, is_best=is_best)

            # ⭐ 在epoch结束时同步所有进程
            if self.world_size > 1:
                dist.barrier()

            # 早停
            if self.patience_counter >= self.config['patience']:
                if self.is_main:
                    logger.info("⚠️  早停触发")
                break

        # ⭐ 训练结束
        if self.is_main:
            if self.config.get('use_wandb', True):
                wandb.run.summary['best_semantic_similarity'] = self.best_metrics.get('semantic_similarity', 0)
                wandb.run.summary['best_bleu_score'] = self.best_metrics.get('bleu_score', 0)
                wandb.run.summary['best_transfer_success_rate'] = self.best_metrics.get('transfer_success_rate', 0)
                wandb.run.summary['best_ai_to_human_success_rate'] = self.best_metrics.get('ai_to_human_success_rate',
                                                                                           0)

            logger.info("\n✅ 训练完成!")
            logger.info(f"📊 最佳指标:")
            logger.info(f"   ⭐ AI→Human 成功率: {self.best_metrics.get('ai_to_human_success_rate', 0):.1%}")
            logger.info(f"   ⭐ AI→Human BLEU: {self.best_metrics.get('ai_to_human_bleu', 0):.3f}")
            logger.info(f"   ⭐ AI→Human 语义相似度: {self.best_metrics.get('ai_to_human_semantic_sim', 0):.3f}")

            if self.config.get('use_wandb', True):
                wandb.finish()


# ===================== 多卡训练主函数 =====================

def train_worker(rank, world_size, config, datasets):
    """每个GPU进程的训练函数"""
    # 设置分布式环境
    setup_distributed(rank, world_size)

    # 设置随机种子
    torch.manual_seed(config['seed'] + rank)
    np.random.seed(config['seed'] + rank)
    random.seed(config['seed'] + rank)

    # 初始化模型
    generator = T5StyleGenerator(
        model_name=config['model_name'],
        device=rank
    )

    discriminator = StyleDiscriminator(
        hidden_dim=512,
        style_dim=256
    )

    # 创建数据集
    train_dataset = StyleTransferDataset(
        datasets['train']['texts'],
        datasets['train']['labels'],
        generator.tokenizer,
        max_length=config['max_length']
    )

    dev_dataset = StyleTransferDataset(
        datasets['dev']['texts'],
        datasets['dev']['labels'],
        generator.tokenizer,
        max_length=config['max_length']
    )

    # ⭐ 使用DistributedSampler
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    dev_sampler = DistributedSampler(
        dev_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=config['batch_size'],
        sampler=dev_sampler,
        num_workers=4,
        pin_memory=True
    )

    # 创建训练器
    trainer = ImprovedAdversarialTrainer(
        generator,
        discriminator,
        train_loader,
        dev_loader,
        config,
        dev_data=datasets['dev'] if rank == 0 else None,
        rank=rank,
        world_size=world_size
    )

    # 开始训练
    trainer.train()

    # 清理
    cleanup_distributed()


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='Style Transfer Training with Multi-GPU Support (AI→Human Focused)')
    parser.add_argument('--gpu_ids', type=str, default="0,1,2",
                        help='GPU ids to use, e.g., "0,1,2,3" (default: use all available GPUs)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size per GPU (default: 16)')
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='Number of training epochs (default: 5)')
    parser.add_argument('--data_dir', type=str, default='./data',
                        help='Data directory (default: ./data)')
    parser.add_argument('--output_dir', type=str, default='./output-small',
                        help='Output directory (default: ./output)')
    parser.add_argument('--model_name', type=str, default='t5-small',
                        help='Model name (default: t5-small)')
    parser.add_argument('--wandb_project', type=str, default='style-transfer-ai2human-t5-small',
                        help='WandB project name')
    parser.add_argument('--no_wandb', action='store_true',
                        help='Disable WandB logging')

    # ⭐ 损失权重参数
    parser.add_argument('--lambda_reconstruction', type=float, default=8.0,
                        help='Weight for reconstruction loss (default: 8.0)')
    parser.add_argument('--lambda_semantic', type=float, default=5.0,
                        help='Weight for semantic loss (default: 5.0)')
    parser.add_argument('--lambda_length', type=float, default=5,
                        help='Weight for length loss (default: 2.0)')
    parser.add_argument('--lambda_adv', type=float, default=2.0,
                        help='Weight for adversarial loss (default: 2.0)')

    return parser.parse_args()


def main():
    args = parse_args()

    config = {
        # 基础配置
        'data_dir': args.data_dir,
        'output_dir': args.output_dir,
        'model_name': args.model_name,
        'max_length': 384,
        'batch_size': args.batch_size,
        'num_epochs': args.num_epochs,
        'patience': 5,
        'seed': 42,

        # 学习率
        'g_learning_rate': 1e-5,
        'd_learning_rate': 2e-5,
        'weight_decay': 0.01,

        # ⭐ 损失权重
        'lambda_reconstruction': args.lambda_reconstruction,
        'lambda_semantic': args.lambda_semantic,
        'lambda_length': args.lambda_length,
        'lambda_adv': args.lambda_adv,

        # 训练策略
        'd_train_interval': 3,
        'eval_interval': 1,

        # WandB配置
        'use_wandb': not args.no_wandb,
        'wandb_project': args.wandb_project,
        'wandb_run_name': f"ai2human_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    }

    os.makedirs(config['output_dir'], exist_ok=True)

    logger.info("📂 加载数据...")
    datasets = load_style_datasets(config['data_dir'])

    # ⭐ 根据命令行参数选择GPU
    if args.gpu_ids is not None:
        gpu_ids = [int(x) for x in args.gpu_ids.split(',')]
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, gpu_ids))
        world_size = len(gpu_ids)
        logger.info(f"🎮 使用指定的GPU: {gpu_ids}")
    else:
        world_size = torch.cuda.device_count()
        logger.info(f"🎮 使用所有可用GPU，共 {world_size} 个")

    # ⭐ 打印配置
    logger.info(f"\n{'=' * 80}")
    logger.info(f"🎯 训练配置 (优化 AI → Human)")
    logger.info(f"{'=' * 80}")
    logger.info(f"损失权重: Recon={config['lambda_reconstruction']}, Adv={config['lambda_adv']}, "
                f"Semantic={config['lambda_semantic']}, Length={config['lambda_length']}")
    logger.info(f"评估权重: AI→Human成功率=40%, 语义=30%, BLEU=20%, Human→AI=10%")
    logger.info(f"{'=' * 80}\n")

    if world_size > 1:
        logger.info("🚀 启动多卡训练...")
        mp.spawn(
            train_worker,
            args=(world_size, config, datasets),
            nprocs=world_size,
            join=True
        )
    else:
        logger.info("🚀 启动单卡训练...")
        train_worker(0, 1, config, datasets)


if __name__ == "__main__":
    main()