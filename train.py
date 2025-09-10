import os

import torch
import yaml

from g2p.dataset import TSVDataset
from g2p.model import TransformerG2P, SimplifiedTransformerG2P
from g2p.trainer import Trainer

def export(input_trainer: Trainer):
    artifacts_dir = input_trainer.artifacts_dir
    model_path = os.path.join(artifacts_dir, 'model-latest.pt')
    onnx_path = os.path.join(artifacts_dir, 'g2p.onnx')
    info_path = os.path.join(artifacts_dir, 'info.yaml')

    print(f"exporting ONNX and metadata to: {artifacts_dir}")
    print("exporting model to ONNX format...")
    trainer.model.load_state_dict(torch.load(model_path))
    trainer.model.cpu()
    trainer.model.eval()

    simplified = SimplifiedTransformerG2P(
        embedding=trainer.model.embedding,
        pe=trainer.model.positional_encoding,
        encoder=trainer.model.encoder,
        fc=trainer.model.fc,
        max_len=trainer.model.pe.max_len,
    )

    for param in simplified.parameters():
        param.requires_grad = False

    simplified.export(onnx_path)
    print(f"onnx model exported: {onnx_path}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"using device: {device}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"\t{i} | {torch.cuda.get_device_name(i)}")

    config_path = "g2p/data/config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    print(f"loaded configuration from {config_path}")

    dict_path = config['data']['dict_path']

    graphemes = set()
    phonemes = set()

    with open(dict_path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                word = parts[0]
                phoneme_list = parts[1].split()
                graphemes.update(word.lower())
                phonemes.update(phoneme_list)

    graphemes = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(list(graphemes))
    phonemes = ['<unk>', '<pad>', '<bos>', '<eos>'] + sorted(list(phonemes))

    dataset = TSVDataset(dict_path, graphemes, phonemes)

    model = TransformerG2P(
        grapheme_vocab_size=len(graphemes),
        phoneme_vocab_size=len(phonemes),
        d_model=512,
        d_fft=1024,
        layers=4,
        dropout=0.1,
        heads=1
    )

    trainer = Trainer(
        run_name=config['training']['run_name'],
        device=device,
        model=model,
        dataset=dataset,
        batch_size=config['training']['batch_size'],
        max_steps=config['training']['max_steps'],
        learning_rate=config['training']['learning_rate'],
        warmup_steps=config['training']['warmup_steps'],
        plateau_factor=config['training']['plateau_factor'],
        plateau_patience=config['training']['plateau_patience'],
        early_stopping_patience=config['training']['early_stopping_patience'],
        generate_steps=config['training']['generate_steps'],
    )

    trainer.train()
    export(trainer)

    artifacts_dir = trainer.artifacts_dir
    print(f"\nall artifacts exported to: {artifacts_dir}")
    print("contents:")
    for file in os.listdir(artifacts_dir):
        file_path = os.path.join(artifacts_dir, file)
        if os.path.isfile(file_path):
            size = os.path.getsize(file_path)
            print(f"  {file} ({size:,} bytes)")
