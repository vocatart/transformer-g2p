import torch
import math
import torch.nn as nn

from g2p.util import make_len_mask, get_dedup_tokens


class PositionalEncoding(nn.Module):
    def __init__(self,
                 d_model: int,
                 dropout: float = 0.1,
                 max_len: int = 5000) -> None:
        """Creates positional encoding for transformer model.

		Args:
			d_model (int): Model size.
			dropout (float, optional): Dropout layer value. Defaults to 0.1.
			max_len (int, optional): Maximum sequence lengthS. Defaults to 5000.
		"""
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)
        self.scale = nn.Parameter(torch.ones(1))

        # create position matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass on positional encoding.

		Args:
			x (torch.Tensor): Input tensor.

		Returns:
			torch.Tensor: Output tensor with forward pass applied.
		"""
        seq_len = x.size(0)
        pe_tensor = getattr(self, 'pe')
        pos_encoding = pe_tensor[:seq_len, :].unsqueeze(1)
        x = x + self.scale * pos_encoding

        return self.dropout(x)


class TransformerG2P(nn.Module):
    def __init__(self,
                 grapheme_vocab_size: int,
                 phoneme_vocab_size: int,
                 d_model: int = 512,
                 d_fft: int = 1024,
                 layers: int = 4,
                 dropout: float = 0.1,
                 heads: int = 1) -> None:
        """Creates g2p with a single transformer encoder.

		Args:
			grapheme_vocab_size (int): Total grapheme count (including reserved tokens)
			phoneme_vocab_size (int): Total phoneme count (including reserved tokens)
			d_model (int, optional): Model size. Defaults to 512.
			d_fft (int, optional): Feedforward network dimension. Defaults to 1024.
			layers (int, optional): Number of encoder layers. Defaults to 4.
			dropout (float, optional): Dropout layer value. Defaults to 0.1.
			heads (int, optional): Number of heads. Defaults to 1.
		"""
        super().__init__()

        self.d_model = d_model
        self.embedding = nn.Embedding(grapheme_vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, dropout)

        nn.init.normal_(self.embedding.weight, mean=0, std=0.1)

        self.encoder = nn.TransformerEncoder(
            encoder_layer=nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=heads,
                dim_feedforward=d_fft,
                dropout=dropout,
                activation='relu'
            ),
            num_layers=layers,
            norm=nn.LayerNorm(d_model)
        )

        self.fc = nn.Linear(d_model, phoneme_vocab_size)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass on the model.

		Args:
			input (torch.Tensor): Input tensor.

		Returns:
			torch.Tensor: Output tensor with forward pass applied.
		"""
        x = input.transpose(0, 1)

        src_pad_mask = make_len_mask(x).to(x.device)

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc(x)
        x = x.transpose(0, 1)

        return x

    def predict(self, input: torch.Tensor) -> torch.Tensor:
        """Prediction with training model.

		Args:
			input (torch.Tensor): Input tensor of grapheme tokens.

		Returns:
			torch.Tensor: Output tensor of phoneme tokens.
		"""
        with torch.no_grad():
            x = self.forward(input)

        tokens, _ = get_dedup_tokens(x)

        return tokens


class SimplifiedTransformerG2P(nn.Module):
    def __init__(self,
                 embedding: nn.Embedding,
                 pe: PositionalEncoding,
                 encoder: nn.TransformerEncoder,
                 fc: nn.Linear,
                 max_len: int) -> None:
        """Creates simplified model architecture from existing weights.

		Args:
			embedding (nn.Embedding): Embedding layer from trained model.
			pe (PositionalEncoding): Positional encoding layer from trained model.
			encoder (nn.TransformerEncoder): Encoder layer from trained model.
			fc (nn.Linear): Linear projection layer from trained model.
			max_len (int): Maximum sequence length.
		"""
        super().__init__()

        self.embedding = embedding
        self.positional_encoding = pe
        self.encoder = encoder
        self.fc = fc
        self.max_len = max_len

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass (prediction) on deployable model type.

		Args:
			input (torch.Tensor): Input grapheme tensor.

		Returns:
			torch.Tensor: Output grapheme tensor
		"""
        x = input.transpose(0, 1)
        src_pad_mask = make_len_mask(x).to(x.device)

        x = self.embedding(x)
        x = self.positional_encoding(x)
        x = self.encoder(x, src_key_padding_mask=src_pad_mask)
        x = self.fc(x)
        x = x.transpose(0, 1)

        return torch.argmax(x, dim=-1)

    def export(self, path: str):
        """Export deployable model to ONNX format. Uses opset 18.

		Args:
			path (str): Path to save model.
		"""
        self.eval()

        src = torch.zeros((1, 8), dtype=torch.long)

        torch.onnx.export(
            self,
            (src,),
            path,
            input_names=['src'],
            output_names=['tgt'],
            dynamic_axes={
                'src': {1: 'T'},
                'tgt': {1: 'T'}
            },
            opset_version=18
        )
