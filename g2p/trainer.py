import torch
import os

from datetime import datetime
from model import TransformerG2P
from dataset import TSVDataset
from util import PAD_IDX, UNK_IDX, BOS_IDX, EOS_IDX, get_dedup_tokens
from torch.utils.data.dataset import random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class Trainer():
	def __init__(self,
			  run_name: str,
			  device: torch.device,
			  model: TransformerG2P,
			  dataset: TSVDataset,
			  batch_size: int = 32,
			  max_steps: int = 100000,
			  learning_rate: float = 0.0001,
			  warmup_steps: int = 10000,
			  plateau_factor: float = 0.5,
			  pleateau_patience: int = 10,
			  early_stopping_patience: int = 5,
			  generate_steps: int = 10000,
			  validate_steps: int = 10000,
			  checkpoint_steps: int = 1000000,
			  generated_samples: int = 10,
			  validation_divide_by: int = 10,
			  dl_workers: int = 0,
			  seed = None) -> None:
		
		self.run_name = run_name
		self.device = device
		self.model = model.to(device)
		self.dataset = dataset
		self.batch_size = batch_size
		self.max_steps = max_steps
		self.learning_rate = learning_rate
		self.warmup_steps = warmup_steps
		self.pleateau_factor = plateau_factor
		self.pleateau_patience = pleateau_patience
		self.early_stopping_patience = early_stopping_patience
		self.generate_steps = generate_steps
		self.validate_steps = validate_steps
		self.checkpoint_steps = checkpoint_steps
		self.generated_samples = generated_samples
		self.validation_divide_by = validation_divide_by
		
		self.run_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

		self.artifacts_dir = f"artifacts/{self.run_name}"
		os.makedirs(self.artifacts_dir, exist_ok=True)

		self.model.to(device)

		self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)
		self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

		self.best_val_loss = float('inf')
		self.early_stopping_counter = 0
		self.best_model_state = None

		self.pleateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer,
																	   factor=plateau_factor,
																	   patience=pleateau_patience,
																	   mode='min')

		self.warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
			self.optimizer,
			lr_lambda=lambda step: min(1.0, step / max(1, warmup_steps)) if warmup_steps > 0 else 1.0
		)

		valid_set_size = len(dataset) // validation_divide_by
		train_set_size = len(dataset) - valid_set_size

		generator = torch.Generator(device=torch.device('cpu'))
		if seed is not None:
			generator.manual_seed(seed)

		valid_set, train_set = random_split(dataset, [valid_set_size, train_set_size], generator=generator)

		def collate_fn(batch):
			grapheme_batch, phoneme_batch = [], []

			for word_tensor, phoneme_tensor in batch:
				graphemes_with_tokens = torch.cat([torch.tensor([BOS_IDX]), word_tensor, torch.tensor([EOS_IDX])])
				grapheme_batch.append(graphemes_with_tokens)

				phonemes_with_tokens = torch.cat([torch.tensor([BOS_IDX]), phoneme_tensor, torch.tensor([EOS_IDX])])
				phoneme_batch.append(phonemes_with_tokens)

			grapheme_lengths = torch.tensor([len(x) for x in grapheme_batch], dtype=torch.long)
			phoneme_lengths = torch.tensor([len(x) for x in phoneme_batch], dtype=torch.long)

			grapheme_batch = torch.nn.utils.rnn.pad_sequence(grapheme_batch, batch_first=True, padding_value=PAD_IDX)
			phoneme_batch = torch.nn.utils.rnn.pad_sequence(phoneme_batch, batch_first=True, padding_value=PAD_IDX)

			return (grapheme_batch, grapheme_lengths), (phoneme_batch, phoneme_lengths)
		
		self.training_loader = torch.utils.data.DataLoader(
			dataset=train_set,
			batch_size=batch_size,
			shuffle=True,
			collate_fn=collate_fn,
			num_workers=dl_workers
		)

		self.validation_loader = torch.utils.data.DataLoader(
			dataset=valid_set,
			batch_size=batch_size,
			shuffle=False,
			collate_fn=collate_fn,
			num_workers=dl_workers
		)

	def _train_epoch(self, step: int, total_loss: float, epoch: int, writer: SummaryWriter):
		self.model.train()

		running_loss = 0.0
		batch_count = 0
		pbar = tqdm(self.training_loader, desc=f"Epoch {epoch} | Step {step}", unit="step")

		for (grapheme_batch, grapheme_lengths), (phoneme_batch, phoneme_lengths) in pbar:
			if step >= self.max_steps:
				break

			grapheme_batch = grapheme_batch.to(self.device)
			phoneme_batch = phoneme_batch.to(self.device)
			grapheme_lengths = grapheme_lengths.to(self.device)
			phoneme_lengths = phoneme_lengths.to(self.device)

			self.optimizer.zero_grad()

			logits = self.model(grapheme_batch)

			# teacher forcing
			targets = phoneme_batch[:, 1:].contiguous()

			# padding
			if logits.size(1) > targets.size(1):
				logits = logits[:, :targets.size(1), :].contiguous
			elif logits.size(1) < targets.size(1):
				targets = targets[:, :logits.size(1)].contiguous()

			# flatten logits/targets for loss
			loss = self.loss_function(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
			
			loss.backward()

			torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

			self.optimizer.step()

			if step <= self.warmup_steps:
				self.warmup_scheduler.step()

			running_loss += loss.item()
			total_loss += loss.item()
			batch_count += 1
			step += 1

			# metrics
			avg_loss = running_loss / batch_count
			warmup_status = f"Warmup: {step}/{self.warmup_steps}" if step <= self.warmup_steps else None
			pbar.set_postfix(loss=f"{avg_loss:.4f}", lr=f"{self.optimizer.param_groups[0]['lr']:.6f}",
					step=f"{step}/{self.max_steps}", status=warmup_status)
			
			writer.add_scalar('loss/train_step', loss.item(), step)
			writer.add_scalar('metrics/lr', self.optimizer.param_groups[0]['lr']/ step)

			if step % self.validate_steps == 0:
				val_loss = self._validate(writer, step)
				if step > self.warmup_steps:
					self.pleateau_scheduler.step(val_loss)

				# early stop
				if self.early_stopping_counter >= self.early_stopping_patience:
					if self.best_model_state is not None:
						self.model.load_state_dict(self.best_model_state)
					return step, total_loss, True

				self.model.train()
			
			if step % self.generate_steps == 0:
				self._save_checkpoint(step, epoch, total_loss)
		
		return step, total_loss, False
	
	def _validate(self, writer: SummaryWriter, step: int):
		self.model.eval()
		total_val_loss = 0.0
		num_batches = 0

		with torch.no_grad():
			for (grapheme_batch, grapheme_lengths), (phoneme_batch, phoneme_lengths) in self.validation_loader:
				grapheme_batch = grapheme_batch.to(self.device)
				phoneme_batch = phoneme_batch.to(self.device)
				grapheme_lengths = grapheme_lengths.to(self.device)
				phoneme_lengths = phoneme_lengths.to(self.device)

				logits = self.model(grapheme_batch)
				targets = phoneme_batch[:, 1:].contiguous()

				if logits.size(1) > targets.size(1):
					logits = logits[:, :targets.size(1), :].contiguous()
				elif logits.size(1) < targets.size(1):
					targets = targets[:, :logits.size(1)].contiguous()

				loss = self.loss_function(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))

				if not torch.isnan(loss):
					total_val_loss += loss.item()
					num_batches += 1

		avg_val_loss = total_val_loss / num_batches if num_batches > 0 else 0

		if avg_val_loss < self.best_val_loss:
			self.best_val_loss = avg_val_loss
			self.early_stopping_counter = 0
			self.best_model_state = self.model.state_dict().copy()
		else:
			self.early_stopping_counter += 1

		writer.add_scalar('loss/valid', avg_val_loss, step)

		return avg_val_loss
	
	def _generate_samples(self, writer: SummaryWriter, step: int):
		self.model.eval()

		generator_dataloader = torch.utils.data.DataLoader(
			dataset=self.dataset,
			batch_size=self.generated_samples,
			shuffle=True,
			collate_fn=self.validation_loader.collate_fn,
			num_workers=self.validation_loader.num_workers
		)

		sample_batch = next(iter(generator_dataloader))
		(grapheme_batch, _), (phoneme_batch, _) = sample_batch

		n_samples = grapheme_batch.size(0)
		grapheme_sample = grapheme_batch.to(self.device)
		phoneme_sample = phoneme_batch.to(self.device)

		logits = self.model(grapheme_sample)

		predictions, _ = get_dedup_tokens(logits)

		for i in range(n_samples):
			input_text = self._tokens_to_graphemes(grapheme_batch[i])
			target_text = self._tokens_to_phonemes(phoneme_sample[i])
			predicted_text = self._tokens_to_phonemes(predictions[i])

			writer.add_text(f'samples/input/{i}', input_text, step)
			writer.add_text(f'samples/target/{i}', target_text, step)
			writer.add_text(f'samples/predicted/{i}', predicted_text, step)

	def _tokens_to_graphemes(self, tokens):
		vocab = {i: token for token, i in self.dataset.grapheme_indices}

		if isinstance(tokens, torch.Tensor):
			tokens = tokens.cpu().tolist()

		return ''.join([vocab.get(token, '<UNK>') for token in tokens])
	
	def _tokens_to_phonemes(self, tokens):
		vocab = {i: token for token, i in self.dataset.phoneme_indices}

		if isinstance(tokens, torch.Tensor):
			tokens = tokens.cpu().tolist()

		return ' '.join([vocab.get(token, '<UNK>') for token in tokens])
	
	def _save_checkpoint(self, step: int, epoch: int, total_loss: float):
		checkpoint = {
			'model_state_dict': self.model.state_dict(),
			'optimizer_state_dict': self.optimizer.state_dict(),
			'warmup_scheduler_state_dict': self.warmup_scheduler.state_dict(),
			'plateau_scheduler_state_dict': self.pleateau_scheduler.state_dict(),
			'step': step,
			'epoch': epoch,
			'loss': total_loss
		}

		torch.save(checkpoint, f"{self.artifacts_dir}/model-step-{step}.pt")

	# TODO load
	
	def train(self):
		log_dir = f"{self.artifacts_dir}/logs/{self.run_datetime}"
		writer = SummaryWriter(log_dir=log_dir)

		step = 0
		total_loss = 0.0
		epoch = 0
		early_stop = False

		try:
			while step < self.max_steps and not early_stop:
				epoch += 1
				step, total_loss, early_stop = self._train_epoch(step, total_loss, epoch, writer)

				writer.add_scalar('loss/epoch', total_loss / step if step > 0 else 0, epoch)

				if early_stop:
					break

				if step >= self.max_steps:
					break

		except KeyboardInterrupt:
			print("\nTraining interrupted")
		finally:
			self._save_checkpoint(step, epoch, total_loss)
			writer.close()

		return step, total_loss