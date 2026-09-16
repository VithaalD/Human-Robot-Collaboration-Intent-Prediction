import torch
import torch.nn as nn
import torch.optim as optim
import tensorboardX
import os
import random
import numpy as np

from train import train_epoch
from torch.utils.data import DataLoader
from validation import val_epoch
from opts import parse_opts
from model import generate_model
from torch.optim import lr_scheduler
from dataset import get_training_set, get_validation_set
from mean import get_mean, get_std
from spatial_transforms import (
	Compose, Normalize, Scale, CenterCrop, CornerCrop, MultiScaleCornerCrop,
	MultiScaleRandomCrop, RandomHorizontalFlip, ToTensor)
from temporal_transforms import LoopPadding, TemporalRandomCrop
from target_transforms import ClassLabel, VideoID
from target_transforms import Compose as TargetCompose


def build_input_config(opt, training_data):
	"""Save preprocessing and class-index order with each checkpoint."""
	labels = [training_data.class_names[i] for i in sorted(training_data.class_names)]
	if len(labels) != opt.n_classes:
		raise ValueError(f"--n_classes={opt.n_classes} but annotations contain {len(labels)} classes.")
	mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	if opt.no_mean_norm and not opt.std_norm:
		mean = [0.0, 0.0, 0.0]
	return {
		'sample_duration': opt.sample_duration, 'sample_size': opt.sample_size,
		'norm_value': opt.norm_value, 'mean': mean,
		'std': get_std(opt.norm_value) if opt.std_norm else [1.0, 1.0, 1.0],
		'color_order': 'RGB', 'n_classes': opt.n_classes, 'class_labels': labels,
	}


def resume_model(opt, model, optimizer):
	"""Resume on the selected device, rejecting a changed input contract."""
	device = next(model.parameters()).device
	try:
		checkpoint = torch.load(opt.resume_path, map_location=device, weights_only=True)
	except TypeError:  # Older PyTorch versions did not expose weights_only.
		checkpoint = torch.load(opt.resume_path, map_location=device)
	saved_input = checkpoint.get('input_config')
	if saved_input:
		for key, value in getattr(opt, 'input_config', {}).items():
			if key in saved_input and saved_input[key] != value:
				raise ValueError(f"Cannot resume: checkpoint {key}={saved_input[key]!r}, current {key}={value!r}. Match the original training settings.")
	else:
		print("Legacy checkpoint has no input metadata; its historical frame duration is unknown.")
	model.load_state_dict(checkpoint['state_dict'])
	optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
	print("Model Restored from Epoch {}".format(checkpoint['epoch']))
	return checkpoint['epoch'] + 1


def get_loaders(opt):
	""" Make dataloaders for train and validation sets
	"""
	if opt.sample_duration <= 0:
		raise ValueError("--sample_duration must be positive.")
	# train loader
	opt.mean = get_mean(opt.norm_value, dataset=opt.mean_dataset)
	opt.std = get_std(opt.norm_value)
	if opt.no_mean_norm and not opt.std_norm:
		norm_method = Normalize([0, 0, 0], [1, 1, 1])
	elif not opt.std_norm:
		norm_method = Normalize(opt.mean, [1, 1, 1])
	else:
		norm_method = Normalize(opt.mean, opt.std)
	spatial_transform = Compose([
		# crop_method,
		Scale((opt.sample_size, opt.sample_size)),
		# RandomHorizontalFlip(),
		ToTensor(opt.norm_value), norm_method
	])
	temporal_transform = TemporalRandomCrop(opt.sample_duration)
	target_transform = ClassLabel()
	training_data = get_training_set(opt, spatial_transform,
									 temporal_transform, target_transform)
	train_loader = torch.utils.data.DataLoader(
		training_data,
		batch_size=opt.batch_size,
		shuffle=True,
		num_workers=opt.num_workers,
		pin_memory=True)

	# validation loader
	spatial_transform = Compose([
		Scale((opt.sample_size, opt.sample_size)),
		# CenterCrop(opt.sample_size),
		ToTensor(opt.norm_value), norm_method
	])
	target_transform = ClassLabel()
	temporal_transform = LoopPadding(opt.sample_duration)
	validation_data = get_validation_set(
		opt, spatial_transform, temporal_transform, target_transform)
	val_loader = torch.utils.data.DataLoader(
		validation_data,
		batch_size=opt.batch_size,
		shuffle=False,
		num_workers=opt.num_workers,
		pin_memory=True)
	return train_loader, val_loader


def main_worker():
	opt = parse_opts()
	print(opt)

	seed = 1
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	# CUDA for PyTorch
	device = torch.device(f"cuda:{opt.gpu}" if opt.use_cuda else "cpu")

	# tensorboard
	summary_writer = tensorboardX.SummaryWriter(log_dir='tf_logs')

	# defining model
	model =  generate_model(opt, device)
	# get data loaders
	train_loader, val_loader = get_loaders(opt)
	opt.input_config = build_input_config(opt, train_loader.dataset)
	os.makedirs("snapshots", exist_ok=True)

	# optimizer
	crnn_params = list(model.parameters())
	optimizer = torch.optim.Adam(crnn_params, lr=opt.lr_rate, weight_decay=opt.weight_decay)

	# scheduler = lr_scheduler.ReduceLROnPlateau(
	# 	optimizer, 'min', patience=opt.lr_patience)
	criterion = nn.CrossEntropyLoss()

	# resume model
	if opt.resume_path:
		start_epoch = resume_model(opt, model, optimizer)
	else:
		start_epoch = 1

	# start training
	for epoch in range(start_epoch, opt.n_epochs + 1):
		train_loss, train_acc = train_epoch(
			model, train_loader, criterion, optimizer, epoch, opt.log_interval, device)
		val_loss, val_acc = val_epoch(
			model, val_loader, criterion, device)

		# saving weights to checkpoint
		if (epoch) % opt.save_interval == 0:
			# scheduler.step(val_loss)
			# write summary
			summary_writer.add_scalar(
				'losses/train_loss', train_loss, global_step=epoch)
			summary_writer.add_scalar(
				'losses/val_loss', val_loss, global_step=epoch)
			summary_writer.add_scalar(
				'acc/train_acc', train_acc * 100, global_step=epoch)
			summary_writer.add_scalar(
				'acc/val_acc', val_acc * 100, global_step=epoch)

			state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'input_config': opt.input_config}
			torch.save(state, os.path.join('snapshots', f'{opt.model}-Epoch-{epoch}-Loss-{val_loss}.pth'))
			print("Epoch {} model saved!\n".format(epoch))


if __name__ == "__main__":
	main_worker()