from __future__ import division
import os, sys, pdb, shutil, time, random, copy
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn as nn
import resnet_mod
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time
import warnings
import pandas as pd
warnings.filterwarnings("ignore")
import numpy as np
import pickle
import datetime

parser = argparse.ArgumentParser(description='Use Resnet for Pruning', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--data_path', type=str, default='./data', help='Path to dataset')
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'], help='Choose between Cifar10/100 and ImageNet.')
parser.add_argument('--arch', metavar='ARCH', default='resnet_mod56', help='model architecture')
# Optimization options
parser.add_argument('--epochs', type=int, default=1, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--learning_rate', type=float, default=0.05, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225], help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1], help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./resnet/', help='Folder to save checkpoints and log.')
# parser.add_argument('--resume', default='./resnet/checkpoint.pth', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--resume', type=str, default='./resnet/model_best.pth', metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
# parser.add_argument('--evaluate', default=True, dest='evaluate', action='store_true', help='evaluate model on validation set')
parser.add_argument('--evaluate', default=True, type=bool, help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
args = parser.parse_args()
args.use_cuda = args.ngpu>0 and torch.cuda.is_available()
if args.manualSeed is None:
	args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
	torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True
if args.use_cuda:
	CUDA = True
else:
	CUDA = False

def print_log(print_string, log):
	print("{}".format(print_string))
	log.write('{}\n'.format(print_string))
	log.flush()

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, weight_mask_arr, set_gradient=False):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()
	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)

		if args.use_cuda:
			target = target.cuda(async=True)
			input = input.cuda()
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# compute gradient and do gradient descent step
		optimizer.zero_grad()
		loss.backward()
		#Put pruned mask
		if set_gradient == True:
			num_layers = len(weight_mask_arr)
			id=0
			for child in list(net.modules()):
				if isinstance(child, nn.Linear) or isinstance(child, nn.Conv2d):
					child.weight.grad.data *= weight_mask_arr[id]
					id+=1
					if id == num_layers:
						break
				
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		# if i % args.print_freq == 0:
			# print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
				# 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
				# 'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
				# 'Loss {loss.val:.4f} ({loss.avg:.4f})   '
				# 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})   '
				# 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
				# epoch, i, len(train_loader), batch_time=batch_time,
				# data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)

	# print_log('  **Train** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)
	return top1.avg, losses.avg

def validate(val_loader, model, criterion, log):
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()

	for i, (input, target) in enumerate(val_loader):
		if args.use_cuda:
			target = target.cuda(async=True)
			input = input.cuda()
		input_var = torch.autograd.Variable(input, volatile=True)
		target_var = torch.autograd.Variable(target, volatile=True)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

	# print_log('  **Test** Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5, error1=100-top1.avg), log)

	# return top1.avg, losses.avg
	return top1.avg

def save_checkpoint(state, is_best, save_path, filename):
	filename = os.path.join(save_path, filename)
	torch.save(state, filename)
	if is_best:
		bestname = os.path.join(save_path, 'model_best2.pth')
		shutil.copyfile(filename, bestname)

def adjust_learning_rate(optimizer, epoch, gammas, schedule):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.learning_rate
	assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
	for (gamma, step) in zip(gammas, schedule):
		if (epoch >= step):
			lr = lr * gamma
		else:
			break
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr
	return lr

def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res

	
# Init logger
if not os.path.isdir(args.save_path):
	os.makedirs(args.save_path)
log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
print_log('save path : {}'.format(args.save_path), log)
state = {k: v for k, v in args._get_kwargs()}
print_log(state, log)

# Init dataset
if not os.path.isdir(args.data_path):
	os.makedirs(args.data_path)
if args.dataset == 'cifar10':
	mean = [x / 255 for x in [125.3, 123.0, 113.9]]
	std = [x / 255 for x in [63.0, 62.1, 66.7]]
elif args.dataset == 'cifar100':
	mean = [x / 255 for x in [129.3, 124.1, 112.4]]
	std = [x / 255 for x in [68.2, 65.4, 70.4]]
else:
	assert False, "Unknown dataset : {}".format(args.dataset)

train_transform = transforms.Compose(
	[transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
	 transforms.Normalize(mean, std)])
test_transform = transforms.Compose(
	[transforms.ToTensor(), transforms.Normalize(mean, std)])

if args.dataset == 'cifar10':
	train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
	test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
	num_classes = 10
elif args.dataset == 'cifar100':
	train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
	test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
	num_classes = 100
elif args.dataset == 'svhn':
	train_data = dset.SVHN(args.data_path, split='train', transform=train_transform, download=True)
	test_data = dset.SVHN(args.data_path, split='test', transform=test_transform, download=True)
	num_classes = 10
elif args.dataset == 'stl10':
	train_data = dset.STL10(args.data_path, split='train', transform=train_transform, download=True)
	test_data = dset.STL10(args.data_path, split='test', transform=test_transform, download=True)
	num_classes = 10
elif args.dataset == 'imagenet':
	assert False, 'Do not finish imagenet code'
else:
	assert False, 'Do not support dataset : {}'.format(args.dataset)

train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
						 num_workers=args.workers, pin_memory=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
						num_workers=args.workers, pin_memory=True)


print_log("=> creating model '{}'".format(args.arch), log)
# Init model, criterion, and optimizer
net = resnet_mod.__dict__[args.arch](num_classes)
# print_log("=> network :\n {}".format(net), log)
net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

# define loss function (criterion) and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), state['learning_rate'], momentum=state['momentum'],
				weight_decay=state['decay'], nesterov=True)

if args.use_cuda:
	net.cuda()
	criterion.cuda()

recorder = RecorderMeter(args.epochs)
# optionally resume from a checkpoint
if args.resume:
	print("Resume is True")
	if os.path.isfile(args.resume):
		print_log("=> loading checkpoint '{}'".format(args.resume), log)
		checkpoint = torch.load(args.resume)
		recorder = checkpoint['recorder']
		args.start_epoch = checkpoint['epoch']
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
		print_log("=> loaded checkpoint '{}' (epoch {})" .format(args.resume, checkpoint['epoch']), log)
	else:
		raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
else:
	print_log("=> do not use any checkpoint for {} model".format(args.arch), log)

#Check whether to evaluate or train
if args.evaluate:
	print("Evaluate is true")
	validate(test_loader, net, criterion, log)
	
else:
	print("Evaluate is False")		
	# Main loop
	start_time = time.time()
	epoch_time = AverageMeter()
	print("start epoch is",args.start_epoch)
	print("epoch is",args.epochs)
	for epoch in range(args.start_epoch, args.epochs):
		current_learning_rate = adjust_learning_rate(optimizer, epoch, args.gammas, args.schedule)

		need_hour, need_mins, need_secs = convert_secs2time(epoch_time.avg * (args.epochs-epoch))
		need_time = '[Need: {:02d}:{:02d}:{:02d}]'.format(need_hour, need_mins, need_secs)

		print_log('\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [learning_rate={:6.4f}]'.format(time_string(), epoch, args.epochs, need_time, current_learning_rate) \
					+ ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False), 100-recorder.max_accuracy(False)), log)

		# train for one epoch
		train_acc, train_los = train(train_loader, net, criterion, optimizer, epoch, log)

		# evaluate on validation set
		val_acc,   val_los   = validate(test_loader, net, criterion, log)
		is_best = recorder.update(epoch, train_los, train_acc, val_los, val_acc, args.epochs)

		save_checkpoint({
		  'epoch': epoch + 1,
		  'arch': args.arch,
		  'state_dict': net.state_dict(),
		  'recorder': recorder,
		  'optimizer' : optimizer.state_dict(),
		  'args'      : copy.deepcopy(args),
		}, is_best, args.save_path, 'checkpoint4.pth')

		# measure elapsed time
		epoch_time.update(time.time() - start_time)
		start_time = time.time()
		recorder.plot_curve( os.path.join(args.save_path, 'curve.png') )
	log.close()

#Define pruning method
def prune(m, alpha):
	global CUDA
	weight_mask = torch.tensor([0.0])
	num_weights = 0                         #Number of weights in layer
	num_pruned = 0							#Number of weights pruned
	if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
		num_weights = torch.numel(m.weight.data)
		# use a byteTensor to represent the mask and convert it to a floatTensor for multiplication
		if CUDA:
			weight_mask = torch.ge(m.weight.data.abs(), (alpha * m.weight.data.std()).type('torch.cuda.FloatTensor')).type('torch.cuda.FloatTensor')
		else:
			weight_mask = torch.ge(m.weight.data.abs(), (alpha * m.weight.data.std()).type('torch.FloatTensor')).type('torch.FloatTensor')
		m.weight.data *= weight_mask
		num_pruned = num_weights - torch.nonzero(weight_mask).size(0)
	return num_pruned, num_weights, weight_mask
	
layer_index=0
num_pruned_arr, num_weights_arr, pruned_pct_arr, weight_mask_arr = [],[],[],[]
prev_pre_acc_arr = np.full(56, 1.0)
min_prune_arr = np.full(56, 0.0)

#Load arrays
# prev_pre_acc_arr = pickle.load(open("prev_pre_acc_arr.p", "rb"))
# min_prune_arr = pickle.load(open("min_prune_arr.p", "rb"))
print(prev_pre_acc_arr)
print(min_prune_arr)

retrain_count=0

def step(alpha, retrain=True, epoch_threshold=1, change_threshold=0.1, baseline_acc=93):		
	pre_acc, post_acc, epoch, num_pruned, num_weights = 0,0,0,0,0
	weight_mask = torch.tensor([0.0])
	global layer_index
	global num_pruned_arr, num_weights_arr, pruned_pct_arr, weight_mask_arr, prev_pre_acc_arr, min_prune_arr
	
	layers = []
	for child in list(net.modules()):
		if (isinstance(child, nn.Conv2d) or isinstance(child, nn.Linear)):
			layers.append(child)
			
	# Do pruning
	print(datetime.datetime.now())
	print("Pruning item ", layers[layer_index])
	pruned_pct = 0
	while True:
		num_pruned, num_weights, weight_mask = prune(layers[layer_index], alpha)
		new_pruned = num_pruned/num_weights - pruned_pct
		pruned_pct = num_pruned/num_weights
		if new_pruned <= 0.01:
			print("Pruning completed")
			break
	print('Weights : ', num_weights)
	print('Pruned  : ', num_pruned)
	print('%% pruned: %.3f %%'%(100*num_pruned/num_weights))
	num_pruned_arr.append(num_pruned)
	num_weights_arr.append(num_weights)
	pruned_pct_arr.append(pruned_pct)
	weight_mask_arr.append(weight_mask)
		
	# Do retraining
	pre_acc = float(validate(test_loader, net, criterion, log))
	print("Pre accuracy is ", pre_acc)
	
	#Our algorithm
	#If current prune pct less than previously seen, and current pre accuracy more than previously seen then assert post accuracy hits baseline_acc
	if pruned_pct <= min_prune_arr[layer_index] and pre_acc>=prev_pre_acc_arr[layer_index]:	
		#Assert post accuracy = baseline_acc
		post_acc = baseline_acc
		print("No need to retrain")
		global retrain_count
		retrain_count+=1
		print("No retrain count is ",retrain_count)
	else:	
		print("Retraining net")
		if retrain and pre_acc < baseline_acc:
			prev_acc=0
			acc_change = -1
			adjust_learning_rate(optimizer, args.start_epoch, args.gammas, args.schedule)
			while epoch < epoch_threshold and post_acc < baseline_acc:
				print('Epoch: [%d]\t'%(epoch))
				train(train_loader, net, criterion, optimizer, epoch, log, weight_mask_arr, set_gradient=True)	
				post_acc = float(validate(test_loader, net, criterion, log))
				acc_change = post_acc - prev_acc
				prev_acc = post_acc
				epoch+=1
				if acc_change < change_threshold:
					print("Retraining completed")
					break
		#Assert pre=post if pre accuracy >=baseline_acc
		# if post_acc == 0:
		else:
			post_acc = pre_acc

		if post_acc >= baseline_acc:
			prev_pre_acc_arr[layer_index] = pre_acc
			if min_prune_arr[layer_index] < pruned_pct:
				min_prune_arr[layer_index] = pruned_pct
			
	
	#Store arrays - caching
	pickle.dump(prev_pre_acc_arr, open("prev_pre_acc_arr.p", "wb"))
	pickle.dump(min_prune_arr, open("min_prune_arr.p", "wb"))
	
	##
	#End of our algorithm
	##
			
	total_pruned  = sum(x for x in num_pruned_arr)
	total_weights = sum(x for x in num_weights_arr)

	#Move to next layer
	layer_index+=1
	model_weights = 848944
	print("Accuracy is ", post_acc)
	
	#Calculate reward per layer
	acc_penalty = -max((baseline_acc - post_acc),0)*10/20*15
	pruned_rew = 100*(num_pruned/model_weights)
	total_rew = (acc_penalty + pruned_rew)*2
	
	#Construct layer embedding
	new_obs = np.array([layer_index, acc_penalty, pruned_rew])
	print("New observation is ",new_obs)
	
	#Set episode done flag
	if layer_index == 56:
		done = True
		print("Total pruned in model ", total_pruned)
		print("Total % pruned ", total_pruned/total_weights)
	else:
		done = False
	print(datetime.datetime.now())
	return new_obs, total_rew, done, 0

def reset():
	global net
	global layer_index
	global num_pruned_arr, num_weights_arr, pruned_pct_arr, weight_mask_arr
	layer_index=0
	num_pruned_arr, num_weights_arr, pruned_pct_arr, weight_mask_arr = [],[],[],[]
	
	if os.path.isfile(args.resume):
		checkpoint = torch.load(args.resume)
		recorder = checkpoint['recorder']
		args.start_epoch = checkpoint['epoch']
		net.load_state_dict(checkpoint['state_dict'])
		optimizer.load_state_dict(checkpoint['optimizer'])
	else:
		raise ValueError("=> no checkpoint found at '{}'".format(args.resume))
	obs = np.array([0, 0, 0])
	print("Obs on reset ", obs)
	return obs
