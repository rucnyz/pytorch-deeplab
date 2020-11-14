# -*- coding: utf-8 -*-
# @Time    : 2020/11/14 9:38
# @Author  : nieyuzhou
# @File    : test.py
# @Software: PyCharm
from tqdm import tqdm
from dataloaders.datasets import flood
from modeling.deeplab import DeepLab
from mypath import Path
import torch


def computeIOU(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()

	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	intersection = torch.sum(output * target)
	union = torch.sum(target) + torch.sum(output) - intersection
	iou = (intersection + .0000001) / (union + .0000001)
	if iou != iou:
		print("failed, replacing with 0")
		iou = torch.tensor(0).float()
	return iou


def computeAccuracy(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()
	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	correct = torch.sum(output.eq(target))
	return correct.float() / len(target)


def truePositives(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()

	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	correct = torch.sum(output * target)
	return correct


def trueNegatives(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()
	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	output = (output == 0)
	target = (target == 0)
	correct = torch.sum(output * target)
	return correct


def falsePositives(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()
	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	output = (output == 1)
	target = (target == 0)
	correct = torch.sum(output * target)
	return correct


def falseNegatives(output, target):
	output = torch.argmax(output, dim = 1).flatten()
	target = target.flatten()
	no_ignore = target.ne(255).cuda()
	output = output.masked_select(no_ignore)
	target = target.masked_select(no_ignore)
	output = (output == 0)
	target = (target == 1)
	correct = torch.sum(output * target)
	return correct


def test_loop(test_data_loader, model):
	model = model.eval()
	model = model.cuda()
	count = 0
	iou = 0
	loss = 0
	accuracy = 0
	tp = 0
	fp = 0
	tn = 0
	fn = 0
	with torch.no_grad():
		for (images, labels) in tqdm(test_data_loader):
			model = model.cuda()
			outputs = model(images.cuda())
			valid_iou = computeIOU(outputs, labels.cuda())
			iou += valid_iou
			accuracy += computeAccuracy(outputs, labels.cuda())
			tp += truePositives(outputs, labels.cuda())
			fp += falsePositives(outputs, labels.cuda())
			tn += trueNegatives(outputs, labels.cuda())
			fn += falseNegatives(outputs, labels.cuda())

			count += 1

	iou = iou / count
	print("Test Mean IOU:", iou)
	print("Total IOU:", (tp.float() / (fn + fp + tp)))
	print("OMISSON:", fn.float() / (fn + tp))
	print("COMMISSON:", fp.float() / (tn + fp))
	print("Test Accuracy:", accuracy / count)


def main():
	workpath = Path.db_root_dir('flood')
	test_data = flood.load_flood_valid_data(workpath)
	test_dataset = flood.InMemoryDataset(test_data, flood.processTestIm)
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size = 4, shuffle = True, sampler = None,
	                                          batch_sampler = None, num_workers = 0, collate_fn = lambda x: (
			torch.cat([a[0] for a in x], 0), torch.cat([a[1] for a in x], 0)),
	                                          pin_memory = True, drop_last = False, timeout = 0,
	                                          worker_init_fn = None)

	model = DeepLab(num_classes = 2,
	                backbone = 'resnet',
	                output_stride = 16,
	                sync_bn = None,
	                freeze_bn = False)
	model.load_state_dict(torch.load('/home/u2019202317/deeplab2/run/flood/deeplab-resnet/model_best.pth.tar')['state_dict'])
	test_loop(test_loader, model)


if __name__ == '__main__':
	main()
