import numpy as np


class Evaluator(object):
	def __init__(self, num_class):
		self.num_class = num_class
		self.confusion_matrix = np.zeros((self.num_class,) * 2)

	def Pixel_Accuracy(self):
		Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
		return Acc

	def Pixel_Accuracy_Class(self):
		Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis = 1)
		Acc = np.nanmean(Acc)
		return Acc

	def OMMISION(self):
		FN = self.confusion_matrix.sum(axis = 1) - np.diag(self.confusion_matrix)
		TP = np.diag(self.confusion_matrix)
		Omision = FN / (FN + TP)
		Omision = np.nanmean(Omision)
		return Omision

	def COMMISION(self):
		FP = self.confusion_matrix.sum(axis = 0) - np.diag(self.confusion_matrix)
		FN = self.confusion_matrix.sum(axis = 1) - np.diag(self.confusion_matrix)
		TP = np.diag(self.confusion_matrix)
		TN = self.confusion_matrix.sum() - (FP + FN + TP)
		COmision = FP / (TN + FP)
		COmision = np.nanmean(COmision)
		return COmision

	def Mean_Intersection_over_Union(self):
		MIoU = np.diag(self.confusion_matrix) / (
				np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) -
				np.diag(self.confusion_matrix))
		MIoU = np.nanmean(MIoU)
		return MIoU

	def Frequency_Weighted_Intersection_over_Union(self):
		freq = np.sum(self.confusion_matrix, axis = 1) / np.sum(self.confusion_matrix)
		iu = np.diag(self.confusion_matrix) / (
				np.sum(self.confusion_matrix, axis = 1) + np.sum(self.confusion_matrix, axis = 0) -
				np.diag(self.confusion_matrix))

		FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
		return FWIoU

	def _generate_matrix(self, gt_image, pre_image):
		mask = (gt_image >= 0) & (gt_image < self.num_class)
		label = self.num_class * gt_image[mask].astype('int') + pre_image[mask]
		count = np.bincount(label, minlength = self.num_class ** 2)
		confusion_matrix = count.reshape(self.num_class, self.num_class)
		return confusion_matrix

	def add_batch(self, gt_image, pre_image):
		assert gt_image.shape == pre_image.shape
		self.confusion_matrix += self._generate_matrix(gt_image, pre_image)

	def reset(self):
		self.confusion_matrix = np.zeros((self.num_class,) * 2)
