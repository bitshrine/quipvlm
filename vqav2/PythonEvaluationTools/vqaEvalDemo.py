# coding: utf-8

import sys
dataDir = 'vqav2'
sys.path.insert(0, '%s/PythonHelperTools/vqaTools' %(dataDir))
from vqav2.PythonHelperTools.vqaTools.vqa import VQA
from vqav2.PythonEvaluationTools.vqaEvaluation.vqaEval import VQAEval
import matplotlib.pyplot as plt
#import skimage.io as io
import json
import random
import os

from argparse import ArgumentParser

import pandas as pd

def eval(version: str, task_type: str, data_type: str, data_subtype: str, result_type: str):

	#parser = ArgumentParser()

	#parser.add_argument('--version', type=str, default='v2_', help="Version of the dataset. This should be '' when using VQA v2.0 dataset")
	#parser.add_argument('--task_type', type=str, default='OpenEnded', help="Type of task. 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0")
	#parser.add_argument('--data_type', type=str, default='mscoco', help="Type of data. 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0.")
	#parser.add_argument('--data_subtype', type=str, default='val2014')
	#parser.add_argument('--result_type', type=str, default='')
	#args = parser.parse_args()

	# set up file names and paths
	versionType = version#='v2_' # this should be '' when using VQA v2.0 dataset
	taskType    = task_type#='OpenEnded' # 'OpenEnded' only for v2.0. 'OpenEnded' or 'MultipleChoice' for v1.0
	dataType    = data_type#='mscoco'  # 'mscoco' only for v1.0. 'mscoco' for real and 'abstract_v002' for abstract for v1.0. 
	dataSubType = data_subtype#='val2014'
	annFile     ='%s/Annotations/%s%s_%s_annotations.json'%(dataDir, versionType, dataType, dataSubType)
	quesFile    ='%s/Questions/%s%s_%s_%s_questions.json'%(dataDir, versionType, taskType, dataType, dataSubType)
	imgDir      ='%s/Images/%s/%s/' %(dataDir, dataType, dataSubType)
	resultType  = result_type
	fileTypes   = ['results', 'accuracy', 'evalQA', 'evalQuesType', 'evalAnsType'] 

	# An example result json file has been provided in './Results' folder.  

	[resFile, accuracyFile, evalQAFile, evalQuesTypeFile, evalAnsTypeFile] = ['%s/Results/%s%s_%s_%s_%s_%s.json'%(dataDir, versionType, taskType, dataType, dataSubType, \
	resultType, fileType) for fileType in fileTypes]  

	# create vqa object and vqaRes object
	vqa = VQA(annFile, quesFile)
	vqaRes = vqa.loadRes(resFile, quesFile)

	# create vqaEval object by taking vqa and vqaRes
	vqaEval = VQAEval(vqa, vqaRes, n=2)   #n is precision of accuracy (number of places after decimal), default is 2

	# evaluate results
	"""
	If you have a list of question ids on which you would like to evaluate your results, pass it as a list to below function
	By default it uses all the question ids in annotation file
	"""
	answers = pd.read_json(resFile)
	answered_question_ids = list(answers['question_id'].unique().astype(int))
	print(answered_question_ids)
	vqaEval.evaluate(answered_question_ids) 

	# print accuracies
	print("\n")
	print("Overall Accuracy is: %.02f\n" %(vqaEval.accuracy['overall']))
	print("Per Question Type Accuracy is the following:")
	for quesType in vqaEval.accuracy['perQuestionType']:
		print("%s : %.02f" %(quesType, vqaEval.accuracy['perQuestionType'][quesType]))
	print("\n")
	print("Per Answer Type Accuracy is the following:")
	for ansType in vqaEval.accuracy['perAnswerType']:
		print("%s : %.02f" %(ansType, vqaEval.accuracy['perAnswerType'][ansType]))
	print("\n")
	"""
	# demo how to use evalQA to retrieve low score result
	evals = [quesId for quesId in vqaEval.evalQA if vqaEval.evalQA[quesId]<35]   #35 is per question percentage accuracy
	if len(evals) > 0:
		print('ground truth answers')
		randomEval = random.choice(evals)
		randomAnn = vqa.loadQA(randomEval)
		vqa.showQA(randomAnn)

		print('\n')
		print('generated answer (accuracy %.02f)'%(vqaEval.evalQA[randomEval]))
		ann = vqaRes.loadQA(randomEval)[0]
		print("Answer:   %s\n" %(ann['answer']))

		imgId = randomAnn[0]['image_id']
		imgFilename = 'COCO_' + dataSubType + '_'+ str(imgId).zfill(12) + '.jpg'
		if os.path.isfile(imgDir + imgFilename):
			I = io.imread(imgDir + imgFilename)
			plt.imshow(I)
			plt.axis('off')
			plt.show()

	# plot accuracy for various question types
	plt.bar(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].values(), align='center')
	plt.xticks(range(len(vqaEval.accuracy['perQuestionType'])), vqaEval.accuracy['perQuestionType'].keys(), rotation='0',fontsize=10)
	plt.title('Per Question Type Accuracy', fontsize=10)
	plt.xlabel('Question Types', fontsize=10)
	plt.ylabel('Accuracy', fontsize=10)
	plt.show()
	"""

	# save evaluation results to ./Results folder
	json.dump(vqaEval.accuracy,     open(accuracyFile,     'w'))
	#json.dump(vqaEval.evalQA,       open(evalQAFile,       'w'))
	#json.dump(vqaEval.evalQuesType, open(evalQuesTypeFile, 'w'))
	#json.dump(vqaEval.evalAnsType,  open(evalAnsTypeFile,  'w'))

