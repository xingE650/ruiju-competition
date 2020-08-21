#coding:utf8
import os
import sys
import csv
import copy
import json
import re
import csv
import random

'''
分割data/good_docs.json，得到训练集ner_train.tsv和测试集ner_test.tsv
'''

def item2example(item, phase='train'):
	text_a=(item["要素原始值"]).strip()
	text_a = re.sub(' +', '', text_a)
	text_a = re.sub('\n+', '', text_a)

	text_b = (item["句子"]).strip()
	text_b = re.sub(' +', '', text_b)
	text_b = re.sub('\n+', '', text_b)
	while len(text_a)+len(text_b)>=509:
		if len(text_a)>len(text_b):
			text_a=text_a[:len(text_a)-1]
		else:
			text_b = text_b[:len(text_b)-1]
	new_text_a = ""
	for index,_char in enumerate(text_a):
		if _char.strip() == "":
			continue
		if index==0:
			new_text_a += _char
		else:
			new_text_a += (" "+_char)

	new_text_b = ""
	for index, _char in enumerate(text_b):
		if _char.strip()=="":
			continue
		if index == 0:
			new_text_b += _char
		else:
			new_text_b += " " + _char

	if phase == 'train':
		labels = ["O" for i in range(len(text_b)+len(text_a))]
		# print(len(labels),len(text_b),len(text_a))
		showed_candidtes = item["被告人"]
		all_candidates = item["被告人集合"]
		for candidate in showed_candidtes:
			tmp_regex = re.compile(candidate)
			_finder = re.finditer(tmp_regex, text_a+text_b)
			for i in _finder:
				k = 0
				for _range in range(i.span()[0],i.span()[1]):
					if k==0:
						labels[_range] = "B-PER"
						k += 1
					else:
						labels[_range] = "I-PER"
						k += 1

		new_labels = ""
		for index, label in enumerate(labels):
			if index == 0:
				new_labels += label
			else:
				new_labels += " " + label
			pass
	else:
		new_labels = ""

	text_c = text_b
	# split text_c to produce new_text_c
	new_text_c = []
	for index, _char in enumerate(text_c):
		if _char.strip()=="":
			continue
		if index == 0:
			new_text_c.append("[PAD]")
		else:
			new_text_c.append(" ")
			new_text_c.append("[PAD]")

	all_candidates = item["被告人集合"]
	for candidate in all_candidates:
		tmp_regex = re.compile(candidate)
		_finder = re.finditer(tmp_regex, text_c)
		for i in _finder:
			k=0
			for _range in range(i.span()[0],i.span()[1]):
				new_text_c[_range*2] = candidate[k]
				k+=1
	new_text_c = "".join(new_text_c)


	
	return new_text_a, new_text_b, new_text_c, new_labels

if __name__ == "__main__":
	# 固定seed
	random.seed(233)


	file_input = os.path.join(os.getcwd(), 'data', "good_docs.json")
	input_json = json.load(open(file_input,"r",encoding="utf-8"))
	train_list = []

	for ind,item in enumerate(input_json):

		new_text_a, new_text_b, new_text_c, new_labels = item2example(item)

		# 这里是要做什么异常处理吗？没太明白
		if len(new_text_a.split(" "))+len(new_text_b.split(" "))!= len(new_labels.split(" ")):
			print (len(new_text_a.split(" ")),len(new_text_b.split(" ")),len(new_labels.split(" ")))
		# 将 new_text_a(分词后的要素原始值)、new_text_b（分词后的句子）、new_text_c（和new_text_b长度一致，然后把被告人集合以外的内容换成[PAD]）、new_labels（序列标注标签）
		train_list.append([new_text_a,new_text_b, new_text_c, new_labels])

	# 随机抽取10%作为测试数据
	prob = 0.1
	test_list=[]
	for _index,item in enumerate(train_list):
		ret = random.random()
		if ret < prob:
			test_list.append(item)
			train_list.pop(_index)

	with open(os.path.join(os.getcwd(), 'data', "ner_train.tsv"),"w",encoding="utf-8")as f_train:
		tsv_w_train = csv.writer(f_train, delimiter='\t')
		tsv_w_train.writerow(["text_a","text_b", "text_c", "label"])
		tsv_w_train.writerows(train_list)
	with open(os.path.join(os.getcwd(), 'data', "ner_test.tsv"),"w",encoding="utf-8")as f_test:
		tsv_w_test = csv.writer(f_test, delimiter='\t')
		tsv_w_test.writerow(["text_a","text_b", "text_c", "label"])
		tsv_w_test.writerows(test_list)

	# 平台测试使用的test.txt应该是和train.txt相同的格式，只是少了预测标签而已
	'''
	with open(os.path.join(os.getcwd(), 'data', "test.txt"), "w", encoding="utf-8") as f_test:
		for l in test_list:
			f_test.write(json.dumps(l, ensure_ascii=False))
			f_test.write('\n')
	'''

	print("finish generate ner_train.tsv and ner_test.tsv and test.txt")