#encoding:utf8
import os
import sys
sys.path.append(".")
sys.path.append("..")
import json
import re

total_data_dir = os.path.join(os.getcwd(), 'data', 'train.txt')

queryKeys = ["被告人","要素原始值","句子"]
error_docs = {}
error_docs_dir = os.path.join(os.getcwd(),'data', 'error_docs.json')
good_docs = []
good_docs_dir = os.path.join(os.getcwd(),'data', 'good_docs.json')

with open(total_data_dir,"r",encoding="utf-8") as input:
	for line in input.readlines():
		case = json.loads(line)
		defendent = case["被告人"]
		defendents = case["被告人集合"]
		doc_mark = ""
		if "⽂号" in case:
			doc_mark = case["⽂号"]
		factor_name = case["要素名称"]
		ovalue = case["要素原始值"]
		sentence = case["句子"]
		paragraph_content = case["段落内容"]
		integrity = True
		for queryKey in queryKeys:
			if not isinstance(case[queryKey],list):
				regex = re.compile(case[queryKey])
				if regex.search(paragraph_content) == None:
					if paragraph_content in error_docs:
						if doc_mark!="":
							if doc_mark not in error_docs[paragraph_content] :
								error_docs[paragraph_content][doc_mark] = {}
								if "unmatch_values" not in error_docs[paragraph_content][doc_mark]:
									error_docs[paragraph_content][doc_mark]["unmatch_values"] = []
								error_docs[paragraph_content][doc_mark]["unmatch_values"].append([queryKey,case[queryKey]])
								integrity = False
						else:
							if "unmatch_values" not in error_docs[paragraph_content]:
								error_docs[paragraph_content]["unmatch_values"] = []
							error_docs[paragraph_content]["unmatch_values"].append([queryKey,case[queryKey]])
							integrity = False
					else:
						error_docs[paragraph_content] = {}
						if doc_mark != "":
							if doc_mark not in error_docs[paragraph_content]:
								error_docs[paragraph_content][doc_mark] = {}
								if "unmatch_values" not in error_docs[paragraph_content][doc_mark]:
									error_docs[paragraph_content][doc_mark]["unmatch_values"] = []
								error_docs[paragraph_content][doc_mark]["unmatch_values"].append(
									[queryKey, case[queryKey]])
								integrity = False
						else:
							if "unmatch_values" not in error_docs[paragraph_content]:
								error_docs[paragraph_content]["unmatch_values"] = []
								error_docs[paragraph_content]["unmatch_values"].append([queryKey, case[queryKey]])
								integrity = False

			else:
				for item in case[queryKey]:
					regex = re.compile(item)
					if regex.search(paragraph_content) == None:
						if paragraph_content in error_docs:
							if doc_mark != "":
								if doc_mark not in error_docs[paragraph_content]:
									error_docs[paragraph_content][doc_mark] = {}
									if "unmatch_values" not in error_docs[paragraph_content][doc_mark]:
										error_docs[paragraph_content][doc_mark]["unmatch_values"] = []
									error_docs[paragraph_content][doc_mark]["unmatch_values"].append(
										[queryKey, case[queryKey]])
									integrity = False
							else:
								if "unmatch_values" not in error_docs[paragraph_content]:
									error_docs[paragraph_content]["unmatch_values"] = []
									error_docs[paragraph_content]["unmatch_values"].append([queryKey, case[queryKey]])
									integrity = False

						else:
							error_docs[paragraph_content] = {}
							if doc_mark != "":
								if doc_mark not in error_docs[paragraph_content]:
									error_docs[paragraph_content][doc_mark] = {}
									if "unmatch_values" not in error_docs[paragraph_content][doc_mark]:
										error_docs[paragraph_content][doc_mark]["unmatch_values"] = []
										error_docs[paragraph_content][doc_mark]["unmatch_values"].append(
										[queryKey, case[queryKey]])
										integrity = False
							else:
								if "unmatch_values" not in error_docs[paragraph_content]:
									error_docs[paragraph_content]["unmatch_values"] = []
									error_docs[paragraph_content]["unmatch_values"].append([queryKey, case[queryKey]])
									integrity = False
		if integrity:
			good_docs.append(case)
with open(error_docs_dir,"w",encoding="utf-8")as write_error:
	json.dump(error_docs,write_error,ensure_ascii=False)
with open(good_docs_dir,"w",encoding="utf-8")as write_good:
	json.dump(good_docs,write_good,ensure_ascii=False)


