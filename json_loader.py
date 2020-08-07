import json
import os
pwd_path = os.path.abspath(os.path.dirname(__file__))

class JsonLoader(object):
	def __init__(self):
		self.test_json_path = pwd_path+"/new_data/res/final_test.txt"
		self.predict_file_path = pwd_path + "/new_data/predict_result"

	def load_test(self):
		res = []
		with open(self.test_json_path,"r",encoding="utf-8") as json_input:
				lines = json_input.readlines()
				for line in lines:
					line = line.strip()
					case = json.loads(line)
					sentence = case["句子"]
					ovalue = case["要素原始值"]
					candidate_forelect = case["被告人集合"]
					res.append([sentence,ovalue,candidate_forelect])
		return json.dumps(res,ensure_ascii=False)

	def load_predict(self):
		res = []
		with open(self.predict_file_path,"r",encoding="utf-8") as json_input:
			predict = json.load(json_input)
		return predict
