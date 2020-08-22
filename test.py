import os
import sys
import json

sys.path.append(".")
sys.path.append("..")

from predictor import Predictor


if __name__ == '__main__':
	records = []
	# '''
	with open("./data/train.txt","r",encoding="UTF-8") as f:
		for line in f.readlines():
			record = json.loads(line)
			# record = line
			records.append(record)
	# '''
	p = Predictor()
	result = p.predict(records=records)

	with open("./data/test_answer_baseline.txt","w",encoding="UTF-8") as f:
		for _result in result:
			line = ""
			for __bgr in _result:
				if line != "":
					line = line + ","
				line = line + __bgr
			f.write(line)
			f.write("\n")


