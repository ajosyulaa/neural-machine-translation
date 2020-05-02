import re
from nltk.tokenize import word_tokenize
filepath = "dev.en"
with open(filepath, "r") as fp:  
	for line in fp:
		s=" "
		line = s.join(word_tokenize(line))
		print(line)
		# if "," in line:
		# 	line = re.sub(r"[,]"," , ", line)
		# if "." in line:
		# 	line = re.sub(r"[.]"," . ", line)
		# if ":" in line:
		# 	line = re.sub(r"[:]"," : ", line)
		# if ";" in line:
		# 	line = re.sub(r"[;]"," ; ", line)
		# if "-" in line:
		# 	line = re.sub(r"[-]"," - ", line)
		# if "?" in line:
		# 	line = re.sub(r"[?]", " ? ", line)
		# if "\"" in line:
		# 	line = re.sub(r"[\"]", " \" ", line)
		# if "!" in line:
		# 	line = re.sub(r"[?]", " ! ", line)
		# line = re.sub(r"[\s*]", " ", line)
		# print(line)

