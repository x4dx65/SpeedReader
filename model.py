from __future__ import print_function
import string, argparse, codecs, os, json, re
from collections import Counter, OrderedDict, defaultdict
from bs4 import BeautifulSoup, Tag
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, train_test_split

from sklearn.externals import joblib
from sklearn_porter import Porter

#standard scaler
standardization = False

def preprocess(dataset_path, labels_filename, filename):
	print("[Preparing Training Data]")
	#filename = "init-html.html" or "final-dom.html"
	html_samples = list()
	readable = set(['y','v','i','g'])
	non_readable = set(['n','l','e','s','f','m'])

	with codecs.open(labels_filename, 'r', 'utf8') as f:
		for line in f:
			label = None
			if line.split('|')[-1].strip() in readable:
				label = 1
			elif line.split('|')[-1].strip() in non_readable:
				label = 0
			if label in (0, 1):
				directory = line.split('|')[1].strip()
				path = os.path.join(dataset_path, line.split('|')[0].strip())
				path = os.path.join(path, directory)
				if all(os.path.isfile(os.path.join(path, _)) for _ in (filename, 'url.txt')):
					with codecs.open(os.path.join(path, 'url.txt'),'r', 'utf8') as f_1:
						url = json.loads(f_1.read())
						if type(url) is unicode:
							url = json.loads(url)
						url = url['url']
					with codecs.open(os.path.join(path, filename), 'r', 'utf8') as f_1:
						html = f_1.read()
						soup = BeautifulSoup(html, 'html.parser')
						features = extract_features(soup, url=url)
						html_samples.append((features, label, os.path.join(path, filename)))

	temp_file = "temp_labels_"+filename.split('-')[0]+".csv"
	with codecs.open(temp_file, 'w', 'utf8') as f:
		#write header
		f.write(",".join(html_samples[1][0].keys()+['class'])+'\n')
		#write data
		for s in html_samples:
			#features
			f.write(",".join([str(s[0][_]) for _ in s[0]]))
			#label
			f.write(",%s\n"%(str(s[1])))

	data = pd.read_csv(temp_file)
	X = data.drop('class', axis=1)
	y = data['class']

	return X, y

def extract_features(soup, url):
	features = OrderedDict()
	min_num_chars = 400
	high_score_tag_counter, high_score_words_counter, high_score_text_block = list(), 0, 0

	for tag in soup.descendants:
		if isinstance(tag, Tag):
			if tag.name and tag.name.upper() in ["BLOCKQUOTE", "DL", "DIV", "OL", "P", "PRE", "TABLE", "UL", "SELECT", "ARTICLE", "SECTION"]:
				high_score_tag_counter.append(tag.name.lower())
				text = tag.find(text=True, recursive=False)
				if text and len("".join(text.split())) >= min_num_chars:
					high_score_text_block += 1
					text = text.translate({ord(k): None for k in string.punctuation})
					high_score_words_counter += len(text.split())

	features['images'] = len(soup.find_all('img'))
	features['anchors'] = len(soup.find_all('a'))
	features['scripts'] = len(soup.find_all('script'))
	features['text_blocks'] = high_score_text_block
	features['words'] = high_score_words_counter
	high_score_tag_counter = Counter(high_score_tag_counter)
	for _ in ["BLOCKQUOTE", "DL", "DIV", "OL", "P", "PRE", "TABLE", "UL", "SELECT", "ARTICLE", "SECTION"]:
		if _.lower() in high_score_tag_counter:
			features[_.lower()] = high_score_tag_counter[_.lower()]
		else:
			features[_.lower()] = 0

	if url:
		features['url_depth'] = len(url.split('://')[1].split('/')[1:])
	features['amphtml'] = 1 if soup.find('link', rel='amphtml') else 0
	features['fb_pages'] = 1 if soup.find('meta', property='fb:pages') else 0
	features['og_article'] = 1 if soup.find('meta', attrs={'property': 'og:type', 'content': 'article'}) else 0
	schemaOrgRegex = re.compile('http(s)?:\/\/schema.org\/(Article|NewsArticle|APIReference)')
	if schemaOrgRegex.search(soup.text):
		features['schema_org_article'] = 1
	else:
		features['schema_org_article'] = 0

	return features

def classifier(X, y, standardization):
	print("[Training the Model]")
	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state=10)
	if standardization:
		sc = StandardScaler()
		X_train = sc.fit_transform(X_train)
		X_test = sc.transform(X_test)

	model = RandomForestClassifier(n_estimators=50, random_state=10, n_jobs=4, oob_score=True)
	model.fit(X_train, y_train)

	if standardization:
		return sc, model
	else:
		return model

def cross_validation(model, X, y):
	print("[Classification Report]")
	score = cross_validate(model, X=X, y=y, scoring=['precision', 'recall', 'f1', 'accuracy'], cv=10, return_train_score=False)
	print("Precision: %0.2f (+/- %0.2f)" % (np.mean(score['test_precision']), np.std(score['test_precision']) * 2))
	print("Recall: %0.2f (+/- %0.2f)" % (np.mean(score['test_recall']), np.std(score['test_precision']) * 2))
	print("F-1: %0.2f (+/- %0.2f)" % (np.mean(score['test_f1']), np.std(score['test_f1']) * 2))
	print("Accuracy: %0.2f (+/- %0.2f)" % (np.mean(score['test_accuracy']), np.std(score['test_accuracy']) * 2))

def save_model(model, filename):
	joblib.dump(model, filename, compress=0)

def convert_to_c_model(python_model, output_c_model):
	print("[Convert to C]")
	porter = Porter(python_model, language='c')
	output_c = porter.export(embed_data=True, compress=0)
	with open(output_c_model, 'w') as f:
		f.write(output_c)


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('--datapath', help="/path/to/Dataset", required=True)
	parser.add_argument('--labels', help="/path/to/labels.csv", required=True)
	parser.add_argument('--filetype', help="init-html.html or final-dom.html", required=True)
	parser.add_argument('--cmodel', help="C model name to be saved")

	args = parser.parse_args()

	X, y = preprocess(args.datapath, args.labels, args.filetype)
	model = classifier(X, y, standardization)
	cross_validation(model, X, y)
	if args.cmodel:
		convert_to_c_model(model, args.cmodel)







