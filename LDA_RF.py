import pandas as pd
import numpy as np
from gensim.models import LdaModel
from gensim.corpora import Dictionary
from gensim.matutils import sparse2full
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from scipy.stats import randint as sp_randint
import logging
import pyLDAvis.gensim_models as gensimvis
import pyLDAvis

# 设置日志
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# 加载数据
data = pd.read_csv('extracted_paragraphs.txt', header=None, sep=":", engine='python', on_bad_lines='warn')
data.columns = ['label', 'text']

# 文本预处理
texts = [doc.split() for doc in data['text']]

# 创建字典和语料库
dictionary = Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]

# 训练LDA模型
num_topics = 10
lda = LdaModel(corpus=corpus, id2word=dictionary, num_topics=num_topics, passes=10)

# 特征提取
features = np.array([sparse2full(lda[doc], num_topics) for doc in corpus])

# 随机森林分类器
classifier = RandomForestClassifier(random_state=42)

# 参数分布设置
param_dist = {"max_depth": [3, None],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "n_estimators": sp_randint(100, 500)}

# 使用StratifiedKFold进行交叉验证
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# 设置RandomizedSearchCV
random_search = RandomizedSearchCV(classifier, param_distributions=param_dist, n_iter=10, cv=cv, random_state=42)

# 执行随机搜索
random_search.fit(features, data['label'])

# 输出最佳参数和准确率
print("Best parameters:", random_search.best_params_)
print("Best cross-validation score: {:.3f}".format(random_search.best_score_))

# 可视化LDA模型
vis = gensimvis.prepare(lda, corpus, dictionary)
pyLDAvis.save_html(vis, 'lda_visualization.html')