import os
import re
import jieba

def load_stopwords(path):
    # 加载停用词
    with open(path, 'r', encoding='utf-8') as file:
        stopwords = set(file.read().split())
    return stopwords

def preprocess_text(text, stopwords):
    # 文本预处理并分字，返回token列表
    # 替换掉非中文字符
    text = re.sub(r'[^\u4e00-\u9fa5]', '', text)
    # 分字
    tokens = list(text)  # 将每个中文字符单独作为一个token
    # 去除停用词
    return [token for token in tokens if token not in stopwords]

def extract_paragraphs(tokens, tokens_per_paragraph, label, num_paragraphs):
    # 将token列表分割成指定数量的段落并附带标签
    paragraphs = []
    total_tokens = len(tokens)
    step = max(total_tokens // num_paragraphs, tokens_per_paragraph)  # 确保每段足够的token数量
    for i in range(0, num_paragraphs):
        start_index = i * step
        if start_index + tokens_per_paragraph > total_tokens:
            break  # 如果剩余的tokens不足，停止提取
        paragraph = ' '.join(tokens[start_index:start_index + tokens_per_paragraph])
        paragraphs.append((paragraph, label))
    return paragraphs

def save_paragraphs(paragraphs, filepath):
    """将段落及其标签保存到文件"""
    with open(filepath, 'w', encoding='utf-8') as file:
        for paragraph, label in paragraphs:
            file.write(f"{label}: {paragraph}\n\n")  # 每个段落前标注标签

def main():
    base_dir = 'jyxstxtqj_downcc.com'
    novel_titles = ['三十三剑客图.txt', '书剑恩仇录.txt', '侠客行.txt', '倚天屠龙记.txt', 
                    '天龙八部.txt', '射雕英雄传.txt', '白马啸西风.txt', '碧血剑.txt',
                    '神雕侠侣.txt', '笑傲江湖.txt', '越女剑.txt', '连城诀.txt',
                    '雪山飞狐.txt', '飞狐外传.txt', '鸳鸯刀.txt', '鹿鼎记.txt']
    stopwords = load_stopwords('cn_stopwords.txt')  # 加载停用词表
    tokens_per_paragraph = 500  # 可以调整每段的token数量
    paragraphs_per_novel = 100  # 每部小说提取的段落数量

    all_paragraphs = []

    for title in novel_titles:
        filepath = os.path.join(base_dir, title)
        label = title.replace('.txt', '')  # 从文件名提取标签
        with open(filepath, 'r', encoding='gb18030') as file:
            text = file.read()
        
        tokens = preprocess_text(text, stopwords)  # 预处理文本并分词
        paragraphs = extract_paragraphs(tokens, tokens_per_paragraph, label, paragraphs_per_novel)  # 提取段落
        all_paragraphs.extend(paragraphs)

    save_paragraphs(all_paragraphs, 'extracted_paragraphs.txt')  # 保存段落到文件
    print(f"Total paragraphs extracted: {len(all_paragraphs)}")

if __name__ == '__main__':
    main()
