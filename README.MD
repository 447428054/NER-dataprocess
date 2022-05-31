# 前言

本工程内容以clue数据格式为例，包括命名实体识别常用的数据处理操作：
- 数据分析
- 数据增强
- 实体投票

# 文件介绍

## 数据分析
代码文件：dataParser.py

trans2view：将标签可视化

readfile：读取BIO格式数据

trans2clue：将BIO格式转为clue数据格式

countShow：将记录数量的字典可视化

showInf：输入BIO格式文件，统计句子长度分布、实体长度分布、实体标签分布、实体标签记录、具体类别示例、多类别实体情况、低频实体情况

showClueInf：输入clue格式文件，统计句子长度分布、实体长度分布、实体标签分布、实体标签记录、具体类别示例、多类别实体情况、低频实体情况

## 数据增强
代码文件：dataaugment.py

getclue2view：输入为文件路径，获取字符串与可视化的映射字典

get_pos_tag：利用LTP获取分词结果

resolveTags：根据分词结果，对齐实体标签

insertAdvs：输入句子、标签、插入比例；根据词性分析结果，在动词前面插入副词

replaceAnt：输入句子、标签、插入比例；根据分词结果，完成反义词替换【考虑实体截断的情况】

replaceSim：输入句子、标签、插入比例；根据分词结果，完成同义词替换【考虑实体截断的情况】

replaceLessEntityCombine：输入句子、标签、实体列表、实体类别字典；随机替换实体列表中的字典为实体类别字典的其他实体

replaceLessEntityContextEntityProbCombine：输入句子、标签、实体列表、实体类别字典；以一定概率，替换除实体列表的其他实体，改变这些实体的上下文

augmainVx：组合上述增强方法，对数据进行增强

augdir：针对文件列表进行数据增强

genclue：根据字符列表与标签列表【BIO】，生成clue数据

## 实体投票

代码文件：voteclue.py

vote_merge：根据模型预测文件列表，通过投票筛选，完成模型融合。

logits_merge_BIO：根据模型预测logits文件列表，对概率进行平均，完成模型融合。
该方法缺点在于，logits矩阵较大，占用内存过大。

model_vote_logits：传入预测文件，以及模型列表，针对每条数据，由不同模型预测
，再将预测logits取平均。
