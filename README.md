# scorecard

[[中文文档]](https://blog.csdn.net/m0_65814643/article/details/135672386?spm=1001.2014.3001.5502)

This is a project that use scorecard to assess the credit risk of the loan users. 

**woe.py** and **iv.py** are for variable selection of data. **scorecard.py** is the main code of socrecard model. After variable selection, run scorecard.py on the new data. You can also run **randomforest.py** to compare the performance of Random Forest and Scorecard model in this project. Data is not available due to privacy issues. However, the code works for all structured data like this:

| id | feature1 | feature2 | feature3 |
| ---- | ---- | ---- | ---- |
| 00001 | xxx | xxx | xxx |
| ..... | ... | ... | ... |
| 99999 | xxx | xxx | xxx |  

## Enviroment

Python==3.8

sklearn==1.2.2

scorecardbundle

<br/>

support installation via pip:

`pip install --upgrade scorecardbundle`
