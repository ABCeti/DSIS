# DSIS
Размер исходной выборки 143 Мб, после сжатия она стала весить 65,3 Мб. Так как на git можно залить файлы весом не более 25 Мб прикладываю ссылку на [Google Drive](https://drive.google.com/drive/folders/1Qs84ZhqFmcwx2sa-XYe86rEL_BKoUWAS?usp=sharing), где лежат исходники.

## О выборке

В качестве выборки в рамках домашнего задания по курсу DSIS была выбрана [Credit Card Fraud](https://www.kaggle.com/code/samkirkiles/credit-card-fraud/input).

Датасет для обнаружения мошенничества. Он содержит 284 807 транзакций с банковскими картами, выполненными европейцами в сентябре 2013 года. Для каждой транзакции представлено: 28 компонент, извлеченных из исходных данных, временная отметка, количество денежных средств. У транзакции 2 метки: 1 для мошеннических и 0 для легитимных (нормальных) операций.

##
В качестве библиотеки для анализа данных используется pandas. В качестве инструментов визуализации используется matplotlib, seaborn.

## Машинное обучение
Как доп. решила обучить модель для выявления мошеннических операций. Обучала с помощью алгоритмов случайного леса, используя sklearn.

## Ход работы

### Установка

Для запуска проекта вам понадобится установить несколько зависимостей. Воспользуйтесь следующими командами:

```
import graphviz
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

```

### Использование

1. Загрузка данных

   Загрузите данные о кредитных картах с помощью следующей команды:

   ```python
   dat = pd.read_csv('creditcard.csv')
   ```

2. Предварительный анализ данных

   Проведите предварительный анализ данных, включающий проверку на наличие пустых значений, а также визуализацию данных с помощью графиков:

   ```python
   dat.count() # количество записей
   dat.columns # названия столбцов
   dat.isna() # проверка на наличие пустых значений
   sns.countplot(x='Class', data=dat) # столбчатая диаграмма для количества мошеннических операций
   sns.scatterplot(data=dat, x="Amount", y="Class") # график зависимости мошеннических операций от суммы
   sns.scatterplot(data=dat_class_0, x='Time', y='Amount', alpha=0.6) # график зависимости мошеннических операций от времени для немошеннических операций
   sns.scatterplot(data=dat_class_1, x='Time', y='Amount', alpha=0.6) # график зависимости мошеннических операций от времени для мошеннических операций
   sns.pairplot(dat_class_0, vars=features, diag_kind="kde") #матрицы графиков рассеяния для признаков v1-v10, класс 0
   sns.pairplot(dat_class_1, vars=features, diag_kind="kde") #матрицы графиков рассеяния для признаков v1-v10, класс 1
   ```

3. Обучение модели

   Обучите модель случайного леса на данных и оцените точность предсказаний на тестовой выборке:

   ```python
   # подготовка данных
   X = data[:, 0:-1]
   y = data[:, -1].reshape(size, 1)

   # разделяем на обучающие и тестовые, первая половина - обучающая, вторая - тестовая
   X_train, X_test = np.split(X, [size // 2])
   y_train, y_test = np.split(y, [size // 2])

   # создание модели и обучение
   clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1)
   clf.fit(X_train, y_train)

   # оценка точности предсказаний
   pred = clf.predict(X_test)
   ```
4. Переобучение модели

   ```python
   def proc(pred):
   mist_leg = ((y_test == 0) * pred).sum()
   all_leg = (y_test == 0).sum()
   proc_leg = 1 - mist_leg/all_leg
   mist_ill = (y_test*(pred == 0)).sum()
   all_ill = y_test.sum()
   proc_ill = 1 - mist_ill/all_ill
   return (proc_leg, proc_ill)

   dep = 6
   estim = 6
   feat = np.shape(X_train)[1] + 1
   siz = np.shape(y_test)[0]
   ans = [0, 0, 0, 0, 0]
   
   for a in range(1, dep):
     for b in range(1, estim):
       for c in range(1, feat):
         clf = RandomForestClassifier(max_depth=a,# максимальная глубина дерева
                                n_estimators=b,# число деревьев в лесу
                                max_features=c)# максимальное число признаков для каждого дерева
         clf.fit(X_train, y_train) # обучаем
         y_pred = clf.predict(X_test)
         y_pred = y_pred.reshape(siz, 1)
         leg_pr, ill_pr = proc(y_pred)
         print(leg_pr, ill_pr, a, b, c)
         if leg_pr + ill_pr > ans[0] + ans[1]:
           ans = [leg_pr, ill_pr, a, b, c]
   ```
   
## Заключение
Был проведен предварительный анализ данных, включающий проверку на наличие пустых значений и визуализацию данных с помощью графиков.
Были использованы популярные инструменты для работы с данными, такие как pandas, matplotlib и seaborn.

Дополнительно была обучена модель для выявления мошеннических операций, используя алгоритм случайного леса из библиотеки scikit-learn (sklearn). Для этого данные были подготовлены, разделены на обучающую и тестовую выборки, и затем модель была обучена на обучающих данных. Точность предсказаний модели была оценена на тестовой выборке.
