import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

name_dataset_train = 'train.csv'
name_dataset_test = 'test.csv'
name_sample_submission = 'sample_submission.csv'
df_train = pd.read_csv(name_dataset_train)
df_test = pd.read_csv(name_dataset_test)
df_sample = pd.read_csv(name_sample_submission)


def data_info(df, name='датасет'):
    print(f'\n{name}')
    print('\nРазмерность')
    print(df.shape)
    print('\nПервые 10 строк')
    print(df.head(10))
    print('\nПоследние 10 строк')
    print(df.tail(10))
    print('\nОбщая информация')
    print(df.info())
    print('\nЧисловые статистики')
    print(df.describe())


data_info(df_train, 'Тренировочный датасет')
data_info(df_test, 'Тестовый датасет')
data_info(df_sample, 'Пример')

bsmt_qual_train = df_train['BsmtQual'].copy()
bsmt_qual_test = df_test['BsmtQual'].copy()
bsmt_qual_train.fillna('not_data', inplace=True)
bsmt_qual_test.fillna('not_data', inplace=True)

bar_container = plt.bar(bsmt_qual_train.value_counts().index, bsmt_qual_train.value_counts(), color='skyblue', edgecolor='black', )
plt.title('Распределение BsmtQual для тренировочной выборки')
plt.xlabel('Оценка высоты')
plt.ylabel('Количество значений')
plt.grid()
plt.bar_label(bar_container)
plt.show()

bar_container = plt.bar(bsmt_qual_test.value_counts().index, bsmt_qual_test.value_counts(), color='skyblue', edgecolor='black')
plt.title('Распределение BsmtQual для тестовой выборки')
plt.xlabel('Оценка высоты')
plt.ylabel('Количество значений')
plt.bar_label(bar_container)
plt.grid()
plt.show()

plt.hist(df_train['GarageYrBlt'], color='skyblue', edgecolor='black')
plt.title('Распределение GarageYrBlt для тренировочной выборки')
plt.xlabel('Дата постройки гаража')
plt.ylabel('Количество значений')
plt.grid()
plt.show()

plt.hist(df_test['GarageYrBlt'], color='skyblue', edgecolor='black')
plt.title('Распределение GarageYrBlt для тестовой выборки')
plt.xlabel('Дата постройки гаража')
plt.ylabel('Количество значений')
plt.grid()
plt.show()

df_test.loc[df_test['GarageYrBlt'] > 2010, 'GarageYrBlt'] = None
print(df_test[df_test['GarageYrBlt'] > 2010])

df_train_corr = df_train.copy()
for column in df_train_corr.columns:
    if df_train_corr[column].isnull().any():
        df_train_corr[f'{column}_missing'] = df_train_corr[column].isnull().astype(int)
label_encoder = {}
for column in df_train_corr.select_dtypes(include=['object']).columns:
    label_encoder[column] = LabelEncoder()
    df_train_corr[column] = label_encoder[column].fit_transform(df_train_corr[column])
full_correlation_matrix = df_train_corr.corr()

plt.figure(figsize=(16, 14))
plt.imshow(full_correlation_matrix, cmap="coolwarm", interpolation='none')
plt.colorbar()
plt.xticks(range(len(full_correlation_matrix.columns)), full_correlation_matrix.columns, rotation=90)
plt.yticks(range(len(full_correlation_matrix.columns)), full_correlation_matrix.columns)

# Добавление значений корреляции на график
for i in range(len(full_correlation_matrix.columns)):
    for j in range(len(full_correlation_matrix.columns)):
        text = plt.text(j, i, f"{full_correlation_matrix.iloc[i, j]:.2f}",
                        ha="center", va="center", color="black")

plt.title("Корреляционная матрица для всех параметров")
plt.show()

'''
Обратная трансформация
for column in label_encoder.keys():
    df_train_corr[column] = label_encoder[column].inverse_transform(df_train_corr[column])
'''


def show_none_column(df, none_column_name, check_column_name):
    df_none = df.loc[df[none_column_name].isnull(), [none_column_name, check_column_name]]
    df_none_min = df_none[check_column_name].min()
    df_none_max = df_none[check_column_name].max()
    df_none_mean = df_none[check_column_name].mean()
    f, ax = plt.subplots()
    plt.scatter(range(len(df_none)), df_none[check_column_name])
    plt.title(f'Значения {check_column_name}, при неизвестной {none_column_name}')
    plt.ylabel(check_column_name)
    plt.text(0.01, 0.99, f'min = {df_none_min}\nmax = {df_none_max}\nmean = {df_none_mean}',
             ha='left', va='top', transform=ax.transAxes, bbox={'facecolor': 'red', 'alpha': 0.5})
    plt.grid()
    plt.show()

show_none_column(df_train, 'BsmtQual', 'TotalBsmtSF')
print(f'Количество значений отличных от none при TotalBsmtSF = 0 : ', end='')
print(len(df_train[df_train['TotalBsmtSF'] == 0]) - len(df_train[df_train['BsmtQual'].isnull()]))
show_none_column(df_test, 'BsmtQual', 'TotalBsmtSF')
print(f'\nКоличество значений отличных от none при TotalBsmtSF = 0 : ', end='')
print(len(df_test[df_test['TotalBsmtSF'] == 0]) - len(df_test[df_test['BsmtQual'].isnull()]))

df_train.loc[df_train['TotalBsmtSF'] == 0, 'BsmtQual'] = 'NA'
df_test.loc[df_test['TotalBsmtSF'] == 0, 'BsmtQual'] = 'NA'

show_none_column(df_train, 'GarageYrBlt', 'GarageCars')
print(f'Количество значений отличных от none при GarageCars = 0 : ', end='')
print(len(df_train[df_train['GarageCars'] == 0]) - len(df_train[df_train['GarageYrBlt'].isnull()]))
show_none_column(df_test, 'GarageYrBlt', 'GarageCars')
print(f'Количество значений отличных от none при GarageCars = 0 : ', end='')
print(len(df_test[df_test['GarageCars'] == 0]) - len(df_test[df_test['GarageYrBlt'].isnull()]))

df_train.loc[df_train['GarageCars'] == 0, 'GarageYrBlt'] = 0
df_train['HaveGarage'] = 0
df_train.loc[df_train['GarageCars'] == 0, 'HaveGarage'] = 1

df_test.loc[df_test['GarageCars'] == 0, 'GarageYrBlt'] = 0
df_test['HaveGarage'] = 0
df_test.loc[df_test['GarageCars'] == 0, 'HaveGarage'] = 1

print(f'Количество пропущенных значений в тренировочном датасете: {df_train.isnull().sum().sum()}')
print(f'Количество пропущенных значений в тестовом датасете: {df_test.isnull().sum().sum()}')

print(df_test[df_test.isna().any(axis=1)])
df_test.loc[df_test['GarageYrBlt'].isnull(), 'GarageYrBlt'] = df_test.loc[df_test['GarageYrBlt'] != 0, 'GarageYrBlt'].mean()
df_test.loc[df_test['BsmtQual'].isnull(), 'BsmtQual'] = 'TA'

total_bsmt_mean = df_test['TotalBsmtSF'].mean()
if total_bsmt_mean == 0:
    df_test.loc[df_test['TotalBsmtSF'].isnull(), 'BsmtQual'] = 'NA'
df_test.loc[df_test['TotalBsmtSF'].isnull(), 'TotalBsmtSF'] = total_bsmt_mean
df_test.loc[df_test['BsmtFinSF1'].isnull(), 'BsmtFinSF1'] = df_test['BsmtFinSF1'].mean()
df_test.loc[df_test['KitchenQual'].isnull(), 'KitchenQual'] = df_test['KitchenQual'].value_counts().index[0]
garage_cars_most_popular = df_test['GarageCars'].value_counts().index[0]
if garage_cars_most_popular == 0:
    df_test.loc[df_test['GarageCars'].isnull(), 'GarageYrBlt'] = garage_cars_most_popular
df_test.loc[df_test['GarageCars'].isnull(), 'GarageCars'] = garage_cars_most_popular
print(f'Количество пропущенных значений в тестовом датасете: {df_test.isnull().sum().sum()}')

categorical_features_list = df_train.columns[df_train.dtypes == 'object']


one_hot_encoder = OneHotEncoder(sparse_output=False)
one_hot_encoder.fit(df_train[categorical_features_list])
encoded_train = pd.DataFrame(one_hot_encoder.transform(df_train[categorical_features_list]), columns=one_hot_encoder.get_feature_names_out(categorical_features_list))
encoded_test = pd.DataFrame(one_hot_encoder.transform(df_test[categorical_features_list]), columns=one_hot_encoder.get_feature_names_out(categorical_features_list))
df_train_encoded = pd.concat([df_train, encoded_train], axis=1)
df_train_encoded.drop(categorical_features_list, axis=1, inplace=True)
df_test_encoded = pd.concat([df_test, encoded_test], axis=1)
df_test_encoded.drop(categorical_features_list, axis=1, inplace=True)

y = df_train_encoded['SalePrice']
df_train_encoded.drop('SalePrice', axis=1, inplace=True)


def min_max_norm(df, num_feature):
    garage_not_zero = df.loc[df_train_encoded['GarageYrBlt'] != 0, 'GarageYrBlt']
    garage_not_zero = (garage_not_zero - garage_not_zero.min()) / (garage_not_zero.max() - garage_not_zero.min())
    df.loc[df_train_encoded['GarageYrBlt'] != 0, 'GarageYrBlt'] = garage_not_zero
    num_feature = num_feature.drop('GarageYrBlt')
    df[num_feature] = ((df[num_feature] - df[num_feature].min()) /
                       (df[num_feature].max() - df[num_feature].min()))
    return df


numeric_feature_list = df_train.columns[df_train.dtypes != 'object']
numeric_feature_list = numeric_feature_list.drop('SalePrice')
df_train_encoded = min_max_norm(df_train_encoded, numeric_feature_list)
df_test_encoded = min_max_norm(df_test_encoded, numeric_feature_list)

x_train, x_test, y_train, y_test = train_test_split(df_train_encoded, y, test_size=0.2, random_state=7)
print(f'Минимальное значение цены на жилье: {y.min()}')
print(f'Максимальное значение цены на жилье: {y.max()}')
print(f'Средние значение цены на жилье: {y.mean()}')
print(f'Медиана значений цены на жилье: {y.median()}')


def show_result(model, is_linear=False, is_rf=False, model_name='model', show_test=True, show_train=False):
    # Реализуем цикл для анализа тестовой и тренировочной выборки
    show_data = []
    if show_test:
        show_data.append('test')
    if show_train:
        show_data.append('train')
    for show in show_data:
        if show == 'test':
            y_pred = model.predict(x_test)
            y_real = y_test
        else:
            y_pred = model.predict(x_train)
            y_real = y_train
        # Рассчитаем метрики и выведем результат
        mse = mean_squared_error(y_real, y_pred)
        rmse = np.sqrt(mse)
        mape = np.mean(np.abs((y_real - y_pred) / y_real)) * 100
        print(f'\nИмя модели: {model_name}')
        print(f'Выборка : {show}')
        print(f'RMSE: {round(rmse, 0)}')
        print(f'MAPE: {round(mape, 2)} %\n')
        print('Графики сравнения актуальных и предсказанных значений')
        # Построим графики значений предсказаний
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.scatter(y_real, y_pred, alpha=0.5, c='blue', label='Точки')
        plt.plot([min(y_real), max(y_real)], [min(y_real), max(y_real)], color='red', linestyle='--',
                 label='Идеальная кривая')
        plt.xlabel('Фактическая цена')
        plt.ylabel('Предсказанная цена')
        plt.title(f'Зависимость предсказанной цены от фактической ({show})')
        plt.legend()
        residuals = y_real - y_pred
        plt.subplot(1, 2, 2)
        plt.scatter(y_pred, residuals, alpha=0.5, c='blue', label='Точки')
        plt.axhline(0, color='red', linestyle='--', label='Идеальная кривая' )
        plt.xlabel('Предсказанная цена')
        plt.ylabel('Разница в ценах')
        plt.title(f'Зависимость разницы в ценах от предсказанной цены ({show})')
        plt.tight_layout()
        plt.show()
        # Построим график распределения разницы
        plt.figure(figsize=(14, 6))
        plt.subplot(1, 2, 1)
        plt.hist(residuals, bins=50, color='skyblue', edgecolor='black')
        plt.xlabel('Разница')
        plt.ylabel('Частота')
        plt.title(f'Распределение разницы между предсказанным и фактическим значением цены ({show})')
        plt.show()
        # Посмотрим на коэффициенты линейной модели
        if is_linear or is_rf:
            # Создадим датафрейм коэффициентов
            if is_linear:
                coefficients = pd.DataFrame({
                    'Feature': x_train.columns,
                    'Coefficient': model.coef_
                })
            else:
                coefficients = pd.DataFrame({
                    'Feature': x_train.columns,
                    'Coefficient': model.feature_importances_
                })
            # Отсортируем значение
            coefficients['abs_Coefficient'] = coefficients['Coefficient'].abs()
            coefficients = coefficients.sort_values(by='abs_Coefficient', ascending=False)
            # Визуализируем результат
            plt.figure(figsize=(12, 8))
            plt.barh(coefficients['Feature'], coefficients['Coefficient'], color='skyblue')
            plt.xlabel('Значение коэффициента')
            plt.ylabel('Признак')
            plt.title(f'Коэффициенты ({show})')
            plt.gca().invert_yaxis()
            plt.show()


model_linear = LinearRegression()
model_linear.fit(x_train, y_train)
show_result(model_linear, is_linear=True, model_name='Линейная регрессия', show_train=True, show_test=True)

model_ridge = Ridge(alpha=1)
model_ridge.fit(x_train, y_train)
show_result(model_ridge, is_linear=True, model_name='Линейная регрессия с L2 регуляризацией', show_train=True,
            show_test=True)

model_lasso = Lasso(alpha=1)
model_lasso.fit(x_train, y_train)
show_result(model_lasso, is_linear=True, model_name='Линейная регрессия с L1 регуляризацией', show_train=True,
            show_test=True)

model_rf = RandomForestRegressor(n_estimators=100, random_state=7)
model_rf.fit(x_train, y_train)
show_result(model_rf, is_linear=False, is_rf=True, model_name='Случайный лес', show_train=True,
            show_test=True)

model_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=7)
model_rf.fit(x_train, y_train)
show_result(model_rf, is_linear=False, is_rf=True, model_name='Случайный лес', show_train=True,
            show_test=True)

model_gb = GradientBoostingRegressor(n_estimators=500, learning_rate=0.04, max_depth=3, random_state=7)
model_gb.fit(x_train, y_train)
show_result(model_gb, is_linear=False, is_rf=True, model_name='Градиентный бустинг', show_train=True,
            show_test=True)

y_test_pred = model_gb.predict(df_test_encoded)
for i in range(len(y_test_pred)):
    y_test_pred[i] = round(y_test_pred[i], 6)
submission = df_sample.copy()
submission['SalePrice'] = y_test_pred
data_info(submission, 'submission')
submission.to_csv('submission.csv')
