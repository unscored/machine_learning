import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from IPython.display import display

def age_cat(years):
    if years <= 20:
        return '0-20'
    elif years > 20 and years <= 30:
        return '20-30'
    elif years > 30 and years <= 40:
        return '30-40'
    elif years > 40 and years <= 50:
        return '40-50'
    elif years > 50 and years <= 60:
        return '50-60'
    elif years > 60 and years <= 70:
        return '60-70'
    elif years > 70:
        return '70+'

# Одномірний (категоріальний vs цільовий) та двомірний (категоріальний vs категоріальний) аналіз (гістограма) - категоріальні стовпчики
def bi_cat_countplot(df, column, hue_column):
    unique_hue_values = df[hue_column].unique()
    fig, axes = plt.subplots(nrows=1, ncols=2)
    fig.set_size_inches(14,6)

    pltname = f'Нормалізований розподіл значень за категорією: {column}'
    proportions = df.groupby(hue_column)[column].value_counts(normalize=True)
    proportions = (proportions*100).round(2)
    ax = proportions.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[0], title=pltname)

    # анотація значень в барплоті
    for container in ax.containers:
        ax.bar_label(container, fmt='{:,.1f}%')

    pltname = f'Кількість даних за категорією: {column}'
    counts = df.groupby(hue_column)[column].value_counts()
    ax = counts.unstack(hue_column).sort_values(
        by=unique_hue_values[0], ascending=False
        ).plot.bar(ax=axes[1], title=pltname)

    for container in ax.containers:
      ax.bar_label(container)


def uni_cat_target_compare(df, column):
    bi_cat_countplot(df, column, hue_column='TARGET' )


def bi_countplot_target(df0, df1, column, hue_column):
  pltname = 'Клієнт зі складнощами щодо платності'
  print(pltname.upper())
  bi_cat_countplot(df1, column, hue_column)
  plt.show()

  pltname = 'Клієнти зі своєчасними платежами'
  print(pltname.upper())
  bi_cat_countplot(df0, column, hue_column)
  plt.show()


# Обчислення мінімального та максимального діапазону відхилень для числових стовпців
def outlier_range(dataset,column):
    Q1 = dataset[column].quantile(0.25)
    Q3 = dataset[column].quantile(0.75)
    IQR = Q3 - Q1
    Min_value = (Q1 - 1.5 * IQR)
    Max_value = (Q3 + 1.5 * IQR)
    return Max_value

# Робота з викидами для числових стовпців
def dist_box(dataset, column):
    with warnings.catch_warnings():
      warnings.simplefilter("ignore")

      plt.figure(figsize=(16,6))

      plt.subplot(1,2,1)
      sns.distplot(dataset[column], color = 'purple')
      pltname = 'Графік розподілу для ' + column
      plt.ticklabel_format(style='plain', axis='x')
      plt.title(pltname)

      plt.subplot(1,2,2)
      red_diamond = dict(markerfacecolor='r', marker='D')
      sns.boxplot(y = column, data = dataset, flierprops = red_diamond)
      pltname = 'Боксплот для ' + column
      plt.title(pltname)

      plt.show()

def pair_cor_analyze(dataFrame0, dataFrame1, column_one_name, column_two_name):
  max_value1_column_one_name = outlier_range(dataFrame1, column_one_name)
  max_value1_column_two_name = outlier_range(dataFrame1, column_two_name)

  max_value0_column_one_name = outlier_range(dataFrame0,column_one_name)
  max_value0_column_two_name = outlier_range(dataFrame0,column_two_name)

  print(f"--- Статистика для колонки '{column_one_name}' ---")
  print(f"df1 (Payment difficulties): Max = {max_value1_column_one_name}")
  print(f"df0 (On-Time Payments): Max = {max_value0_column_one_name}")

  print(f"--- Статистика для колонки '{column_two_name}' ---")
  print(f"df1 (Payment difficulties): Max = {max_value1_column_two_name}")
  print(f"df0 (On-Time Payments): Max = {max_value0_column_two_name}")

  plt.figure(figsize = (14, 6))

  plt.subplot(1,2,1)
  plt.title('Payment difficulties')
  sns.scatterplot(x = dataFrame1[dataFrame1[column_one_name] < max_value1_column_one_name][column_one_name],
                  y = dataFrame1[dataFrame1[column_two_name] < max_value1_column_two_name][column_two_name], data = dataFrame1)
  plt.ticklabel_format(style='plain', axis='x')
  plt.ticklabel_format(style='plain', axis='y')

  plt.subplot(1,2,2)
  plt.title('On-Time Payments')
  sns.scatterplot(x = dataFrame0[dataFrame0[column_one_name] < max_value0_column_one_name][column_one_name],
                    y = dataFrame0[dataFrame0[column_two_name] < max_value0_column_two_name][column_two_name], data = dataFrame0)
  plt.ticklabel_format(style='plain', axis='x')
  plt.ticklabel_format(style='plain', axis='y')

  plt.tight_layout(pad = 4)
  plt.show()

  


class OutlierAnalyzer:
    def __init__(self, column: str, df0, df1):
        self.column = column
        self.df0 = df0
        self.df1 = df1

        # Расчеты для df0
        self.Q1_0 = self.df0[self.column].quantile(0.25)
        self.Q3_0 = self.df0[self.column].quantile(0.75)
        self.IQR_0 = self.Q3_0 - self.Q1_0
        self.Min_value0 = self.Q1_0 - 1.5 * self.IQR_0
        self.Max_value0 = self.Q3_0 + 1.5 * self.IQR_0

        # Расчеты для df1
        self.Q1_1 = self.df1[self.column].quantile(0.25)
        self.Q3_1 = self.df1[self.column].quantile(0.75)
        self.IQR_1 = self.Q3_1 - self.Q1_1
        self.Min_value1 = self.Q1_1 - 1.5 * self.IQR_1
        self.Max_value1 = self.Q3_1 + 1.5 * self.IQR_1

    def print_stats(self):
        print(f"--- Статистика для колонки '{self.column}' ---")
        print(f"df1 (Payment difficulties): IQR = {self.IQR_1}, Min = {self.Min_value1}, Max = {self.Max_value1}")
        print(f"df0 (On-Time Payments):     IQR = {self.IQR_0}, Min = {self.Min_value0}, Max = {self.Max_value0}")

    def kde_no_outliers(self):
        plt.figure(figsize=(14, 6))
        
        sns.kdeplot(
            self.df1[self.df1[self.column] <= self.Max_value1][self.column],
            label='Payment difficulties'
        )
        sns.kdeplot(
            self.df0[self.df0[self.column] <= self.Max_value0][self.column],
            label='On-Time Payments'
        )
        
        plt.ticklabel_format(style='plain', axis='x')
        plt.xticks(rotation=45)
        plt.legend()
        plt.show()

def draw_boxplot(df, categorical, continuous, max_continuous, title, hue_column, subplot_position):
    """
    Малює блок-діаграму для заданого DataFrame, категоріальної та неперервної змінної.
    """
    plt.subplot(1, 2, subplot_position)
    plt.title(title)
    red_diamond = dict(markerfacecolor='r', marker='D')
    sns.boxplot(x=categorical,
                y=df[df[continuous] < max_continuous][continuous],
                data=df,
                flierprops=red_diamond,
                order=sorted(df[categorical].unique(), reverse=True),
                hue=hue_column, hue_order=sorted(df[hue_column].unique(), reverse=True))
    plt.ticklabel_format(style='plain', axis='y')
    plt.xticks(rotation=90)

def bi_boxplot(df1, df0, categorical, continuous, max_continuous1, max_continuous0, hue_column):
    """
    Створює паралельні блок-діаграми для двох груп, визначених у наборі даних, на основі
    категоріальної та неперервної змінної, виділяючи відмінності за допомогою відтінків.
    """
    plt.figure(figsize=(16, 10))

    # Графік для першо групи "Труднощі з платежами" (Payment Difficulties)
    draw_boxplot(df1, categorical, continuous, max_continuous1, 'Payment Difficulties', hue_column, 1)

    # Графік для другої групи "Вчасні оплати" (On-Time Payments)
    draw_boxplot(df0, categorical, continuous, max_continuous0, 'On-Time Payments', hue_column, 2)

    plt.tight_layout(pad=4)
    plt.show()

def numeric_vs_categorical_analysis(df0, df1, column_1, column_2, column_3):
    """
    Виконує повний цикл аналізу зв'язку між числовою змінною (column_1) 
    та двома категоріальними змінними (column_2, column_3) для двох груп клієнтів:
    1. Обчислює верхню межу викидів (outlier_range) для df1 та df0.
    2. Виводить зведену статистику (.describe()) для кожної групи.
    3. Будує парні діаграми розмаху (bi_boxplot) без викидів.
    """
    # 1. Пошук викидів для обох груп
    max_value1 = outlier_range(df1, column_1)
    max_value0 = outlier_range(df0, column_1)
    
    print(f"Верхня межа викидів для '{column_1}':")
    print(f"  df1 (Payment Difficulties): {max_value1}")
    print(f"  df0 (On-Time Payments):     {max_value0}\n")

    # 2. Вивід зведеної статистики для клієнтів із платіжними труднощами (df1)
    print("Клієнт з платіжними труднощами:")
    display(df1.groupby(by=[column_2, column_3])[column_1].describe().head(10))

    # 3. Вивід зведеної статистики для клієнтів зі вчасними платежами (df0)
    print("\nКлієнт зі вчасними платежами:")
    display(df0.groupby(by=[column_2, column_3])[column_1].describe().head(10))

    # 4. Побудова парних боксплотів
    bi_boxplot(df1, df0, column_2, column_1, max_value1, max_value0, column_3)