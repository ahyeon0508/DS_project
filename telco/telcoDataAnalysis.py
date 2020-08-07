import numpy as np
import pandas as pd
from util import get_one_fold, runCV

# 데이터 읽기
telco = pd.read_csv('telco.csv')
telco.head(5)
telco.shape # (31769, 40)
telco.columns
telco.info()

# 결측치 제거
telco = pd.read_csv('telco.csv', na_values='?').dropna()
telco.info()
telco.shape # (31743, 40)

# 데이터 시각화
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.font_manager as fm
from matplotlib import rc
import matplotlib

# 글꼴 깨짐 방지
matplotlib.rcParams['axes.unicode_minus'] = False
font_name = fm.FontProperties(fname='NanumBarunGothic.ttf').get_name()
rc('font', family=font_name)

# 범주형 변수 데이터 시각화
sns.catplot(data=telco, x='핸드셋', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='요금제', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='납부여부', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='통화량구분', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='지불방법', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='통화품질불만', hue='이탈여부', kind='count', aspect=4)
sns.catplot(data=telco, x='미사용', hue='이탈여부', kind='count', aspect=4)

# 연속형 변수 데이터 시각화
facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '연령')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '서비스기간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '단선횟수')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주간통화횟수')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주간통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '야간통화횟수')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '야간통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주말통화횟수')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주말통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국제통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국내통화요금_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '평균주간통화시간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '평균야간통화시간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '평균주말통화시간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국내통화횟수')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국내통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '평균국내통화시간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '총통화시간_분')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '요금부과시간')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '분당통화요금')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국내통화요금')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '총통화요금')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '부과요금')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '평균납부요금')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주간통화비율')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '야간통화비율')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '주말통화비율')
facet.add_legend()

facet = sns.FacetGrid(telco, hue='이탈여부', aspect=2)
facet.map(sns.kdeplot, '국제통화비율')
facet.add_legend()

# 성별에 따른 이탈여부
sns.countplot(data=telco, x='성별', hue='이탈여부')

# Pie chart
# 남자의 이탈, 유지 비율 확인
man_out = sum(((telco['성별']=='남') & (telco['이탈여부']=='이탈'))==True)
man_stay = sum(((telco['성별']=='남') & (telco['이탈여부']=='유지'))==True)

group_names = ['이탈', '유지']
group_sizes = [man_out, man_stay]
group_colors = ['yellowgreen', 'lightskyblue']
group_explodes = (0.1, 0)  # explode 1st slice

plt.pie(group_sizes,
        explode=group_explodes,
        labels=group_names,
        colors=group_colors,
        autopct='%1.2f%%',  # second decimal place
        shadow=True,
        startangle=90,
        textprops={'fontsize': 14})  # text font size
plt.axis('equal')  # equal length of X and Y axis
plt.title('남자 이탈 여부', fontsize=20)
plt.show()

# 여자의 이탈, 유지 비율 확인
woman_out = sum(((telco['성별']=='여') & (telco['이탈여부']=='이탈'))==True)
woman_stay = sum(((telco['성별']=='여') & (telco['이탈여부']=='유지'))==True)

group_names = ['이탈', '유지']
group_sizes = [woman_out, woman_stay]
group_colors = ['yellowgreen', 'lightskyblue']
group_explodes = (0.1, 0)  # explode 1st slice

plt.pie(group_sizes,
        explode=group_explodes,
        labels=group_names,
        colors=group_colors,
        autopct='%1.2f%%',  # second decimal place
        shadow=True,
        startangle=90,
        textprops={'fontsize': 14})  # text font size
plt.axis('equal')  # equal length of X and Y axis
plt.title('여자 이탈 여부', fontsize=20)
plt.show()


# 데이터 전처리
# 범주형 변수 -> 더미변수로 변환
set(telco['성별'])
telco['성별'] = telco['성별'].map({
    '남' : 0,
    '여' : 1
})

set(telco['이탈여부'])
telco['이탈여부'] = telco['이탈여부'].map({
    '유지' : 0,
    '이탈' : 1
})

set(telco['지불방법'])

set(telco['요금제'])
telco['요금제'] = telco['요금제'].map({
    'CAT 200' : 'CAT200',
    'Play 100': 'Play100',
    'CAT 100': 'CAT100',
    'CAT 50': 'CAT50',
    'Play 300': 'Play300'
})
telcoDummySet = pd.get_dummies(telco['요금제'])
telcoDummySet = telco.join(telcoDummySet.add_prefix('요금제_'))
telco = telcoDummySet.drop(['요금제'], axis=1)

set(telco['핸드셋'])
telcoDummySet = pd.get_dummies(telco['핸드셋'])
telcoDummySet = telco.join(telcoDummySet.add_prefix('핸드셋_'))
telco = telcoDummySet.drop(['핸드셋'], axis=1)

set(telco['납부여부'])
telco['납부여부'] = telco['납부여부'].map({
    'High Play 100' : 'HighPlay100',
    'High CAT 100': 'HighCAT100',
    'High CAT 50': 'HighCAT50',
    'OK': 'OK'
})
telcoDummySet = pd.get_dummies(telco['납부여부'])
telcoDummySet = telco.join(telcoDummySet.add_prefix('납부여부_'))
telco = telcoDummySet.drop(['납부여부'], axis=1)

set(telco['통화량구분'])
telcoDummySet = pd.get_dummies(telco['통화량구분'])
telcoDummySet = telco.join(telcoDummySet.add_prefix('통화량구분_'))
telco = telcoDummySet.drop(['통화량구분'], axis=1)

set(telco['통화품질불만'])
telco['통화품질불만'] = telco['통화품질불만'].map({
    'F' : 0,
    'T' : 1
})

set(telco['미사용'])
telco['미사용'].value_counts()

telco.info()


# 0-1 Normalization : 데이터의 단위 및 스케일이 변수마다 다르므로
ptile = np.percentile(telco['연령'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['연령'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['연령'].values, bins=10)
plt.hist(telco['연령'].values)

telco['연령'] = np.log(telco['연령'].values + 1)
plt.hist(telco['연령'])
telco['연령'] = (telco['연령']-min(telco['연령']))/(max(telco['연령'])-min(telco['연령']))
plt.hist(telco['연령'])

ptile = np.percentile(telco['서비스기간'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['서비스기간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['서비스기간'].values, bins=10)
plt.hist(telco['서비스기간'].values)

np_period = telco['서비스기간'].values
np_period = (np_period-min(np_period))/(max(np_period)-min(np_period))
plt.hist(np_period)
telco['서비스기간'] = np_period

ptile = np.percentile(telco['단선횟수'].values, [0, 25, 50, 75, 100])
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['단선횟수'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['단선횟수'].values, bins=10)
plt.hist(telco['단선횟수'].values)

telco['단선횟수'] = np.log(telco['단선횟수'].values + 1)
plt.hist(telco['단선횟수']) # 이상


telco['단선횟수'] = (telco['단선횟수']-min(telco['단선횟수']))/(max(telco['단선횟수'])-min(telco['단선횟수']))
plt.hist(telco['단선횟수'])


ptile = np.percentile(telco['주간통화횟수'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주간통화횟수'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['주간통화횟수'].values, bins=10)
plt.hist(telco['주간통화횟수'].values)

telco['주간통화횟수'] = np.log(telco['주간통화횟수'].values + 1)
plt.hist(telco['주간통화횟수'])
telco['주간통화횟수'] = (telco['주간통화횟수']-min(telco['주간통화횟수']))/(max(telco['주간통화횟수'])-min(telco['주간통화횟수']))
plt.hist(telco['주간통화횟수'])


ptile = np.percentile(telco['주간통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주간통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['주간통화시간_분'].values, bins=10)
plt.hist(telco['주간통화시간_분'].values)

telco['주간통화시간_분'] = np.log(telco['주간통화시간_분'].values + 1)
plt.hist(telco['주간통화시간_분'])
telco['주간통화시간_분'] = (telco['주간통화시간_분']-min(telco['주간통화시간_분']))/(max(telco['주간통화시간_분'])-min(telco['주간통화시간_분']))
plt.hist(telco['주간통화시간_분'])



ptile = np.percentile(telco['야간통화횟수'].values, [0, 25, 50, 75, 100])
print(ptile)
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['야간통화횟수'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['야간통화횟수'].values, bins=10)
plt.hist(telco['야간통화횟수'].values)

telco['야간통화횟수'] = np.log(telco['야간통화횟수'].values + 1)
plt.hist(telco['야간통화횟수'])
telco['야간통화횟수'] = (telco['야간통화횟수']-min(telco['야간통화횟수']))/(max(telco['야간통화횟수'])-min(telco['야간통화횟수']))
plt.hist(telco['야간통화횟수'])


ptile = np.percentile(telco['야간통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['야간통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['야간통화시간_분'].values, bins=10)
plt.hist(telco['야간통화시간_분'].values)

telco['야간통화시간_분'] = np.log(telco['야간통화시간_분'].values + 1)
plt.hist(telco['야간통화시간_분'])
telco['야간통화시간_분'] = (telco['야간통화시간_분']-min(telco['야간통화시간_분']))/(max(telco['야간통화시간_분'])-min(telco['야간통화시간_분']))
plt.hist(telco['야간통화시간_분'])



ptile = np.percentile(telco['주말통화횟수'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주말통화횟수'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['주말통화횟수'].values, bins=10)
plt.hist(telco['주말통화횟수'].values)

telco['주말통화횟수'] = np.log(telco['주말통화횟수'].values + 1)
plt.hist(telco['주말통화횟수'])
telco['주말통화횟수'] = (telco['주말통화횟수']-min(telco['주말통화횟수']))/(max(telco['주말통화횟수'])-min(telco['주말통화횟수']))
plt.hist(telco['주말통화횟수'])


ptile = np.percentile(telco['주말통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주말통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['주말통화시간_분'].values, bins=10)
plt.hist(telco['주말통화시간_분'].values)

telco['주말통화시간_분'] = np.log(telco['주말통화시간_분'].values + 1)
plt.hist(telco['주말통화시간_분'])
telco['주말통화시간_분'] = (telco['주말통화시간_분']-min(telco['주말통화시간_분']))/(max(telco['주말통화시간_분'])-min(telco['주말통화시간_분']))
plt.hist(telco['주말통화시간_분'])


ptile = np.percentile(telco['국제통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국제통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['국제통화시간_분'].values, bins=10)
plt.hist(telco['국제통화시간_분'].values)

telco['국제통화시간_분'] = np.log(telco['국제통화시간_분'].values + 1)
plt.hist(telco['국제통화시간_분'])
telco['국제통화시간_분'] = (telco['국제통화시간_분']-min(telco['국제통화시간_분']))/(max(telco['국제통화시간_분'])-min(telco['국제통화시간_분']))
plt.hist(telco['국제통화시간_분'])


ptile = np.percentile(telco['국내통화요금_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국내통화요금_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['국내통화요금_분'].values, bins=10)
plt.hist(telco['국내통화요금_분'].values)

telco['국내통화요금_분'] = np.log(telco['국내통화요금_분'].values + 1)
plt.hist(telco['국내통화요금_분'])
telco['국내통화요금_분'] = (telco['국내통화요금_분']-min(telco['국내통화요금_분']))/(max(telco['국내통화요금_분'])-min(telco['국내통화요금_분']))
plt.hist(telco['국내통화요금_분'])


ptile = np.percentile(telco['평균주간통화시간'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['평균주간통화시간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['평균주간통화시간'].values, bins=10)
plt.hist(telco['평균주간통화시간'].values)

telco['평균주간통화시간'] = np.log(telco['평균주간통화시간'].values + 1)
plt.hist(telco['평균주간통화시간'])
telco['평균주간통화시간'] = (telco['평균주간통화시간']-min(telco['평균주간통화시간']))/(max(telco['평균주간통화시간'])-min(telco['평균주간통화시간']))
plt.hist(telco['평균주간통화시간'])


ptile = np.percentile(telco['평균야간통화시간'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['평균야간통화시간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['평균야간통화시간'].values, bins=10)
plt.hist(telco['평균야간통화시간'].values)

telco['평균야간통화시간'] = np.log(telco['평균야간통화시간'].values + 1)
plt.hist(telco['평균야간통화시간'])
telco['평균야간통화시간'] = (telco['평균야간통화시간']-min(telco['평균야간통화시간']))/(max(telco['평균야간통화시간'])-min(telco['평균야간통화시간']))
plt.hist(telco['평균야간통화시간'])



ptile = np.percentile(telco['평균주말통화시간'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['평균주말통화시간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['평균주말통화시간'].values, bins=10)
plt.hist(telco['평균주말통화시간'].values)

telco['평균주말통화시간'] = np.log(telco['평균주말통화시간'].values + 1)
plt.hist(telco['평균주말통화시간'])
telco['평균주말통화시간'] = (telco['평균주말통화시간']-min(telco['평균주말통화시간']))/(max(telco['평균주말통화시간'])-min(telco['평균주말통화시간']))
plt.hist(telco['평균주말통화시간'])


ptile = np.percentile(telco['국내통화횟수'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국내통화횟수'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['국내통화횟수'].values, bins=10)
plt.hist(telco['국내통화횟수'].values)

telco['국내통화횟수'] = np.log(telco['국내통화횟수'].values + 1)
plt.hist(telco['국내통화요금_분'])
telco['국내통화횟수'] = (telco['국내통화횟수']-min(telco['국내통화횟수']))/(max(telco['국내통화횟수'])-min(telco['국내통화횟수']))
plt.hist(telco['국내통화횟수'])


ptile = np.percentile(telco['국내통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국내통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['국내통화시간_분'].values, bins=10)
plt.hist(telco['국내통화시간_분'].values)

telco['국내통화시간_분'] = np.log(telco['국내통화시간_분'].values + 1)
plt.hist(telco['국내통화시간_분'])
telco['국내통화시간_분'] = (telco['국내통화시간_분']-min(telco['국내통화시간_분']))/(max(telco['국내통화시간_분'])-min(telco['국내통화시간_분']))
plt.hist(telco['국내통화시간_분'])



ptile = np.percentile(telco['평균국내통화시간'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['평균국내통화시간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['평균국내통화시간'].values, bins=10)
plt.hist(telco['평균국내통화시간'].values)

telco['평균국내통화시간'] = np.log(telco['평균국내통화시간'].values + 1)
plt.hist(telco['평균국내통화시간'])
telco['평균국내통화시간'] = (telco['평균국내통화시간']-min(telco['평균국내통화시간']))/(max(telco['평균국내통화시간'])-min(telco['평균국내통화시간']))
plt.hist(telco['평균국내통화시간'])



ptile = np.percentile(telco['총통화시간_분'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['총통화시간_분'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['총통화시간_분'].values, bins=10)
plt.hist(telco['총통화시간_분'].values)

telco['총통화시간_분'] = np.log(telco['총통화시간_분'].values + 1)
plt.hist(telco['평균국내통화시간'])
telco['총통화시간_분'] = (telco['총통화시간_분']-min(telco['총통화시간_분']))/(max(telco['총통화시간_분'])-min(telco['총통화시간_분']))
plt.hist(telco['총통화시간_분'])



ptile = np.percentile(telco['요금부과시간'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['요금부과시간'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['요금부과시간'].values, bins=10)
plt.hist(telco['요금부과시간'].values)

telco['요금부과시간'] = telco['요금부과시간'].values
telco['요금부과시간'] = (telco['요금부과시간']-min(telco['요금부과시간']))/(max(telco['요금부과시간'])-min(telco['요금부과시간']))
plt.hist(telco['요금부과시간'])



ptile = np.percentile(telco['분당통화요금'].values, [0, 25, 50, 75, 100])
print(ptile)
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['분당통화요금'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
print(diff, "length:", len(diff))
plt.plot(diff)
hist = np.histogram(telco['분당통화요금'].values, bins=10)
plt.hist(telco['분당통화요금'].values)

telco['분당통화요금'] = telco['분당통화요금'].values
telco['분당통화요금'] = (telco['분당통화요금']-min(telco['분당통화요금']))/(max(telco['분당통화요금'])-min(telco['분당통화요금']))
plt.hist(telco['분당통화요금'])


ptile = np.percentile(telco['국내통화요금'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국내통화요금'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['국내통화요금'].values, bins=10)
plt.hist(telco['국내통화요금'].values)

telco['국내통화요금'] = np.log(telco['국내통화요금'].values + 1)
plt.hist(telco['국내통화요금'])
telco['국내통화요금'] = (telco['국내통화요금']-min(telco['국내통화요금']))/(max(telco['국내통화요금'])-min(telco['국내통화요금']))
plt.hist(telco['국내통화요금'])


ptile = np.percentile(telco['총통화요금'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['총통화요금'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['총통화요금'].values, bins=10)
plt.hist(telco['총통화요금'].values)

telco['총통화요금'] = np.log(telco['총통화요금'].values + 1)
plt.hist(telco['총통화요금'])
telco['총통화요금'] = (telco['총통화요금']-min(telco['총통화요금']))/(max(telco['총통화요금'])-min(telco['총통화요금']))
plt.hist(telco['총통화요금'])


ptile = np.percentile(telco['부과요금'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['부과요금'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['부과요금'].values, bins=10)
plt.hist(telco['부과요금'].values)

telco['부과요금'] = np.log(telco['부과요금'].values + 1)
plt.hist(telco['부과요금'])
telco['부과요금'] = (telco['부과요금']-min(telco['부과요금']))/(max(telco['부과요금'])-min(telco['부과요금']))
plt.hist(telco['부과요금'])


ptile = np.percentile(telco['평균납부요금'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['평균납부요금'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['평균납부요금'].values, bins=10)
plt.hist(telco['평균납부요금'].values)

telco['평균납부요금'] = np.log(telco['평균납부요금'].values + 1)
plt.hist(telco['평균납부요금'])
telco['평균납부요금'] = (telco['평균납부요금']-min(telco['평균납부요금']))/(max(telco['평균납부요금'])-min(telco['평균납부요금']))
plt.hist(telco['평균납부요금'])



ptile = np.percentile(telco['주간통화비율'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주간통화비율'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['주간통화비율'].values, bins=10)
plt.hist(telco['주간통화비율'].values)

telco['주간통화비율'] = np.log(telco['주간통화비율'].values + 1)
plt.hist(telco['주간통화비율'])
telco['주간통화비율'] = (telco['주간통화비율']-min(telco['주간통화비율']))/(max(telco['주간통화비율'])-min(telco['주간통화비율']))
plt.hist(telco['주간통화비율'])



ptile = np.percentile(telco['야간통화비율'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['야간통화비율'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['야간통화비율'].values, bins=10)
plt.hist(telco['야간통화비율'].values)

telco['야간통화비율'] = np.log(telco['야간통화비율'].values + 1)
plt.hist(telco['야간통화비율'])
telco['야간통화비율'] = (telco['야간통화비율']-min(telco['야간통화비율']))/(max(telco['야간통화비율'])-min(telco['야간통화비율']))
plt.hist(telco['야간통화비율'])



ptile = np.percentile(telco['주말통화비율'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['주말통화비율'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['주말통화비율'].values, bins=10)
plt.hist(telco['주말통화비율'].values)

telco['주말통화비율'] = np.log(telco['주말통화비율'].values + 1)
plt.hist(telco['주말통화비율'])
telco['주말통화비율'] = (telco['주말통화비율']-min(telco['주말통화비율']))/(max(telco['주말통화비율'])-min(telco['주말통화비율']))
plt.hist(telco['주말통화비율'])



ptile = np.percentile(telco['국제통화비율'].values, [0, 25, 50, 75, 100])
import matplotlib.pyplot as plt
l_count = [0]
for i in range(1, 11, 1):
    pivot = i*ptile[-1]/10
    c = sum(telco['국제통화비율'].values <= pivot)
    l_count.append(c)
np_counts = np.array(l_count)
diff = np_counts[1:] - np_counts[:-1]
plt.plot(diff)
hist = np.histogram(telco['국제통화비율'].values, bins=10)
plt.hist(telco['국제통화비율'].values)

telco['국제통화비율'] = np.log(telco['국제통화비율'].values + 1)
plt.hist(telco['국제통화비율'])
telco['국제통화비율'] = (telco['국제통화비율']-min(telco['국제통화비율']))/(max(telco['국제통화비율'])-min(telco['국제통화비율']))
plt.hist(telco['국제통화비율'])


# 데이터 셋 구성(data/label)
telco_data = telco.drop(['이탈여부'], axis = 1)
telco_label = telco['이탈여부']

# 분류에 영향을 미치지 않는 변수 제거
telco_data = telco_data.drop(['id', '고객ID', '개시일', '지불방법', '미사용'], axis = 1)

# 연속형-연속형 변수의 상관관계를 히트맵으로 표현
# 더미변수는 범주형 변수를 연속형 변수로 변환한 것
corr = telco_data.corr()
plt.figure(figsize=(60,40))
sns.heatmap(corr, annot=True, fmt='.1f')

# 모델 학습
from sklearn.utils import shuffle
from random import seed

# Random Forest
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf = clf.fit(telco_data[:22000], telco_label[:22000]) # 모델 학습
pred = clf.predict(telco_data[22000:]) # 모델 예측
correct = pred==telco_label[22000:]
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
print("Random Forest : ", acc) # 0.9294

# 10 fold cross validation
numbers = list(range(len(telco_data)))
numbers = shuffle(numbers)
shuffled_data = telco_data.iloc[numbers]
shuffled_labels = telco_label.iloc[numbers]
clf = RandomForestClassifier()
results = runCV(clf, shuffled_data, shuffled_labels, fold=10, isAcc=True)
print("Random Forest(10 fold) : ", np.mean(results)) # 0.9279


# Forward selection
import statsmodels.formula.api as smf
cnames = telco_data.columns.values

telco_train = telco[:22000]
telco_test = telco[22000:]

def train_test_once(_candi):
    order = "이탈여부 ~ %s" % ' + '.join([cnames[each_idx] for each_idx in _candi])
    lm_candi = smf.ols(order, telco_train).fit()
    pred = lm_candi.predict(telco_test)
    diff = np.power(pred.values - telco_test['이탈여부'].values, 2)
    mse = np.sum(diff) / len(diff)
    return mse

# 다중공선성 해결하기 위하여 단계적 회귀분석 이용
# Forward Selection
# mse 0.0999
res_forward = []
for max_num in range(1, len(cnames)):
    if len(res_forward) != 0:
     before_best = res_forward[-1][0]
    else:
        before_best = []
    idx_combi = [[*before_best, idx] for idx in range(len(cnames)) if idx not in before_best]
    res_temp = []
    for candidate in idx_combi:
        mse = train_test_once(candidate)
        res_temp.append((candidate, mse))
    temp_best = sorted(res_temp, key=lambda x:x[1])[0]
    res_forward.append(temp_best)
ascending = sorted(res_forward, key=lambda x:x[1])
best_idx = ascending[0][0]
forward_best_features = cnames[list(best_idx)]

# Backward Selection
# mse 0.2279
res_backward = []
before_best = list(range(len(cnames)))
for max_num in range(1, len(cnames)):
    idx_combi = [list(before_best) for _ in range(len(before_best))]
    list(map(lambda x: idx_combi[x[0]].remove(x[1]), enumerate(before_best)))
    res_temp = []
    for candidate in idx_combi:
        mse = train_test_once(candidate)
        res_temp.append((candidate, mse))
    temp_best = sorted(res_temp, key=lambda x:x[1])[0]
    res_backward.append(temp_best)
    before_best = temp_best[0]
ascending = sorted(res_backward, key=lambda x:x[1])
best_idx = ascending[0][0]
backward_best_features = cnames[list(best_idx)]

# MSE가 작을수록 좋으므로 Forward selection에서 나온 best_feature 이용

# best_feature
telco_data = telco_data[['핸드셋_ASAD90', '통화품질불만', '핸드셋_CAS30', '통화량구분_중저', '요금제_CAT50',
       '통화량구분_저', '핸드셋_S50', '납부여부_OK', '핸드셋_BS110', '핸드셋_SOP20',
       '핸드셋_SOP10', '서비스기간', '요금제_Play100', '요금제_Play300', '핸드셋_BS210',
       '핸드셋_S80', '핸드셋_CAS60', '연령', '요금제_CAT100', '평균납부요금', '평균주간통화시간',
       '납부여부_HighCAT100', '평균국내통화시간', '요금부과시간', '요금제_CAT200', '평균야간통화시간',
       '주말통화비율', '통화량구분_중고', '단선횟수', '주간통화비율']]




# LogisticRegression
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression()
clf = clf.fit(telco_data[:22000], telco_label[:22000]) # 모델 학습
pred = clf.predict(telco_data[22000:]) # 모델 예측
correct = pred==telco_label[22000:]
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
print("Logistic Regression : ", acc) # 0.8929

seed(0)
numbers = list(range(len(telco_data)))
numbers = shuffle(numbers)
shuffled_data = telco_data.iloc[numbers]
shuffled_labels = telco_label.iloc[numbers]
clf = LogisticRegression()
results = runCV(clf, telco_data, telco_label, fold = 10, isAcc=True)
mean_acc = np.mean(results)
print("LogisticRegression(10 fold) : ""%.4f" % np.mean(results)) # 0.8812

from sklearn.model_selection import GridSearchCV
grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty':['l1', 'l2']} # l1 lasso l2 ridge
clf = LogisticRegression()
clf_cv = GridSearchCV(clf, grid, cv=10)
clf_cv.fit(telco_data, telco_label)

print("tuned hpyerparameters :(best parameters) ", clf_cv.best_params_)
print("accuracy :", clf_cv.best_score_)

logreg = LogisticRegression(C = 10, penalty = 'l2')
logreg.fit(telco_data, telco_label)
print("LogisticRegression(grid) : ", logreg.score(telco_data, telco_label)) # 0.8855

logreg = LogisticRegression(C = 10, penalty = 'l2')
results = runCV(logreg, telco_data, telco_label, fold = 10, isAcc=True)
mean_acc = np.mean(results)
print("LogisticRegression(10 fold_grid) : ""%.4f" % np.mean(results)) # 0.8818



# LDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
clf = LDA()
clf = clf.fit(telco_data[:22000], telco_label[:22000]) # 모델 학습
pred = clf.predict(telco_data[22000:]) # 모델 예측
correct = pred==telco_label[22000:]
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
print("LDA : ", acc) # 0.8806

numbers = list(range(len(telco_data)))
numbers = shuffle(numbers)
shuffled_data = telco_data.iloc[numbers]
shuffled_labels = telco_label.iloc[numbers]
clf = LDA()
results = runCV(clf, telco_data, telco_label, fold = 10, isAcc=True)
mean_acc = np.mean(results)
print("LinearDiscriminantAnalysis(10 fold) : ""%.4f" % np.mean(results)) # 0.8742



# QDA
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
clf = QDA()
clf = clf.fit(telco_data[:22000], telco_label[:22000]) # 모델 학습
pred = clf.predict(telco_data[22000:]) # 모델 예측
correct = pred==telco_label[22000:]
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
print("QDA : ", acc) # 0.8792

numbers = list(range(len(telco_data)))
numbers = shuffle(numbers)
shuffled_data = telco_data.iloc[numbers]
shuffled_labels = telco_label.iloc[numbers]
clf = QDA()
results = runCV(clf, telco_data, telco_label, fold = 10, isAcc=True)
mean_acc = np.mean(results)
print("QuadraticDiscriminantAnalysis(10 fold) : ""%.4f" % np.mean(results)) # 0.8725




# SVM
from sklearn.svm import SVC
clf = SVC()
clf = clf.fit(telco_data[:22000], telco_label[:22000]) # 모델 학습
pred = clf.predict(telco_data[22000:]) # 모델 예측
correct = pred==telco_label[22000:]
acc = sum([1 if x == True else 0 for x in correct])/len(correct)
print("SVM(SVC) : ", acc) # 0.9047

numbers = list(range(len(telco_data)))
numbers = shuffle(numbers)
shuffled_data = telco_data.iloc[numbers]
shuffled_labels = telco_label.iloc[numbers]
clf = SVC()
results = runCV(clf, telco_data, telco_label, fold = 5, isAcc=True)
mean_acc = np.mean(results)
print("SVM(SVC)(5 fold) : ", mean_acc) # 0.8911



# Ensemble
# Logistic Regression, LDA, QDA, SVM
# 네가지 모형을 다수결로 합쳐 모델의 성능 확인함
d_telco = telco_data
l_telco = telco_label
d_telco_tr = d_telco[:18000]
l_telco_tr = l_telco[:18000]
d_telco_v = d_telco[18000:22000]
l_telco_v = l_telco[18000:22000]
d_telco_te = d_telco[22000:]
l_telco_te = l_telco[22000:]

from sklearn.metrics import f1_score
l_clf = []
l_clf.append(LogisticRegression())
l_clf.append(LDA())
l_clf.append(QDA())
l_clf.append(SVC())
voted = []
l_f1_score = []
for each_clf in l_clf:
    each_clf.fit(d_telco_tr, l_telco_tr)
    pred = each_clf.predict(d_telco_te)
    s = f1_score(l_telco_te, pred, average='macro')
    l_f1_score.append(s)
    voted.append(pred)

voted = []
l_f1_score = []
for each_clf in l_clf:
    each_clf.fit(d_telco_tr, l_telco_tr)
    pred = each_clf.predict(d_telco_te)
    s = f1_score(l_telco_te, pred, average='macro')
    l_f1_score.append(s)
    voted.append(pred)

from collections import Counter
voted = np.array(voted)
voted = np.transpose(voted)
final_pred = []
for each in voted:
    ed = Counter(each)
    majority = sorted(ed.items(), key=lambda x:x[1], reverse=True)[0]
    final_pred.append(majority[0])
fscore = f1_score(l_telco_te, final_pred, average='macro') # 0.8939



# SVM모델이 약 90.47%로 가장 높은 정확도를 보임