import pandas as pandas
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

pandas.set_option('display.float_format', lambda x:'%.2f'%x)

data = pandas.read_csv('https://raw.githubusercontent.com/Vazelisk/coursera/main/student-mat.csv', sep=';')
# want to look how final grades depends on the romantic status

for line in range(len(data['romantic'])):
    if data['romantic'][line] == 'yes':
        data['romantic'][line] = 1
    else:
        data['romantic'][line] = 0

print('frequency table')
print(data['romantic'].value_counts())

data['romantic'] = pandas.to_numeric(data['romantic'], errors='coerce')
data['G3'] = pandas.to_numeric(data['G3'], errors='coerce')

sns.factorplot(x="romantic", y="G3", data=data, kind="bar", ci=None)
plt.xlabel('has romantic status (0 - no, 1 - yes)')
plt.ylabel('Grades')

reg1 = smf.ols('G3 ~ romantic', data=data).fit()
print (reg1.summary())

#The results of the linear regression model indicated that having a romantic partner (Beta = -0.0134, p=.0001, but R-squared only 1.4%) wasn't significantly 
# and positively associated with the final grade.
