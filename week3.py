import pandas as pandas
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn
import statsmodels.api as sm
import numpy


pandas.set_option('display.float_format', lambda x:'%.2f'%x)

data = pandas.read_csv('https://raw.githubusercontent.com/Vazelisk/coursera/main/student-mat.csv', sep=';')
data.drop(columns=['G1', 'G2'])
# want to look how final grades depends on the romantic status and school support

for line in range(len(data['romantic'])):
    if data['romantic'][line] == 'yes':
        data['romantic'][line] = 1
    else:
        data['romantic'][line] = 0
        
for line in range(len(data['schoolsup'])):
    if data['schoolsup'][line] == 'yes':
        data['schoolsup'][line] = 1
    else:
        data['schoolsup'][line] = 0
        
for line in range(len(data['internet'])):
    if data['internet'][line] == 'yes':
        data['internet'][line] = 1
    else:
        data['internet'][line] = 0

for line in range(len(data['sex'])):
    if data['sex'][line] == 'M':
        data['sex'][line] = 1
    else:
        data['sex'][line] = 0

print('frequency table')
print(data['romantic'].value_counts())
print(data['schoolsup'].value_counts())
print(data['internet'].value_counts())
print(data['sex'].value_counts())

data['romantic'] = pandas.to_numeric(data['romantic'], errors='coerce')
data['G3'] = pandas.to_numeric(data['G3'], errors='coerce')
data['schoolsup'] = pandas.to_numeric(data['schoolsup'], errors='coerce')
data['internet'] = pandas.to_numeric(data['internet'], errors='coerce')
data['sex'] = pandas.to_numeric(data['sex'], errors='coerce')


# adding number of cigarettes smoked as an explanatory variable 
# center quantitative IVs for regression analysis
data['age_c'] = (data['age'] - data['age'].mean())
print(data['age_c'].mean())


# multiple regression with romantic and centered age PASSED
reg2 = smf.ols('G3 ~ age_c + romantic', data=data).fit()
print(reg2.summary())

###############
# linear regression analysis with internet NOT PASSED
reg3 = smf.ols('G3 ~ internet', data=data).fit()
print (reg3.summary())

# multiple regression analysis with internet & romantic
reg4 = smf.ols('G3 ~ internet + romantic', data=data).fit()
print (reg4.summary())
#thus, internet is a confounding variable regarding with romantic status

###############
# linear regression analysis with internet NOT PASSED
reg3 = smf.ols('G3 ~ sex', data=data).fit()
print (reg3.summary())

# multiple regression analysis with internet & romantic
reg4 = smf.ols('G3 ~ sex + romantic', data=data).fit()
print (reg4.summary())
#thus, internet is a confounding variable regarding with romantic status

# multiple regression analysis with covariates
reg5 = smf.ols('G3 ~ internet + romantic + age_c + schoolsup + sex', data=data).fit()
print (reg5.summary())
###########################
# linear regression analysis with age 
reg6 = smf.ols('G3 ~ age_c', data=data).fit()
print (reg6.summary())

#EVALUATING
#Q-Q plot for normality
fig1=sm.qqplot(reg5.resid, line='r')

#standardized residuals for all observations
stdres=pandas.DataFrame(reg5.resid_pearson)
plt.plot(stdres, 'o', ls='None')
l = plt.axhline(y=0, color='r')
plt.ylabel('Standardized Residual')
plt.xlabel('Observation Number')


# leverage plot
fig3=sm.graphics.influence_plot(reg5, size=8)
print(fig3)

#############################################################
reg6 = smf.ols('G3 ~ internet + romantic + age_c + schoolsup + sex + C(Mjob)', data=data).fit()
print (reg6.summary())


reg7 = smf.ols('G3 ~ internet + romantic + age_c + schoolsup + sex + C(Mjob, Treatment(reference=1))', data=data).fit()
print (reg7.summary())


##############################################################################
# LOGISTIC REGRESSION
##############################################################################

lreg1 = smf.logit(formula = 'romantic ~ sex', data = data).fit()
print (lreg1.summary())

# odds ratios
print ("Odds Ratios")
print (numpy.exp(lreg1.params))


# odd ratios with 95% confidence intervals
params = lreg1.params
conf = lreg1.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))



lreg2 = smf.logit(formula = 'romantic ~ sex + internet', data = data).fit()
print (lreg2.summary())

# odd ratios with 95% confidence intervals
params = lreg2.params
conf = lreg2.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))


# logistic regression with panic
lreg3 = smf.logit(formula = 'romantic ~ internet', data = data).fit()
print (lreg3.summary())

# odd ratios with 95% confidence intervals
print ("Odds Ratios")
params = lreg3.params
conf = lreg3.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))

# logistic regression with panic and depression
lreg4 = smf.logit(formula = 'romantic ~ internet + sex', data = data).fit()
print (lreg4.summary())

# odd ratios with 95% confidence intervals
print ("Odds Ratios")
params = lreg4.params
conf = lreg4.conf_int()
conf['OR'] = params
conf.columns = ['Lower CI', 'Upper CI', 'OR']
print (numpy.exp(conf))