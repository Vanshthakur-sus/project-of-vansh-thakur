import pandas as pd
import seaborn as sns
tips= sns.load_dataset('tips')
tips.head()
titanic = pd.read_csv('tested.csv')
flights = sns.load_dataset('flights')
iris = sns.load_dataset('iris')
sns.scatterplot(data=tips, x='total_bill', y='tip', hue=tips['sex'], style=tips['smoker'], size=tips['size'])
titanic.head()
sns.barplot(x=titanic['Pclass'], y=titanic['Age'], hue=titanic['Sex']) # here pclass is the categorial data and age is the numerical data
sns.barplot(x=titanic['Pclass'], y=titanic['Fare'],hue=titanic['Sex'])
#3. BoxPlot (Numerical- Categorical)
sns.boxplot(x=titanic['Sex'],y=titanic['Age'], hue=titanic['Survived'])
#4. Distplot (Numerical - Categorical)
sns.distplot(titanic[titanic['Survived']==0]['Age'], hist=False)
sns.distplot(titanic[titanic['Survived']==1]['Age'], hist=False)
#5. Heatmap (Categorical - Categorical)
titanic.head(3)
sns.heatmap(pd.crosstab(titanic['Pclass'], titanic['Survived'])) #crosstab enables u to create tables like these
#6. ClusterMap (Categorical - Categorical)
sns.clustermap(pd.crosstab(titanic['SibSp'],titanic['Survived']))
#7. PairPlot
iris.head()
sns.pairplot(iris) # provides scatter plot after comparing each numerical cols with each other two check the relation b/w the cols
sns.pairplot(iris,hue='species')
#8. LinePlot (Numerical - Numerical)
# it is a special case of scatterplot, if you will connect all the pts. in the graph then it will become a lineplot
# it is used mostly when the value on the x-axis is time based or moving forward

flights.head()
