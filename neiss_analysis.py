# import relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

#  ############################  Data Exploration and Preparation ############################# #
# Read csv files
neiss_2014 = pd.read_csv("./data/NEISS2014.csv")
body_parts = pd.read_csv("./data/BodyParts.csv")
diagnosis = pd.read_csv("./data/DiagnosisCodes.csv")
disposition = pd.read_csv("./data/Disposition.csv")

# explore body_parts
body_parts.head()

# explore diagnosis by code
diagnosis.head()

# explore Disposition by code
disposition.head()


# Total number of records in NEISS dataframe
neiss_row_count = neiss_2014.shape[0]
print("Total number of records in NEISS data:", neiss_row_count)

neiss_unique_cases = len(neiss_2014['CPSC Case #'].unique().tolist())
print("Unique cases:", neiss_row_count)

# rename column BodyPart from data body_parts

body_parts.rename(columns={"BodyPart": "body_part_name"}, inplace=True)

# merge with NEISS data to get body_part_name
neiss_2014 = pd.merge(neiss_2014,
                  body_parts,
                  left_on='body_part',
                  right_on='Code',
                  how='left')

# Drop the column Code from merged table as it is not needed
neiss_2014 = neiss_2014.drop('Code', 1)

# rename column Diagnosis from diagnosis data

diagnosis.rename(columns={"Diagnosis": "diagnosis_name"}, inplace=True)

# merge with NEISS data to get relevant name of the diagnosis
neiss_2014 = pd.merge(neiss_2014,
                  diagnosis,
                  left_on='diag',
                  right_on='Code',
                  how='left')

# Drop the column Code from merged table as it is not needed
neiss_2014 = neiss_2014.drop('Code', 1)

# rename column Disposition from disposition

disposition.rename(columns={"Disposition": "disposition_name"}, inplace=True)

# merge with NEISS data to get body_part_name
neiss_2014 = pd.merge(neiss_2014,
                  disposition,
                  left_on='disposition',
                  right_on='Code',
                  how='left')

# Drop the column Code from merged table as it is not needed
neiss_2014 = neiss_2014.drop('Code', 1)


#  #############################  Data Analysis  ############################# #

### Question 1

# (A) What are the top three body parts most frequently represented in this dataset?
 # Printing top 6 for further analysis

top_body_parts = neiss_2014['body_part_name'].value_counts().nlargest(6)
print("\n Top Body Parts:\n")
print(top_body_parts)


# (B) What are the top three body parts that are least frequently represented?
    # Printing top 6 for further analysis
bottom_body_parts = neiss_2014['body_part_name'].value_counts().nsmallest(6)
print("\nBottom Body Parts:\n")
print(bottom_body_parts)

### Question 2

# (A) How many injuries in this dataset involve a skateboard?

# Count of narratives that had words such as skateboard, skate board, skate boarding
skateboard_condition = neiss_2014.narrative.str.contains(r'SKATEBOARD|SKATE BOARD', case=False)
total_skateboard_injuries = skateboard_condition.sum()

print("\nTotal Skateboard related injuries identified from narrative/description of the injuriy: ")
print(total_skateboard_injuries)


#  (B) Of those injuries, what percentage were male and what percentage were female?

# Total female skateboard injuries
female_skateboard_injuries = ((neiss_2014['sex'] == 'Female') & (skateboard_condition == True)).sum()

# % of skateboard related injuries that were female
percent_female = "{:.2%}".format(female_skateboard_injuries/total_skateboard_injuries)

# Total male skateboard injuries
male_skateboard_injuries = ((neiss_2014['sex'] == 'Male') & (skateboard_condition == True)).sum()

# % of skateboard related injuries that were male
percent_male = "{:.2%}".format(male_skateboard_injuries/total_skateboard_injuries)

print("\nSkateboard Injuries by Sex")
print("Female: ", percent_female)
print("Male: ", percent_male)



#### (C) What was the average age of someone injured in an incident involving a skateboard?

# create a new column called age_in_years that shows all age in terms of years
# 1 month = 0.083333 years
neiss_2014['age_in_years'] = np.where(neiss_2014['age'] >=200,round((neiss_2014['age']-200)*0.083333,2),\
                                      neiss_2014['age'])

# Average age of someone injuried in an incident involving a skateboard in terms of mean
mean_age = neiss_2014[skateboard_condition==True].mean()['age_in_years']
# Average age of someone injuried in an incident involving a skateboard in terms of median
median_age = neiss_2014[skateboard_condition==True].median()['age_in_years']

print("\nAverage age of someone injuried because of skateboard:")

print("Mean: ", int(round(mean_age,1)), "years")
print("Median: ",int(round(median_age,1)), "years")


### Question 3

# (A) What diagnosis had the highest hospitalization rate?
    # Hospitalization rate is calculated as number of hospital inpatient discharges in particular group
        # divided by the population in that group * 1000. In our case group can be diagnosis.

# total cases by diagnosis
diagnosis_total =  neiss_2014.groupby('diagnosis_name')['diagnosis_name'].agg(['size']).rename(columns={'size': 'total_population'})

# cases hospitalized by diagnosis
diagnosis_hospitalized = neiss_2014[(neiss_2014['disposition_name']=='Treated and admitted for hospitalization (within same facility)')]
diagnosis_hospitalized = diagnosis_hospitalized.groupby('diagnosis_name')['diagnosis_name'].agg(['size']).rename(columns={'size': 'total_hospitalized'})

# merge total cases and cases hospitalized to calculate hospitalization rate by diagnosis
# hospitalization_Rate = total cases hospitalized / total cases

hospitalization_data = pd.concat([diagnosis_total, diagnosis_hospitalized], axis=1)
hospitalization_data['hospitalization_rate (%)'] = round(hospitalization_data['total_hospitalized']/hospitalization_data['total_population'] * 100,2)
hospitalization_data = hospitalization_data.sort_values(by='hospitalization_rate (%)', ascending=False)

# view the top diagnosis by hospitalization rate
print("\nDiagnosis that had highest hospitalization Rate:")
print(hospitalization_data.head(1))



#(B) What diagnosis most often concluded with the individual leaving without being seen?

# cases left without being seen by diagnosis
diagnosis_not_seen = neiss_2014[(neiss_2014['disposition_name']=='Left without being seen/Left against medical advice')]
diagnosis_not_seen = diagnosis_not_seen.groupby('diagnosis_name')['diagnosis_name'].agg(['size']).rename(columns={'size': 'total_not_seen'})

# merge total cases and cases left without being seen to calculate rate by diagnosis

not_seen_data = pd.concat([diagnosis_total, diagnosis_not_seen], axis=1)
not_seen_data['not_seen_rate (%)'] = round(not_seen_data['total_not_seen']/not_seen_data['total_population'] * 100,2)
not_seen_data = not_seen_data.sort_values(by='not_seen_rate (%)', ascending=False)

# view top diagnosis
print("\nDiagnosis most often concluded with the individual leaving without being seen:")
print(not_seen_data.head(1))

# (C) Briefly discuss your findings and any caveats you'd mention when discussing this data

# cases with diagnosis of value other or not stated

diagnosis_not_stated = ((neiss_2014['diag'] == 71).sum() / neiss_unique_cases)* 10
print(" \n% of cases with diagnosis not stated or marked as other: ", round(diagnosis_not_stated,2))


#### Question 4:  Visualize any existing relationship between age and reported injuries


# Provide meaningful group to age

neiss_2014['age_group'] = pd.cut(neiss_2014['age_in_years'], [0, 2, 14, 24, 64,150], labels=['Infant', 'Children', 'Youth', 'Adults', 'Seniors'])

# reported cases by age group
by_age_group =  neiss_2014.groupby('age_group')['age_group'].agg(['size']).rename(columns={'size': 'total_cases'})

# adding column to show in terms of percentage of total
by_age_group['percent_total'] = round((by_age_group['total_cases']/neiss_unique_cases)*100,2)


# slightly increase size of font
matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)

# define label and titles
plt.title('Reported Injuries by Age Group')
plt.xlabel('Age Group')
plt.ylabel('Total Cases')

by_age_group['total_cases'].plot.bar(figsize=(10,5))
plt.show()


# Prepare data to visualize by diagnosis and age group

neiss_2014['cases'] = 1

# group by age group and diagnosis and get total count of cases for each combination

by_age_group_diagnosis = neiss_2014.groupby(['age_group', "diagnosis_name"], as_index=False).cases.sum()

# sort by total cases
by_age_group_diagnosis = by_age_group_diagnosis.sort_values(['age_group','cases'],ascending=False).groupby('age_group').head(10)

# pivot the data to plot stacked bar chart
age_group_pivoted = by_age_group_diagnosis.pivot(index='age_group', columns='diagnosis_name', values='cases')


# tableau 20 color scale
tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),
             (44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),
             (148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),
             (227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),
             (188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]

# Rescale to values between 0 and 1
for i in range(len(tableau20)):
    r, g, b = tableau20[i]
    tableau20[i] = (r / 255., g / 255., b / 255.)


# horizontal stacked bar chart
age_group_pivoted.plot.barh(stacked = True, color = tableau20, figsize=(15,10))

# slightly increase size of font
matplotlib.rc('xtick', labelsize=15)
matplotlib.rc('ytick', labelsize=15)

# set axis label and title
plt.title('Diagnosis for Reported Injuries by Age Group ')
plt.ylabel('Age Group')
plt.xlabel('Total Cases')

# make sure legend is on side of the bar not over them
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

# show the visualization
plt.show()