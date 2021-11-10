import pandas as pd

# # Question 3

# ## Part 1 

cohorts_set = ['G', 'H', 'I', 'J']

file_urls = ["https://wwwn.cdc.gov/Nchs/Nhanes/" + str(k) + '-' + 
            str(k + 1) + "/DEMO_" + str(j) + ".XPT" 
            for (k, j) in (zip(range(2011, 2018, 2), cohorts_set))]

Demo_cohorts = ["DEMO_" + cohort for cohort in cohorts_set]
Demo_col = ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'RIAGENDR', 'DMDEDUC2', 
           'DMDMARTL', 'RIDSTATR', 'SDMVPSU', 'SDMVSTRA', 
           'WTMEC2YR', 'WTINT2YR']

Demo_unique_col = ['SEQN', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2']

Demo_variable = ["id", "age", "race", "gender", "education", 
                 "marital_status", "exam_status", "psu", 
                 "stratum", "exam_weight", "interview_weight",
                 "cohort"]

Demo_dataset = pd.DataFrame(columns=Demo_variable)
for cohort_num in range(len(Demo_cohorts)):
    data = pd.read_sas(file_urls[cohort_num])[Demo_col]
    data = data.drop_duplicates(subset=Demo_unique_col)
    data["cohort"] = Demo_cohorts[cohort_num]
    data.columns = Demo_variable
    Demo_dataset = Demo_dataset.append(data, ignore_index=True)

Demo_type = [pd.Int64Dtype(), int, pd.CategoricalDtype(),
             pd.CategoricalDtype(), 
             pd.CategoricalDtype(), pd.CategoricalDtype(), 
             pd.Int64Dtype(), pd.Int64Dtype(),
             pd.Float64Dtype(), pd.Float64Dtype(), 
             pd.CategoricalDtype()]

Demo_name_type = dict(zip(Demo_variable, Demo_type))
for key in Demo_name_type.keys():
    Demo_dataset[key] = Demo_dataset[key].astype(Demo_name_type[key])

Demo_dataset.to_pickle('demographic_dataset.pkl')
print(Demo_dataset.head())


# ## Part 2

file_urls = ["https://wwwn.cdc.gov/Nchs/Nhanes/" + str(k) + '-' + 
            str(k + 1) + "/OHXDEN_" + str(j) + ".XPT" 
            for (k, j) in (zip(range(2011, 2018, 2), cohorts_set))]
data = pd.read_sas(file_urls[0])
OHXDEN_col = list(filter(lambda x: 'TC' in x and 'RTC' not in x, 
            data.columns))
TC_variable = pd.Series(list(filter(lambda x: 'TC' in x and 'CTC' not in x,
            OHXDEN_col)))
CTC_variable= pd.Series(list(filter(lambda x: 'CTC' in x, OHXDEN_col)))

TC_variable = TC_variable.str.replace('OHX', 'tc_')
TC_variable = list(TC_variable.str.replace('TC', ''))
CTC_variable = CTC_variable.str.replace('OHX', 'ctc_')
CTC_variable = list(CTC_variable.str.replace('CTC', ''))

OHXDEN_col = ['SEQN', 'OHDDESTS'] + OHXDEN_col
OHXDEN_variable = ['id', 'dentition_status']
OHXDEN_variable += TC_variable + CTC_variable + ['cohort']

OHXDEN_cohorts = ["OHXDEN_" + cohort for cohort in cohorts_set]
OHXDEN_dataset = pd.DataFrame(columns=OHXDEN_col)
for cohort_num in range(len(OHXDEN_cohorts)):
    data = pd.read_sas(file_urls[cohort_num])[OHXDEN_col]
    data = data.drop_duplicates(subset=['SEQN', 'OHDDESTS'])
    data["cohort"] = OHXDEN_cohorts[cohort_num]
    OHXDEN_dataset = OHXDEN_dataset.append(data, ignore_index=True)
OHXDEN_dataset.columns = OHXDEN_variable

OHXDEN_dataset['id'] = OHXDEN_dataset['id'].astype(int)
OHXDEN_dataset['dentition_status'] = (OHXDEN_dataset[
    'dentition_status'].astype(int)).astype(pd.CategoricalDtype())
OHXDEN_dataset[TC_variable] = (OHXDEN_dataset[
    TC_variable].astype(pd.Int64Dtype())
                              ).astype(pd.CategoricalDtype())
OHXDEN_dataset[CTC_variable] = OHXDEN_dataset[
    CTC_variable].astype(pd.StringDtype())
OHXDEN_dataset['cohort'] = (OHXDEN_dataset[
    'cohort'].astype(pd.StringDtype())
                           ).astype(pd.CategoricalDtype())

for ctc in CTC_variable:
    OHXDEN_dataset[ctc] = OHXDEN_dataset[ctc].str.replace(
        "b'", '')
    OHXDEN_dataset[ctc] = OHXDEN_dataset[ctc].str.replace(
        "'", '')
    OHXDEN_dataset[ctc] = OHXDEN_dataset[ctc].replace('', pd.NA)
OHXDEN_dataset[CTC_variable] = OHXDEN_dataset[
    CTC_variable].astype(pd.CategoricalDtype())    

print(OHXDEN_dataset)
OHXDEN_dataset.to_pickle('oral_health_and_dentition_dataset.pkl')

# Part c

print("The case number of demographic dataset is", 
      len(Demo_dataset), ".\n")
print("The case number of oral health and dentition dataset is", 
      len(OHXDEN_dataset), ".")