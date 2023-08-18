import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression

primary = pd.read_spss("Safeplay-Baseline_Caregiver_working_Final.sav")


#preparing secondary df (learning materials at home)
df_l = primary[["story_book_p3", "how_many_p3", "bp3", "cp3", "dp3", "ep3", "fp3", "gp3", "a2_p3", "b2_p3", "c2_p3", "d2_p3", "e2_p3", "f2_p3", "g2_p3", "h2_p3", "i2_p3", "j2_p3"]].copy()

#functions
def appending_to_remove_these(column, remove_these):
    for i in range(len(df_l[column])):
        if i not in remove_these and df_l[column][i] == "Do not Know":
            remove_these.append(i)

#removing don't know entries
remove_these = []

#SES

df_ses = primary[["roof_p4", "wall_p4", "barth_room", "indoor_toilet", "b_p4", "c_p4", "d_p4", "e_p4", "g_p4", "h_p4", "i_p4"]].copy()

#removing do not knows
for column in df_ses:
    for i in range(len(df_ses[column])):
        if i not in remove_these and df_ses[column][i] == "Do not Know":
            remove_these.append(i)
            
remove_these.append(54)

columns = ["story_book_p3", "ep3", "f2_p3", "g2_p3", "j2_p3"]

for i in range(len(columns)):
    appending_to_remove_these(columns[i], remove_these)

#Q47 for how many story books are there. getting rid of nans
df_l["how_many_p3"] = df_l["how_many_p3"].fillna(0)

#Encoding
for column in df_l:
    df_l[column] = df_l[column].replace(["Yes", "No"], [1, 0])

l_scores = []

for i in range(len(df_l["story_book_p3"])):
    if i not in remove_these:
        l_scores.append(df_l.iloc[i].sum())
###############################################################################

#SES
            
#Encoding
for column in df_ses:
    if column == "roof_p4":
        df_ses[column] = df_ses[column].replace(["Iron sheets", "Thatch", "Cement", "4"], [1, 0, 1, 0]) #Sturdier roof(?)
    elif column == "wall_p4":
        df_ses[column] = df_ses[column].replace(["Mud", "Thatch", "Stone", "Wood", "Cement"],[0, 0, 0, 0, 1]) #Sturdier walls
    else:
        df_ses[column] = df_ses[column].replace(["Yes", "No"],[1, 0])



ses = pd.DataFrame()

#df_ses.to_csv("SES.csv", index = True)

temp = pd.read_spss("SES.sav")

ses_scores = temp["FAC1_1"].tolist()


########################################################################################################

#For parent education
df_edu = primary[["mother_edu", "father_edu"]].copy()

#There are no do not know entries

#Encoding
for column in ["mother_edu", "father_edu"]:
    df_edu[column] = df_edu[column].replace(["No institutional education", "Pre-Primary", "Primary", "SSC", "HSC"], [1, 2, 3, 4, 5]).astype("int")

edu_scores = []
for i in range(0, 1184):
    if i not in remove_these:
        edu_scores.append(df_edu.iloc[i].sum())

###########################################################################################################

#behavior management

df_manage = primary[["k3_p3___0", "l3_p3___0", "m3_p3___0", 'n3_p3___0']].copy()

#there are no do not know entries

#Encoding
for column in df_manage:
    if(column != "k3_p3___0"):
       df_manage[column] = df_manage[column].replace(["Yes", "No"], [0, 1]).astype("int")
    else:
        df_manage[column] = df_manage[column].replace(["Yes", "No"], [1, 0]).astype("int")
        
manage_scores = []
for i in range(0, 1184):
    if i not in remove_these:
        manage_scores.append(df_manage.iloc[i].sum())

############################################################################################################
#Engagement
df_engage = primary[["a3_p3___1", "b3_p3___0", "c3_p3___0", "d3_p3___0", "e3_p3___0", "f3_p3___0", "g3_p3___0", "h3_p3___0", "i3_p3___0", "j3_p3___0"]].copy()

#there are no do not know values

#Encoding
for column in df_engage:
    df_engage[column] = df_engage[column].replace(["Yes", "No"], [1, 0]).astype("int")
    
engage_scores = []
for i in range(0, 1184):
    if i not in remove_these:
        engage_scores.append(df_engage.iloc[i].sum())
        
#########################################################################################################
#age
df_age = primary[["father_age", "c_mother_age"]].copy()

ages = []
for i in range(0, 1184):
    if i not in remove_these:
       ages.append(df_age.iloc[i].sum())


######################################################################################################    
#Regression analysis
def create_model(x, y):
    model = LinearRegression()
    np_x = np.array(x).reshape((-1, 1))
    np_y = np.array(y)
    model.fit(np_x, np_y)
    return np_x, np_y, model

def reg_analysis(x, y):
    np_x, np_y, model = create_model(x,y)
    return model.score(np_x, np_y)

#SES and engagement
r_sq_ses_engage = reg_analysis(ses_scores, engage_scores)
r_sq_ses_manage = reg_analysis(ses_scores, manage_scores)
r_sq_ses_l = reg_analysis(ses_scores, l_scores)

r_sq_ages_engage = reg_analysis(ages, engage_scores)
r_sq_ages_manage = reg_analysis(ages, manage_scores)
r_sq_ages_l = reg_analysis(ages, l_scores)

r_sq_edu_engage = reg_analysis(edu_scores, engage_scores)
r_sq_edu_manage = reg_analysis(edu_scores, manage_scores)
r_sq_edu_l = reg_analysis(edu_scores, l_scores)


r_sq_list = [["R-squared for Wealth Index and caregiver engagement scores", r_sq_ses_engage], ["R-squared for Wealth Index and behavior management scores", r_sq_ses_manage], ["R-squared for Wealth Index and home learning materials scores" , r_sq_ses_l], ["R-squared for parental age and caregiver engagement scores", r_sq_ages_engage], ["R-squared for parental age and behavior management scores", r_sq_ages_manage], ["R-squared for parental age and home learning materials scores" , r_sq_ages_l], ["R-squared for parental education and caregiver engagement scores", r_sq_edu_engage], ["R-squared for parental education and behavior management scores", r_sq_edu_manage], ["R-squared for parental education and home learning materials scores", r_sq_edu_l]]
r_sq = pd.DataFrame(r_sq_list, columns = ["Pairs", "R-squared values"])

#Plotting
def plotting(x, y, x_label, y_label):
    np_x, np_y, model = create_model(x,y)
    y_pred = model.predict(np_x)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    #plt.figure(figsize = (5,5))
    plt.scatter(np_x, np_y)
    plt.plot(np_x, y_pred, color = "k")
    plt.show()
    

