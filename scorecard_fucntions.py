import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve 

### Calculate the cumulative frequences of events for each selected time window
def TimeWindowSelection(df, daysCol, time_windows):
    '''
    :param df: the dataset containg variabel of days
    :param daysCol: the column of days
    :param time_windows: the list of time window
    :return:
    '''
    freq_tw = {}
    for tw in time_windows:
        freq = sum(df[daysCol].apply(lambda x: int(x<=tw)))
        freq_tw[tw] = freq
    return freq_tw


def ChangeContent(x):
    y = x.upper()
    if y == '_MOBILEPHONE':
        y = '_PHONE'
    return y

def MissingCategorial(df,x):
    missing_vals = df[x].map(lambda x: int(x!=x))
    return sum(missing_vals)*1.0/df.shape[0]

def MissingContinuous(df,x):
    missing_vals = df[x].map(lambda x: int(np.isnan(x)))
    return sum(missing_vals) * 1.0 / df.shape[0]

def MakeupRandom(x, sampledList):
    if x==x:
        return x
    else:
        randIndex = random.randint(0, len(sampledList)-1)
        return sampledList[randIndex]

def AssignBin(x, cutOffPoints,special_attribute=[]):
    '''
    :param x: the value of variable
    :param cutOffPoints: the ChiMerge result for continous variable
    :param special_attribute:  the special attribute which should be assigned separately
    :return: bin number, indexing from 0
    for example, if cutOffPoints = [10,20,30], if x = 7, return Bin 0. If x = 35, return Bin 3
    '''
    numBin = len(cutOffPoints) + 1 + len(special_attribute)
    if x in special_attribute:
        i = special_attribute.index(x)+1
        return 'Bin {}'.format(0-i)
    if x<=cutOffPoints[0]:
        return 'Bin 0'
    elif x > cutOffPoints[-1]:
        return 'Bin {}'.format(numBin-1)
    else:
        for i in range(0,numBin-1):
            if cutOffPoints[i] < x <=  cutOffPoints[i+1]:
                return 'Bin {}'.format(i+1)


def MaximumBinPcnt(df,col):
    N = df.shape[0]
    total = df.groupby([col])[col].count()
    pcnt = total*1.0/N
    return max(pcnt)

def CalcWOE(df, col, target):
    '''
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be calculated the WOE and iv, usually categorical type
    :param target: good/bad indicator
    :return: WOE and IV in a dictionary
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    N = sum(regroup['total'])
    B = sum(regroup['bad'])
    regroup['good'] = regroup['total'] - regroup['bad']
    G = N - B
    regroup['bad_pcnt'] = regroup['bad'].map(lambda x: x*1.0/B)
    regroup['good_pcnt'] = regroup['good'].map(lambda x: x * 1.0 / G)
    regroup['WOE'] = regroup.apply(lambda x: np.log(x.good_pcnt*1.0/(x.bad_pcnt+0.0001)+0.0001),axis = 1)
    WOE_dict = regroup[[col,'WOE']].set_index(col).to_dict(orient='index')
    for k, v in WOE_dict.items():
        WOE_dict[k] = v['WOE']
    IV = regroup.apply(lambda x: (x.good_pcnt-x.bad_pcnt)*np.log(x.good_pcnt*1.0/(x.bad_pcnt+0.0001)+0.0001),axis = 1)
    IV = sum(IV)
    return {"WOE": WOE_dict, 'IV':IV}


def BadRateEncoding(df, col, target):
    '''
    :param df: dataframe containing feature and target
    :param col: the feature that needs to be encoded with bad rate, usually categorical type
    :param target: good/bad indicator
    :return: the assigned bad rate to encode the categorical feature
    '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    br_dict = regroup[[col,'bad_rate']].set_index([col]).to_dict(orient='index')
    for k, v in br_dict.items():
        br_dict[k] = v['bad_rate']
    badRateEnconding = df[col].map(lambda x: br_dict[x])
    return {'encoding':badRateEnconding, 'br_rate':br_dict}



def Chi2(df, total_col, bad_col, overallRate):
    '''
    :param df: the dataset containing the total count and bad count
    :param total_col: total count of each value in the variable
    :param bad_col: bad count of each value in the variable
    :param overallRate: the overall bad rate of the training set
    :return: the chi-square value
    '''
    df2 = df.copy()
    df2['expected'] = df[total_col].apply(lambda x: x*overallRate)
    combined = zip(df2['expected'], df2[bad_col])
    chi = [(i[0]-i[1])**2/i[0] for i in combined]
    chi2 = sum(chi)
    return chi2

def AssignGroup(x, bin):
    N = len(bin)
    if x<=min(bin):
        return min(bin)
    elif x>max(bin):
        return 10e10
    else:
        for i in range(N-1):
            if bin[i] < x <= bin[i+1]:
                return bin[i+1]


### ChiMerge_MaxInterval: split the continuous variable using Chi-square value by specifying the max number of intervals
def ChiMerge_MaxInterval(df, col, target, max_interval=5,special_attribute=[]):
    '''
    :param df: the dataframe containing splitted column, and target column with 1-0
    :param col: splitted column
    :param target: target column with 1-0
    :param max_interval: the maximum number of intervals. If the raw column has attributes less than this parameter, the function will not work
    :return: the combined bins
    '''
    colLevels = sorted(list(set(df[col])))
    N_distinct = len(colLevels)
    if N_distinct <= max_interval:  # If the raw column has attributes less than this parameter, the function will not work
        print("The number of original levels for {} is less than or equal to max intervals".format(col))
        return colLevels[:-1]
    else:
        if len(special_attribute)>=1:
            df1 = df.loc[df[col].isin(special_attribute)]
            df2 = df.loc[~df[col].isin(special_attribute)]
        else:
            df2 = df.copy()
        N_distinct = len(list(set(df2[col])))
        # Step 1: group the dataset by col and work out the total count & bad count in each level of the raw column
        if N_distinct > 100:
            ind_x = [int(i / 100.0 * N_distinct) for i in range(1, 100)]
            split_x = [colLevels[i] for i in ind_x]
            df2['temp'] = df2[col].map(lambda x: AssignGroup(x, split_x))
        else:
            df2['temp'] = df[col]
        total = df2.groupby(['temp'])[target].count()
        total = pd.DataFrame({'total': total})
        bad = df2.groupby(['temp'])[target].sum()
        bad = pd.DataFrame({'bad': bad})
        regroup = total.merge(bad, left_index=True, right_index=True, how='left')
        regroup.reset_index(level=0, inplace=True)
        N = sum(regroup['total'])
        B = sum(regroup['bad'])
        # the overall bad rate will be used in calculating expected bad count
        overallRate = B * 1.0 / N
        # initially, each single attribute forms a single interval
        # since we always combined the neighbours of intervals, we need to sort the attributes
        colLevels = sorted(list(set(df2['temp'])))
        groupIntervals = [[i] for i in colLevels]
        groupNum = len(groupIntervals)
        #the final splitted intervals should be the specified max intervals minus the number of special attributes
        split_intervals = max_interval - len(special_attribute)
        while (len(groupIntervals) > split_intervals):  # the termination condition: the number of intervals is equal to the pre-specified threshold
            # in each step of iteration, we calcualte the chi-square value of each atttribute
            chisqList = []
            for interval in groupIntervals:
                df2b = regroup.loc[regroup['temp'].isin(interval)]
                chisq = Chi2(df2b, 'total', 'bad', overallRate)
                chisqList.append(chisq)
            # find the interval corresponding to minimum chi-square, and combine with the neighbore with smaller chi-square
            min_position = chisqList.index(min(chisqList))
            if min_position == 0:
                combinedPosition = 1
            elif min_position == groupNum - 1:
                combinedPosition = min_position - 1
            else:
                if chisqList[min_position - 1] <= chisqList[min_position + 1]:
                    combinedPosition = min_position - 1
                else:
                    combinedPosition = min_position + 1
            groupIntervals[min_position] = groupIntervals[min_position] + groupIntervals[combinedPosition]
            # after combining two intervals, we need to remove one of them
            groupIntervals.remove(groupIntervals[combinedPosition])
            groupNum = len(groupIntervals)
        groupIntervals = [sorted(i) for i in groupIntervals]
        cutOffPoints = [max(i) for i in groupIntervals[:-1]]
        cutOffPoints = special_attribute + cutOffPoints
        return cutOffPoints

## determine whether the bad rate is monotone along the sortByVar
def BadRateMonotone(df, sortByVar, target,special_attribute = []):
    '''
    :param df: the dataset contains the column which should be monotone with the bad rate and bad column
    :param sortByVar: the column which should be monotone with the bad rate
    :param target: the bad column
    :param special_attribute: some attributes should be excluded when checking monotone
    :return:
    '''
    df2 = df.loc[~df[sortByVar].isin(special_attribute)]
    df2 = df2.sort_values([sortByVar])
    total = df2.groupby([sortByVar])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df2.groupby([sortByVar])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    combined = zip(regroup['total'],regroup['bad'])
    badRate = [x[1]*1.0/x[0] for x in combined]
    badRateMonotone = [badRate[i]<badRate[i+1] for i in range(len(badRate)-1)]
    Monotone = len(set(badRateMonotone))
    if Monotone == 1:
        return True
    else:
        return False

### If we find any categories with 0 bad, then we combine these categories with that having smallest non-zero bad rate
def MergeBad0(df,col,target):
    '''
     :param df: dataframe containing feature and target
     :param col: the feature that needs to be calculated the WOE and iv, usually categorical type
     :param target: good/bad indicator
     :return: WOE and IV in a dictionary
     '''
    total = df.groupby([col])[target].count()
    total = pd.DataFrame({'total': total})
    bad = df.groupby([col])[target].sum()
    bad = pd.DataFrame({'bad': bad})
    regroup = total.merge(bad, left_index=True, right_index=True, how='left')
    regroup.reset_index(level=0, inplace=True)
    regroup['bad_rate'] = regroup.apply(lambda x: x.bad*1.0/x.total,axis = 1)
    regroup = regroup.sort_values(by = 'bad_rate')
    col_regroup = [[i] for i in regroup[col]]
    for i in range(regroup.shape[0]):
        col_regroup[1] = col_regroup[0] + col_regroup[1]
        col_regroup.pop(0)
        if regroup['bad_rate'][i+1] > 0:
            break
    newGroup = {}
    for i in range(len(col_regroup)):
        for g2 in col_regroup[i]:
            newGroup[g2] = 'Bin '+str(i)
    return newGroup



def KS_AR(df, score, target, plot = False):
    '''
    :param df: the dataset containing probability and bad indicator
    :param score:
    :param target:
    :return:
    '''
    total = df.groupby([score])[target].count()
    bad = df.groupby([score])[target].sum()
    all = pd.DataFrame({'total':total, 'bad':bad})
    all['good'] = all['total'] - all['bad']
    all[score] = all.index
    all = all.sort_values(by=score, ascending=False)
    all.index = range(len(all))
    all['badCumRate'] = all['bad'].cumsum() / all['bad'].sum()
    all['goodCumRate'] = all['good'].cumsum() / all['good'].sum()
    all['totalPcnt'] = all['total'] / all['total'].sum()
    arList = [0.5 * all.loc[0, 'badCumRate'] * all.loc[0, 'totalPcnt']]
    for j in range(1, len(all)):
        ar0 = 0.5 * sum(all.loc[j - 1:j, 'badCumRate']) * all.loc[j, 'totalPcnt']
        arList.append(ar0)
    arIndex = (2 * sum(arList) - 1) / (all['good'].sum() * 1.0 / all['total'].sum())
    KS = all.apply(lambda x: x.badCumRate - x.goodCumRate, axis=1)
    KS1=KS
    KS = max(KS)
#    Index = range(1,len(KS1)+1)
#
#    Index = 1.0 * Index/len(Index)
    all['totalPcn'] = all['total'].cumsum() / all['total'].sum()
    index1 = KS1.idxmax()
    if plot:
        plt.figure(figsize=(6.6,6))
        plt.plot(all['totalPcn'], all['badCumRate'],color='red', linewidth=1.2, label='Accumulated early-stage sample persentage')
        plt.plot(all['totalPcn'], all['goodCumRate'],color='blue', linewidth=1.2, label='Accumulated late-stage sample persentage')
        plt.axvline(all['totalPcn'][index1], color='gray', linestyle='--')
        plt.axhline(all['badCumRate'][index1], color='gray', linestyle='--')
        plt.axhline(all['goodCumRate'][index1], color='gray', linestyle='--')
        plt.title('KS = ' + str(np.round(KS, 4)),fontsize=15)
        plt.xlabel('Accumulated sample persentage',fontsize=12)
        plt.tick_params(labelsize=12)
        plt.xlim(0, 1)
        plt.ylim(0, 1.01)
        plt.legend()
        plt.savefig("ks.eps",format="eps")
    return {'AR':arIndex, 'KS': KS}

def ROC_AUC( score, target, plot = True):
    fpr, tpr, thresholds  = roc_curve(score, target)
    plt.figure(figsize=(4.4,4))
    plt.plot(fpr, tpr, color='red',lw=1,label='ROC curve (area = %0.3f)' % roc_auc_score(score, target))
#    plt.legend('ROC curve', fontsize=20)
    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate',fontsize=10)
    plt.ylabel('True Positive Rate',fontsize=10)
    plt.tick_params(labelsize=10)
    plt.title('Receiver operating characteristic example',fontsize=13)
    plt.legend(loc="lower right")
    plt.savefig("auc.eps",format="eps")

def Prob2Score(prob, basePoint, PDO):
    #将概率转化成分数且为正整数
    y = np.log(prob/(1-prob))
    y2 = basePoint+PDO/np.log(2)*(-y)
    score = y2.astype("int")
    return score