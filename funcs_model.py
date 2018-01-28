from __init__ import *
pd.options.mode.chained_assignment = None

# ---------------- import
from sklearn import base
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import FunctionTransformer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import auc, roc_auc_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import precision_recall_curve

# models
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV

from sklearn.feature_selection import RFECV
from mlxtend.classifier import StackingClassifier
from mlxtend.classifier import StackingCVClassifier
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ColumnSelector
from sklearn.preprocessing import FunctionTransformer

#from glmnet import LogitNet
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE  
from imblearn.pipeline import Pipeline as ImbPipe
from sklearn.metrics import make_scorer

#from xgboost import XGBRegressor


def PowLog(X):
    X=pd.DataFrame(X)
    Xpow=X.apply(lambda row: np.power(row,2), axis=1)
    X = pd.concat([X, Xpow], axis=1)
    return X
    
FeaturePowLog = FunctionTransformer(PowLog)


#--------calculate fees
def get_fees(returns,FEE_BUY,FEE_SELL):
    returns=np.array(returns)
    v=np.repeat(0.0,len(returns))
    for i in np.arange(len(returns)):
        if returns[i] != 1: v[i] = (returns[i] * FEE_BUY + returns[i] * FEE_SELL)
    return v

#-----calculate ROI depending on cutoff value
def get_ROI(test,cutoff,P,strategy="reinvest"):
    
    '''================adds===========
    y_pred
    return
    return_cum
    lags
    trade
    fees
    '''
    y_pred =np.where(test.y_pred_prob.values >= cutoff, 1,0) 
    test['y_pred'] = y_pred
    test['return'] = 1.0
    test.loc[test.y_pred == 1,'return']  = test.loc[test.y_pred == 1,'target'] + 1
    
    test['lags'] = np.concatenate((np.array([0.0]),np.diff(np.array(test.timstamp), n=1, axis=-1)), axis=0)
    test['trade'] = test['y_pred']
    
    if (P['TARGET_TIME'] > 0) : 
        test['trade']  = skip_trades(test['trade'].values,test['lags'].values,P['TARGET_TIME'] )
        test.loc[test.trade == 0,'return']  = 1
        
    if (strategy=="reinvest"):
        test['fees'] = get_fees(test['return'],P['FEE_BUY'],P['FEE_SELL'])  
        test['return_cum'] = np.cumprod(np.array(test['return']) - np.array(test['fees']) )
        
    if (strategy=="fixed") : 
        test['fees'] = np.where(test['return'] != 1,(P['FEE_BUY']+P['FEE_SELL']),0)
        test['return_cum'] = 1 + np.cumsum(test['return'].values - 1) - np.array(test['fees']) 
    
    ROI = float(test['return_cum'].tail(1)) 
    return ROI

def eval_model(test,cutoff,PARAS):
    
    P=PARAS.copy()
    
    '''==================Auto find cutoff'''
    #if TYPE == "classify": bounds=(0,1)
    #if TYPE == "regression": bounds=(0,0.015)
    if type(cutoff) == list:
        rois = [get_ROI(test,c,PARAS) for c in cutoff]
        i_best = rois.index(np.max(rois))
        #best_cutoff=minimize_scalar(find_opt_ROI,bounds=bounds, method='bounded',options={'maxiter' : 500000})
        cutoff=cutoff[i_best]
    
    
    '''=================Add some vars'''
    test['time'] = [datetime.fromtimestamp(e) for e in test.timstamp.values]
    test['price_real'] = test.price_real.values /test.price_real.values[0]
    test['y_pred'] = y_pred =np.where(test.y_pred_prob.values >= cutoff, 1,0) 
    
    '''=================cutoff'''
    ROI = get_ROI(test,cutoff,P,strategy="reinvest")
    
    '''=================kpis'''
    trades = sum(test['trade'])
    days = (float(test.timstamp.tail(1)) - float(test.timstamp.head(1))) / 3600 / 24
    
    kpis=OrderedDict({
      'ROI': ROI, 
      'ROI_bench' : test.price_real.values[-1] / test.price_real.values[0],
      'ROI_day' : ROI/days,
      'trades' : trades, 
      'days' : days, 
      'max_trades' : confusion_matrix(test.target_cat,y_pred )[1,0],
      'days_trading' : len(set([str(datetime.fromtimestamp(e).date()) for e in test.iloc[np.where(y_pred == 1)[0],:]['timstamp'].values])),
      'cutoff' : cutoff,
      'precision' : precision_score(test.target_cat,y_pred,pos_label=1)
     })
    
    '''=================combine kpis'''
    del P['MODEL']
    P['FEATURES'] = [P['FEATURES']]
    for k in P.keys(): kpis[k] = P[k]
    
    return {'data' : test, 'kpis' : pd.DataFrame(kpis)}

#------once traded, wait until previous trade was completed. Skip test cases until gap > time_future
def skip_trades(trade,lags,seconds):
  
    counter=[0.0]
    counting = False

    for i in np.arange(len(trade)):

        # set value to 1
        if counting == True:
            trade[i] = 0
            counter.append(lags[i])

        # ON
        if trade[i] ==1: counting = True

        # OFF
        if sum(counter) > seconds:
            counter=[0.0]
            counting = False

    return trade

#-----optimization: find best cutoff for max ROI
from scipy.optimize import minimize_scalar

def find_opt_ROI(cutoff):
    y_pred=np.where(test.y_pred_prob >= cutoff, 1,0) 
    obj_ROI = get_ROI(test,y_pred,strategy="reinvest")
    roi=float(obj_ROI['kpis']['ROI'].values)
    if (roi < 0) : roi = 0.0001
    return float(-roi)



def expandgrid(*itrs):
   product = list(itertools.product(*itrs))
   return {'Var{}'.format(i+1):[x[i] for x in product] for i in range(len(itrs))}



#-----------precision custom scorer
def custom_precision(y_true,probas_pred): 
    pos_label=1
    precision, recall, _ = precision_recall_curve(y_true,probas_pred,pos_label=pos_label)
    cut = np.where(precision >= 0.5)[0]
    if len(cut) <= 1: area = 0
    else: area = auc(recall[cut], precision[cut])
    area = auc(recall, precision)
    return area

def custom_precision_wrapper(y_true,probas_pred):
    return custom_precision(y_true,probas_pred[:, 1])

#------MAXIMIZE: bigger is better!
score_custom_precision = make_scorer(custom_precision_wrapper,greater_is_better=True,needs_proba=True)

#------------custom regression scorers

def custom_score_regression1(y_true,y_pred): 
    
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    
    #  money 
    i_in = np.where(y_true >= THRES_PROF)[0]
    i_out = np.where(y_true < THRES_PROF)[0]
    
    
    # >= 0.0042 in the money
    list_i = []
    for i in i_in:
        if y_pred[i] >= y_true[i]: list_i.append( (y_pred[i]- y_true[i]) * 1 )
        if y_pred[i] < y_true[i]: list_i.append( (y_true[i]- y_pred[i] ) * 0 )
    
    # < 0.0042 out of money 
    for i in i_out:
        if y_pred[i] >= THRES_PROF: list_i.append( abs(y_pred[i] - y_true[i]) * 10 )
        if y_pred[i] < THRES_PROF: list_i.append(0)
    
    out = sum(np.array(list_i)) / (len(i_in))
    print('in, out, score: ',len(i_in), len(i_out) ,out)
    
    return out

def custom_score_regression2(y_true,y_pred): 
    
    
    THRES=TRAIN_THRES
    
    y_pred=np.array(y_pred)
    y_true=np.array(y_true)
    
    # punish false positives
    i_wrong = np.where((y_pred > THRES) & (y_true <= THRES))[0]
    wrong = y_true[i_wrong] - THRES
    wrong = abs(wrong * 3)
    loss1 = np.sum(wrong)
    av_loss1 = loss1/len(wrong)
    
    # punish false negatives
    i_missed = np.where((y_pred < THRES) & (y_true >= THRES))[0]
    missed = y_true[i_missed] - THRES
    missed = abs(missed)
    loss2 = np.sum(missed * 2)
    av_loss2 = loss2/len(missed)
    
    out=np.nansum([av_loss1,av_loss2])
    
    print(len(wrong), len(missed), out)
    return out

score_custom_regression = make_scorer(custom_score_regression2,greater_is_better=False) #MINIMIZE: smaller is better!

def backtesting(d,P):
    
    import warnings
    warnings.filterwarnings("ignore")
    
    '''=========== test data ===========
    timstamp      unix timestamp
    price_real    real price
    target        % delta price in future
    target_cat    % 0=dont trade, 1=trade
    y_pred_prob   probabilty predictions
    '''
    '''=========== evlauation function adds'''

    gc.collect()
    cv=TimeSeriesSplit(n_splits=P['BACKTEST_SPLITS'],max_train_size=P['N_TRAIN_MIN'])
    test_list = []
    best_params=[]
    for i in cv.split(d):
        
        i_train, i_test = i
        
        if (len(i_train) >= P['N_TRAIN_MIN']):
            
            # Set training N
            i_train = i_train[-P['N_TRAIN']:] 
            X=d.iloc[i_train,:]
            X = X[(X.target >= P['TRAIN_THRES']) | (X.target < P['THRES_PROF']) ] 
            
            target_freq = np.sum(X.target_cat) / P['N_TRAIN'] * 2
            #new=d.iloc[0:i_train[0],:]
            #new=new[(new.target >= TRAIN_THRES)]
            #X = new.append(X)
            
            print('from_i:',i_test[0],'to_i:',i_test[-1],'targets_total:',X.target_cat.sum(),'targets_prop',sum(X.target_cat)/X.shape[0])
            
            if P['TYPE'] == "classify": y= X.target_cat    
            if P['TYPE'] == "regression": y= X.target
                
            X=X.loc[:,P['FEATURES']]
            test=d.iloc[i[1],:]
            
            MODEL = P['MODEL']
            MODEL.fit(X,y)

            if (hasattr(MODEL,'best_estimator_')) == True: PREDICTOR = MODEL.best_estimator_
            if (hasattr(MODEL,'best_estimator_')) == False: PREDICTOR = MODEL

            if P['TYPE'] == "classify":
                y_pred_prob2=PREDICTOR.predict_proba(test.loc[:,P['FEATURES']])
                test['y_pred_prob'] = np.asarray([p[1] for p in y_pred_prob2 ])

            if P['TYPE'] == "regression":
                test['y_pred_prob'] = PREDICTOR.predict(test.loc[:,P['FEATURES']])

            if (hasattr(MODEL,'best_estimator_')):
                best_params.append(MODEL.best_params_)
                print(MODEL.best_params_, MODEL.best_score_)

            test = test[['timstamp','price_real','target','target_cat','y_pred_prob']]
   
            test_list.append(test)

    test = pd.concat(test_list)
    return test