from mlm.scorers import MLMScorer, MLMScorerPT, LMScorer
from mlm.models import get_pretrained
import mxnet as mx
import operator
import numpy as np
ctxs = [mx.gpu(0)] # or, e.g., [mx.gpu(0), mx.gpu(1)]
model, vocab, tokenizer = get_pretrained(ctxs, 'gpt2-117m-en-cased')
scorer = LMScorer(model, vocab, tokenizer, ctxs)
import pandas as pd
dfs=pd.read_excel("FinalDatasetAutoDatasetBert.xlsx",index_col=0)
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        print(index)
        findVMpO=False
        findVOpM=False
        sentencePatternList=['MVOList','MVpOList','OVMList','OVpMList','VOpMList','VpOpMList','VMpOList','VpMpOList','MList','OBVList','BVOList','OVBList','VOBList','V1List','V2List','MVpOPassiveList','MpOVPassiveList','OVpMPassiveList']
        sentencePatternNameList=['MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','M','OBV','BVO','OVB','VOB','V1','V2','MVpOPassive','MpOVPassive','OVpMPassive']
        for name,pattern in zip(sentencePatternNameList,sentencePatternList):
            if not pd.isnull(dfs.loc[index,pattern]):
                pattern=dfs.loc[index,pattern].split("%")
                pattern=[x for x in pattern if str(x) != 'nan']
                #print(name,pattern)
                tempResult=[]
                for choice in pattern:
                    if not pd.isnull(choice):
                        tempResult.append(scorer.score_sentences([choice])[0])
                        #print(choice,scorer.score_sentences([choice])[0])
                if tempResult!=[]:
                    dfs.loc[index,name]=pattern[tempResult.index(max(tempResult))]
                    if name=='VMpO':
                        VMpOList=pattern
                        bestVMpOIndex=tempResult.index(max(tempResult))
                        findVMpO=True
                    if name=='VOpM':
                        VOpMList=pattern
                        bestVOpMIndex=tempResult.index(max(tempResult))
                        findVOpM=True
        if findVMpO==True and not pd.isnull(dfs.loc[index,'MVpOPassive']):
            VMpOPrepList=dfs.loc[index,'VMpOPrepList'].split("%")
            for checkWord in dfs.loc[index,'pastParticipleVerb'].split():
                if checkWord in dfs.loc[index,'MVpOPassive']:
                    MVpOPassivePrep=dfs.loc[index,'MVpOPassive'].split()[dfs.loc[index,'MVpOPassive'].split().index(checkWord)+1]
            for findPrep,target in enumerate(VMpOList[bestVMpOIndex].split()):
                if target == VMpOPrepList[bestVMpOIndex]:
                    dfs.loc[index,'MVpOPassiveVMpO']=" ".join(VMpOList[bestVMpOIndex].split()[0:findPrep])+" "+MVpOPassivePrep+" "+" ".join(VMpOList[bestVMpOIndex].split()[findPrep+1:])
                    dfs.loc[index,'MpOVPassiveVMpO']=dfs.loc[index,'MVpOPassiveVMpO']
                    break
        else:
            dfs.loc[index,'MVpOPassive']=np.NaN
            dfs.loc[index,'MpOVPassive']=np.NaN
        if findVOpM==True and not pd.isnull(dfs.loc[index,'OVpMPassive']) :
            VOpMPrepList=dfs.loc[index,'VOpMPrepList'].split("%")
            for checkWord in dfs.loc[index,'pastParticipleVerb'].split():
                if checkWord in dfs.loc[index,'OVpMPassive']:
                    OVpMPassivePrep=dfs.loc[index,'OVpMPassive'].split()[dfs.loc[index,'OVpMPassive'].split().index(checkWord)+1]
            for findPrep,target in enumerate(VOpMList[bestVOpMIndex].split()):
                if target == VOpMPrepList[bestVOpMIndex]:
                    dfs.loc[index,'OVpMPassiveVOpM']=" ".join(VOpMList[bestVOpMIndex].split()[0:findPrep])+" "+OVpMPassivePrep+" "+" ".join(VOpMList[bestVOpMIndex].split()[findPrep+1:])
                    break
        else:
            dfs.loc[index,'OVpMPassive']=np.NaN
sentencePattern=['MVO','MVpO','OVM','OVpM','VOpM','VpOpM','VMpO','VpMpO','M','OBV','BVO','OVB','VOB','V1','V2','MVpOPassive','MpOVPassive','OVpMPassive']
for index,i in enumerate(dfs['sentence']):
    if not pd.isnull(dfs.loc[index,'v']):
        sentenceList=[]
        scoreDict={}
        for pattern in sentencePattern:
            if not pd.isnull(dfs.loc[index,pattern]):
                sentenceList.append(dfs.loc[index,pattern])
                scoreDict[pattern]=0
        #print(scoreDict)
        print(sentenceList)
        #print(subjectList)
        #print(verbList)
        #print(MVOSentenceList)
        tempResult=[]
        for choice,pattern in zip(sentenceList,scoreDict):
                scoreDict[pattern]=scorer.score_sentences([choice])[0]
                dfs.loc[index,pattern+"Score"]=scorer.score_sentences([choice])[0]
        dfs.loc[index,'HighestScorePattern'] = max(scoreDict, key=scoreDict.get)
dfs=dfs[['adj/noun', 'n_v', 'prep', 'pobj', 'adj/noun-label',
       'pobj-label', 'adj/noun.1', 'arg0', 'V', 'baseV', 'arg1', 'PP',
       'adverb', 'sentence', 'adv', 'V1verbForV', 'V2verbForV', 'pluralVerb',
       'pluralVerbV1', 'pluralVerbV2', 'pastParticipleVerb', 'originalPattern',
       'v','s',
       'MVO','MVpO','OVM', 'OVpM', 'VOpM', 'VpOpM', 'VMpO', 'VpMpO',
       'M', 'V1', 'V2', 'MVpOPassive', 'MpOVPassive', 'OVpMPassive','OBV','BVO',
       'OVB', 'VOB','OVpMPassiveVOpM','MVpOPassiveVMpO','MpOVPassiveVMpO', 'MVOScore','MVpOScore', 'OVMScore', 'OVpMScore',
       'VOpMScore', 'VpOpMScore', 'VMpOScore', 'VpMpOScore',
       'MScore', 'V1Score', 'V2Score', 'MVpOPassiveScore', 'MpOVPassiveScore',
       'OVpMPassiveScore','OBVScore','BVOScore', 'OVBScore',
       'VOBScore','HighestScorePattern']]
dfs.to_excel("FinalDatasetAutoGPT.xlsx")