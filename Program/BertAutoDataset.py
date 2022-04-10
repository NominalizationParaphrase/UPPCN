import pandas as pd
dfs=pd.read_excel("AutoACL2022_v4.xlsx")
dfs['adj/noun.2']=dfs['adj/noun']
for columns in dfs:
    for index,i in enumerate(dfs['sentence']):
        dfs.loc[index,'sentence']=" ".join(dfs.loc[index,'sentence'].split())
        if not pd.isnull(dfs.loc[index,columns]):
            dfs.loc[index,columns]=str(dfs.loc[index,columns]).strip()
        if "-" in dfs.loc[index,'adj/noun.2']:
            dfs.loc[index,'adj/noun.2']=dfs.loc[index,'adj/noun.2'].replace("-"," ")
from allennlp.predictors.predictor import Predictor
predictor = Predictor.from_path("bert-masked-lm-2019.09.17",cuda_device=0)
import spacy
nlp = spacy.load("en_core_web_sm")
import nltk
from nltk.corpus import wordnet as wn
import numpy as np
for index,i in enumerate(dfs['sentence']):
    dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
    if index<427:
        if pd.isnull(dfs.loc[index,'s']):
            dfs.loc[index,'v']=np.NaN
        if not pd.isnull(dfs.loc[index,'v']):
            print(index)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            wordChoice=[]
            MVOList=[]
            MVpOList=[]
            MVpOPassiveList=[]
            MVpOPassiveVMpOList=[]
            MpOVPassiveList=[]
            MpOVPassiveVMpOList=[]
            OVpMPassiveList=[]
            OVpMPassiveVOpMList=[]
            OVMList=[]
            OVpMList=[]
            VOMList=[]
            VOpMList=[]
            VpOpMList=[]
            VMOList=[]
            VMpOList=[]
            VpMpOList=[]
            MList=[]
            OVBList=[]
            OBVList=[]
            VOBList=[]
            BVOList=[]
            V1List=[]
            V2List=[]
            VMpOPrepList=[]
            VOpMPrepList=[]
            findVMpO=False
            findVOpM=False
            vWordList=dfs.loc[index,'v'].split()
            pluralWordList=dfs.loc[index,'pluralVerb'].split()
            pastParticipleVerbWordList=dfs.loc[index,'pastParticipleVerb'].split()
            for sWord in dfs.loc[index,'s'].split():
                wordChoice.append(sWord)
                if sWord[-1]=='s' or sWord[-2:]=='ch' or sWord[-2:]=='sh' or sWord[-1]=='z' or sWord[-1]=='x':
                    addSWord=sWord+'es'
                elif sWord[-1]=='y':
                    addSWord=sWord[0:-1]+'ies'
                else:
                    addSWord=sWord+'s'
                wordChoice.append(addSWord)
                if sWord[0].lower()=='a' or sWord[0].lower()=='e' or sWord[0].lower()=='i' or sWord[0].lower()=='u' or sWord[0].lower()=='o':
                    wordChoice.append("an "+sWord)
                else:
                    wordChoice.append("a "+sWord)
                wordChoice.append("the "+sWord)
                wordChoice.append("the "+addSWord)
            for word in wordChoice:
                plural=False
                plural2=False
                doc=nlp(word)
                for vWord,pluralWord,pastParticipleVerb in zip(vWordList,pluralWordList,pastParticipleVerbWordList):
                    verbWord=vWord
                    if doc[len(doc)-1].tag_=='NN' or  doc[len(doc)-1].tag_=='NNP' or "a" == word.split()[0] or "an" == word.split()[0]:
                        verbWord=pluralWord
                        plural=True
                    if doc[len(doc)-1].tag_=='NNS' or  doc[len(doc)-1].tag_=='NNPS':
                        if verbWord[-1]=='s' and verbWord[-2:]!='ss':
                            verbWord=verbWord[0:-1]
                            plural=False
                    MVO=word+" "+verbWord+" "+dfs.loc[index,'pobj']
                    MVOList.append(MVO)
                    MVpO=word+" "+verbWord+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
                    if dfs.loc[index,'prep']=='of':
                        MVpO=str(np.NaN)
                    MVpOList.append(MVpO)
                    if plural==True:
                        MVpOPassive=word+" is "+pastParticipleVerb+" [MASK] "+dfs.loc[index,'pobj']
                    else:
                        MVpOPassive=word+" are "+pastParticipleVerb+" [MASK] "+dfs.loc[index,'pobj']
                    result=predictor.predict(MVpOPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        MVpOPassive=MVpOPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        MVpOPassive=MVpOPassive.replace("[MASK] ","")
                        MVpOPassive=str(np.NaN)
                    MVpOPassiveList.append(MVpOPassive)
                    if plural==True:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" [MASK] "+dfs.loc[index,'pobj']+" is "+pastParticipleVerb
                    else:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" [MASK] "+dfs.loc[index,'pobj']+" are "+pastParticipleVerb
                    result=predictor.predict(MpOVPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='DT':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        MpOVPassive=MpOVPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        MpOVPassive=MpOVPassive.replace("[MASK] ","")
                    MpOVPassiveList.append(MpOVPassive)
                    doc2=nlp(dfs.loc[index,'pobj'].split()[-1])
                    verbWord2=vWord
                    if not "and" in dfs.loc[index,'pobj'].split():
                        print(dfs.loc[index,'pobj'].split()[0])
                        if doc2[len(doc2)-1].tag_=='NN' or  doc2[len(doc2)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                            verbWord2=pluralWord
                            plural2=True
                        if doc2[len(doc2)-1].tag_=='NNS' or  doc2[len(doc2)-1].tag_=='NNPS':
                            if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                                verbWord2=verbWord2[0:-1]
                                plural2=False
                    else:
                        plural2=False
                        if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                            verbWord2=verbWord2[0:-1]
                            plural2=False
                    OVM=dfs.loc[index,'pobj']+" "+verbWord2+" "+word
                    OVMList.append(OVM)
                    OVpM=dfs.loc[index,'pobj']+" "+verbWord2+" [MASK] "+word
                    result=predictor.predict(OVpM)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        OVpM=OVpM.replace("[MASK]",result['words'][0][0],1)
                    else:
                        OVpM=OVpM.replace("[MASK] ","")
                        OVpM=str(np.NaN)
                    OVpMList.append(OVpM)
                    if not pd.isnull(dfs.loc[index,'adv']):
                        for adv in dfs.loc[index,'adv'].split():
                            OVB=dfs.loc[index,'pobj']+" "+verbWord2+" "+adv
                            OVBList.append(OVB)
                            OBV=dfs.loc[index,'pobj']+" "+adv+" "+verbWord2
                            OBVList.append(OBV)
                    else:
                        OVB=str(np.NaN)
                        OVBList.append(OVB)
                        OBV=str(np.NaN)
                        OBVList.append(OBV)
                    if plural2==True:
                        OVpMPassive=dfs.loc[index,'pobj']+" is "+pastParticipleVerb+" [MASK] "+word
                    else:
                        OVpMPassive=dfs.loc[index,'pobj']+" are "+pastParticipleVerb+" [MASK] "+word
                    result=predictor.predict(OVpMPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        OVpMPassive=OVpMPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        OVpMPassive=OVpMPassive.replace("[MASK] ","")
                        OVpMPassive=str(np.NaN)
                    OVpMPassiveList.append(OVpMPassive)
                    if not pd.isnull(dfs.loc[index,'V2verbForV']):
                        V2=dfs.loc[index,'V2verbForV']+" "+dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
                    else:
                        V2=str(np.NaN)
                    V2List.append(V2)
                    vWord4=vWord
                    VOM=vWord4+" "+dfs.loc[index,'pobj']+" "+word
                    VOMList.append(VOM)
                    VOpM=vWord4+" "+dfs.loc[index,'pobj']+" [MASK] "+word
                    result=predictor.predict(VOpM)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        VOpM=VOpM.replace("[MASK]",result['words'][0][0],1)
                        VOpMPrepList.append(result['words'][0][0])
                    else:
                        VOpM=VOpM.replace("[MASK] ","")
                        VOpM=str(np.NaN)
                    VOpMList.append(VOpM)
                    if VOpM!=str(np.NaN):
                        VpOpM=VOpM.replace(vWord4+" "+dfs.loc[index,'pobj'],vWord4+" [MASK] "+dfs.loc[index,'pobj'])
                        result=predictor.predict(VpOpM)
                        find=False
                        for prep in result['words'][0]:
                            doc5=nlp(prep)
                            if doc5[0].tag_=='IN':
                                if prep == 'that':
                                    find=False
                                else:
                                    result['words'][0][0]=doc5[0].text
                                    find=True
                                    break
                        if find==True:
                            VpOpM=VpOpM.replace("[MASK]",result['words'][0][0],1)
                        else:
                            VpOpM=VpOpM.replace("[MASK] ","")
                            VpOpM=str(np.NaN)
                        VpOpMList.append(VpOpM)
                    VMO=vWord4+" "+word+" "+dfs.loc[index,'pobj']
                    VMOList.append(VMO)
                    VMpO=vWord4+" "+word+" [MASK] "+dfs.loc[index,'pobj']
                    result=predictor.predict(VMpO)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        VMpO=VMpO.replace("[MASK]",result['words'][0][0],1)
                        VMpOPrepList.append(result['words'][0][0])
                    else:
                        VMpO=VMpO.replace("[MASK] ","")
                        VMpO=str(np.NaN)
                    VMpOList.append(VMpO)
                    if VMpO!=str(np.NaN):
                        VpMpO=VMpO.replace(vWord4+" "+word,vWord4+" [MASK] "+word)
                        result=predictor.predict(VpMpO)
                        find=False
                        for prep in result['words'][0]:
                            doc5=nlp(prep)
                            if doc5[0].tag_=='IN':
                                if prep == 'that':
                                    find=False
                                else:
                                    result['words'][0][0]=doc5[0].text
                                    find=True
                                    break
                        if find==True:
                            VpMpO=VpMpO.replace("[MASK]",result['words'][0][0],1)
                        else:
                            VpMpO=VpMpO.replace("[MASK] ","")
                            VpMpO=str(np.NaN)
                        VpMpOList.append(VpMpO)
                if not pd.isnull(dfs.loc[index,'adv']):
                    for adv in dfs.loc[index,'adv'].split():
                        VOB=vWord4+" "+dfs.loc[index,'pobj']+" "+adv
                        VOBList.append(VOB)
                        BVO=adv+" "+vWord4+" "+dfs.loc[index,'pobj']
                        BVOList.append(BVO)
                else:
                    VOB=str(np.NaN)
                    VOBList.append(VOB)
                    BVO=str(np.NaN)
                    BVOList.append(BVO)
            doc3=nlp(dfs.loc[index,'n_v'])
            if doc3[len(doc3)-1].tag_=='NN' or  doc3[len(doc3)-1].tag_=='NNP':
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"is"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            elif doc3[len(doc3)-1].tag_=='NNS' or  doc3[len(doc3)-1].tag_=='NNPS':
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"are"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            else:
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"is"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            MList.append(M)
            if not pd.isnull(dfs.loc[index,'V1verbForV']):
                verbWord3=dfs.loc[index,'V1verbForV']
                if not "and" in dfs.loc[index,'pobj'].split():
                    doc7=nlp(dfs.loc[index,'pobj'].split()[-1])
                    if doc7[len(doc7)-1].tag_=='NN' or  doc7[len(doc7)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                        verbWord3=dfs.loc[index,'pluralVerbV1']
                    if doc7[len(doc7)-1].tag_=='NNS' or  doc7[len(doc7)-1].tag_=='NNPS':
                        if verbWord3[-1]=='s' and verbWord3[-2:]!='ss':
                            verbWord3=verbWord3[0:-1]
                else:
                    if verbWord3[-1]=='s' and verbWord3[-2:]!='ss':
                            verbWord3=verbWord3[0:-1]
                V1=dfs.loc[index,'pobj']+" "+verbWord3+" [MASK] "+dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']
                result=predictor.predict(V1)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT' or doc5[0].text=='a':
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    V1=V1.replace("[MASK]",result['words'][0][0],1)
                else:
                    V1=V1.replace("[MASK] ","")
            else:
                V1=str(np.NaN)
            V1List.append(V1)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            dfs.loc[index,'MVOList']="%".join(MVOList)
            dfs.loc[index,'MVpOList']="%".join(MVpOList)
            dfs.loc[index,'OVMList']="%".join(OVMList)
            dfs.loc[index,'OVpMList']="%".join(OVpMList)
            dfs.loc[index,'VOpMList']="%".join(VOpMList)
            dfs.loc[index,'VpOpMList']="%".join(VpOpMList)
            dfs.loc[index,'VMpOList']="%".join(VMpOList)
            dfs.loc[index,'VpMpOList']="%".join(VpMpOList)
            dfs.loc[index,'MList']="%".join(MList)
            dfs.loc[index,'OBVList']="%".join(OBVList)
            dfs.loc[index,'BVOList']="%".join(BVOList)
            dfs.loc[index,'OVBList']="%".join(OVBList)
            dfs.loc[index,'VOBList']="%".join(VOBList)
            dfs.loc[index,'V1List']="%".join(V1List)
            dfs.loc[index,'V2List']="%".join(V2List)
            dfs.loc[index,'MVpOPassiveList']="%".join(MVpOPassiveList)
            dfs.loc[index,'MpOVPassiveList']="%".join(MpOVPassiveList)
            dfs.loc[index,'OVpMPassiveList']="%".join(OVpMPassiveList)
            dfs.loc[index,'VMpOPrepList']="%".join(VMpOPrepList)
            dfs.loc[index,'VOpMPrepList']="%".join(VOpMPrepList)
for index,i in enumerate(dfs['sentence']):
    dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
    if index>=427:
        if not pd.isnull(dfs.loc[index,'v']):
            print(index)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            wordChoice=[]
            MVOList=[]
            MVpOList=[]
            MVpOPassiveList=[]
            MVpOPassiveVMpOList=[]
            MpOVPassiveList=[]
            MpOVPassiveVMpOList=[]
            OVpMPassiveList=[]
            OVpMPassiveVOpMList=[]
            OVMList=[]
            OVpMList=[]
            VOMList=[]
            VOpMList=[]
            VpOpMList=[]
            VMOList=[]
            VMpOList=[]
            VpMpOList=[]
            MList=[]
            OVBList=[]
            OBVList=[]
            VOBList=[]
            BVOList=[]
            V1List=[]
            V2List=[]
            VMpOPrepList=[]
            VOpMPrepList=[]
            findVMpO=False
            findVOpM=False
            vWordList=dfs.loc[index,'v'].split()
            pluralWordList=dfs.loc[index,'pluralVerb'].split()
            pastParticipleVerbWordList=dfs.loc[index,'pastParticipleVerb'].split()
            wordChoice.append(dfs.loc[index,'adj/noun.2'])
            if dfs.loc[index,'adj/noun.2'][-1]=='s' or dfs.loc[index,'adj/noun.2'][-2:]=='sh' or dfs.loc[index,'adj/noun.2'][-2:]=='ch' or dfs.loc[index,'adj/noun.2'][-1]=='x' or dfs.loc[index,'adj/noun.2'][-1]=='z':
                addSWord=dfs.loc[index,'adj/noun.2']+'es'
            elif dfs.loc[index,'adj/noun.2'][-1]=='y':
                addSWord=dfs.loc[index,'adj/noun.2'][0:-1]+'ies'
            else:
                addSWord=dfs.loc[index,'adj/noun.2']+'s'
            wordChoice.append(addSWord)
            if dfs.loc[index,'adj/noun.2'][0].lower()=='a' or dfs.loc[index,'adj/noun.2'][0].lower()=='e' or dfs.loc[index,'adj/noun.2'][0].lower()=='i' or dfs.loc[index,'adj/noun.2'][0].lower()=='u' or dfs.loc[index,'adj/noun.2'][0].lower()=='o':
                wordChoice.append("an "+dfs.loc[index,'adj/noun.2'])
            else:
                wordChoice.append("a "+dfs.loc[index,'adj/noun.2'])
            wordChoice.append("the "+dfs.loc[index,'adj/noun.2'])
            wordChoice.append("the "+addSWord)
            for word in wordChoice:
                plural=False
                plural2=False
                doc=nlp(word)
                for vWord,pluralWord,pastParticipleVerb in zip(vWordList,pluralWordList,pastParticipleVerbWordList):
                    verbWord=vWord
                    if doc[len(doc)-1].tag_=='NN' or  doc[len(doc)-1].tag_=='NNP' or "a" == word.split()[0] or "an" == word.split()[0]:
                        verbWord=pluralWord
                        plural=True
                    if doc[len(doc)-1].tag_=='NNS' or  doc[len(doc)-1].tag_=='NNPS':
                        if verbWord[-1]=='s' and verbWord[-2:]!='ss':
                            verbWord=verbWord[0:-1]
                            plural=False
                    MVO=word+" "+verbWord+" "+dfs.loc[index,'pobj']
                    MVOList.append(MVO)
                    MVpO=word+" "+verbWord+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
                    if dfs.loc[index,'prep']=='of':
                        MVpO=str(np.NaN)
                    MVpOList.append(MVpO)
                    if plural==True:
                        MVpOPassive=word+" is "+pastParticipleVerb+" [MASK] "+dfs.loc[index,'pobj']
                    else:
                        MVpOPassive=word+" are "+pastParticipleVerb+" [MASK] "+dfs.loc[index,'pobj']
                    result=predictor.predict(MVpOPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        MVpOPassive=MVpOPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        MVpOPassive=MVpOPassive.replace("[MASK] ","")
                        MVpOPassive=str(np.NaN)
                    MVpOPassiveList.append(MVpOPassive)
                    if plural==True:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" [MASK] "+dfs.loc[index,'pobj']+" is "+pastParticipleVerb
                    else:
                        MpOVPassive=word+" "+dfs.loc[index,'prep']+" [MASK] "+dfs.loc[index,'pobj']+" are "+pastParticipleVerb
                    result=predictor.predict(MpOVPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='DT':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        MpOVPassive=MpOVPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        MpOVPassive=MpOVPassive.replace("[MASK] ","")
                    MpOVPassiveList.append(MpOVPassive)
                    doc2=nlp(dfs.loc[index,'pobj'].split()[-1])
                    verbWord2=vWord
                    if not "and" in dfs.loc[index,'pobj'].split():
                        print(dfs.loc[index,'pobj'].split()[0])
                        if doc2[len(doc2)-1].tag_=='NN' or  doc2[len(doc2)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                            verbWord2=pluralWord
                            plural2=True
                        if doc2[len(doc2)-1].tag_=='NNS' or  doc2[len(doc2)-1].tag_=='NNPS':
                            if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                                verbWord2=verbWord2[0:-1]
                                plural2=False
                    else:
                        plural2=False
                        if verbWord2[-1]=='s' and verbWord2[-2:]!='ss':
                            verbWord2=verbWord2[0:-1]
                            plural2=False
                    OVM=dfs.loc[index,'pobj']+" "+verbWord2+" "+word
                    OVMList.append(OVM)
                    OVpM=dfs.loc[index,'pobj']+" "+verbWord2+" [MASK] "+word
                    result=predictor.predict(OVpM)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        OVpM=OVpM.replace("[MASK]",result['words'][0][0],1)
                    else:
                        OVpM=OVpM.replace("[MASK] ","")
                        OVpM=str(np.NaN)
                    OVpMList.append(OVpM)
                    if not pd.isnull(dfs.loc[index,'adv']):
                        for adv in dfs.loc[index,'adv'].split():
                            OVB=dfs.loc[index,'pobj']+" "+verbWord2+" "+adv
                            OVBList.append(OVB)
                            OBV=dfs.loc[index,'pobj']+" "+adv+" "+verbWord2
                            OBVList.append(OBV)
                    else:
                        OVB=str(np.NaN)
                        OVBList.append(OVB)  
                        OBV=str(np.NaN)
                        OBVList.append(OBV)
                    if plural2==True:
                        OVpMPassive=dfs.loc[index,'pobj']+" is "+pastParticipleVerb+" [MASK] "+word
                    else:
                        OVpMPassive=dfs.loc[index,'pobj']+" are "+pastParticipleVerb+" [MASK] "+word
                    result=predictor.predict(OVpMPassive)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        OVpMPassive=OVpMPassive.replace("[MASK]",result['words'][0][0],1)
                    else:
                        OVpMPassive=OVpMPassive.replace("[MASK] ","")
                        OVpMPassive=str(np.NaN)
                    OVpMPassiveList.append(OVpMPassive)
                    if not pd.isnull(dfs.loc[index,'V2verbForV']):
                        V2=dfs.loc[index,'V2verbForV']+" "+dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
                    else:
                        V2=str(np.NaN)
                    V2List.append(V2)
                    vWord4=vWord
                    VOM=vWord4+" "+dfs.loc[index,'pobj']+" "+word
                    VOMList.append(VOM)
                    VOpM=vWord4+" "+dfs.loc[index,'pobj']+" [MASK] "+word
                    result=predictor.predict(VOpM)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        VOpM=VOpM.replace("[MASK]",result['words'][0][0],1)
                        VOpMPrepList.append(result['words'][0][0])
                    else:
                        VOpM=VOpM.replace("[MASK] ","")
                        VOpM=str(np.NaN)
                    VOpMList.append(VOpM)
                    if VOpM!=str(np.NaN):
                        VpOpM=VOpM.replace(vWord4+" "+dfs.loc[index,'pobj'],vWord4+" [MASK] "+dfs.loc[index,'pobj'])
                        result=predictor.predict(VpOpM)
                        find=False
                        for prep in result['words'][0]:
                            doc5=nlp(prep)
                            if doc5[0].tag_=='IN':
                                if prep == 'that':
                                    find=False
                                else:
                                    result['words'][0][0]=doc5[0].text
                                    find=True
                                    break
                        if find==True:
                            VpOpM=VpOpM.replace("[MASK]",result['words'][0][0],1)
                        else:
                            VpOpM=VpOpM.replace("[MASK] ","")
                            VpOpM=str(np.NaN)
                        VpOpMList.append(VpOpM)
                    VMO=vWord4+" "+word+" "+dfs.loc[index,'pobj']
                    VMOList.append(VMO)
                    VMpO=vWord4+" "+word+" [MASK] "+dfs.loc[index,'pobj']
                    result=predictor.predict(VMpO)
                    find=False
                    for prep in result['words'][0]:
                        doc5=nlp(prep)
                        if doc5[0].tag_=='IN':
                            if prep == 'that':
                                find=False
                            else:
                                result['words'][0][0]=doc5[0].text
                                find=True
                                break
                    if find==True:
                        VMpO=VMpO.replace("[MASK]",result['words'][0][0],1)
                        VMpOPrepList.append(result['words'][0][0])
                    else:
                        VMpO=VMpO.replace("[MASK] ","")
                        VMpO=str(np.NaN)
                    VMpOList.append(VMpO)
                    if VMpO!=str(np.NaN):
                        VpMpO=VMpO.replace(vWord4+" "+word,vWord4+" [MASK] "+word)
                        result=predictor.predict(VpMpO)
                        find=False
                        for prep in result['words'][0]:
                            doc5=nlp(prep)
                            if doc5[0].tag_=='IN':
                                if prep == 'that':
                                    find=False
                                else:
                                    result['words'][0][0]=doc5[0].text
                                    find=True
                                    break
                        if find==True:
                            VpMpO=VpMpO.replace("[MASK]",result['words'][0][0],1)
                        else:
                            VpMpO=VpMpO.replace("[MASK] ","")
                            VpMpO=str(np.NaN)
                        VpMpOList.append(VpMpO)
            if not pd.isnull(dfs.loc[index,'adv']):
                for adv in dfs.loc[index,'adv'].split():
                    VOB=vWord4+" "+dfs.loc[index,'pobj']+" "+adv
                    VOBList.append(VOB)
                    BVO=adv+" "+vWord4+" "+dfs.loc[index,'pobj']
                    BVOList.append(BVO)
            else:
                VOB=str(np.NaN)
                VOBList.append(VOB)
                BVO=str(np.NaN)
                BVOList.append(BVO)
            doc3=nlp(dfs.loc[index,'n_v'])
            if doc3[len(doc3)-1].tag_=='NN' or  doc3[len(doc3)-1].tag_=='NNP':
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"is"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            elif doc3[len(doc3)-1].tag_=='NNS' or  doc3[len(doc3)-1].tag_=='NNPS':
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"are"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            else:
                M="[MASK] "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']+" "+"is"+" "+dfs.loc[index,'adj/noun.2']
                result=predictor.predict(M)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT':
                        if prep == 'that':
                            find=False
                        else:
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    M=M.replace("[MASK]",result['words'][0][0].lower(),1)
                else:
                    M=M.replace("[MASK] ","")
            MList.append(M)
            if not pd.isnull(dfs.loc[index,'V1verbForV']):
                verbWord3=dfs.loc[index,'V1verbForV']
                if not "and" in dfs.loc[index,'pobj'].split():
                    doc7=nlp(dfs.loc[index,'pobj'].split()[-1])
                    if doc7[len(doc7)-1].tag_=='NN' or  doc7[len(doc7)-1].tag_=='NNP' or "a" == dfs.loc[index,'pobj'].split()[0] or "an" == dfs.loc[index,'pobj'].split()[0]:
                        verbWord3=dfs.loc[index,'pluralVerbV1']
                    if doc7[len(doc7)-1].tag_=='NNS' or  doc7[len(doc7)-1].tag_=='NNPS':
                        if verbWord3[-1]=='s' and verbWord3[-2:]!='ss':
                            verbWord3=verbWord3[0:-1]
                else:
                    if verbWord3[-1]=='s' and verbWord3[-2:]!='ss':
                            verbWord3=verbWord3[0:-1]
                V1=dfs.loc[index,'pobj']+" "+verbWord3+" [MASK] "+dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']
                result=predictor.predict(V1)
                find=False
                for prep in result['words'][0]:
                    doc5=nlp(prep)
                    if doc5[0].tag_=='DT' or doc5[0].text=='a':
                            result['words'][0][0]=doc5[0].text
                            find=True
                            break
                if find==True:
                    V1=V1.replace("[MASK]",result['words'][0][0],1)
                else:
                    V1=V1.replace("[MASK] ","")
            else:
                V1=str(np.NaN)
            V1List.append(V1)
            dfs.loc[index,'originalPattern']=dfs.loc[index,'adj/noun.2']+" "+dfs.loc[index,'n_v']+" "+dfs.loc[index,'prep']+" "+dfs.loc[index,'pobj']
            dfs.loc[index,'MVOList']="%".join(MVOList)
            dfs.loc[index,'MVpOList']="%".join(MVpOList)
            dfs.loc[index,'OVMList']="%".join(OVMList)
            dfs.loc[index,'OVpMList']="%".join(OVpMList)
            dfs.loc[index,'VOpMList']="%".join(VOpMList)
            dfs.loc[index,'VpOpMList']="%".join(VpOpMList)
            dfs.loc[index,'VMpOList']="%".join(VMpOList)
            dfs.loc[index,'VpMpOList']="%".join(VpMpOList)
            dfs.loc[index,'MList']="%".join(MList)
            dfs.loc[index,'OBVList']="%".join(OBVList)
            dfs.loc[index,'BVOList']="%".join(BVOList)
            dfs.loc[index,'OVBList']="%".join(OVBList)
            dfs.loc[index,'VOBList']="%".join(VOBList)
            dfs.loc[index,'V1List']="%".join(V1List)
            dfs.loc[index,'V2List']="%".join(V2List)
            dfs.loc[index,'MVpOPassiveList']="%".join(MVpOPassiveList)
            dfs.loc[index,'MpOVPassiveList']="%".join(MpOVPassiveList)
            dfs.loc[index,'OVpMPassiveList']="%".join(OVpMPassiveList)
            dfs.loc[index,'VMpOPrepList']="%".join(VMpOPrepList)
            dfs.loc[index,'VOpMPrepList']="%".join(VOpMPrepList)
dfs.to_excel("FinalDatasetAutoDatasetBert.xlsx")