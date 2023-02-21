import streamlit as st
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords 
import contractions
from PIL import Image
import string
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

punct=string.punctuation
nltk.download('popular')
@st.cache
def convert_df(df):
     # IMPORTANT: Cache the conversion to prevent computation on every rerun
     return df.to_csv(index=False).encode('utf-8')


col1, col2, col3 = st.columns([1,6,1])
with col2:
    st.header(":green[Simple Text Preprocessing App!]")


Top=st.container()
Medium=st.container()
with Top:
    
    
    st.markdown("---")
with st.sidebar:
    st.markdown("FILTER OPTIONS")
    radio_selection_1=st.radio("Mode of input",["Text Input","Excel/CSV Upload"],help="Choose 2nd option to uplaod file!",key='sf')
    with st.form("Filter"):
        if radio_selection_1=="Text Input":
            st.markdown("Select based on preference")
            radio_selection=st.radio("Type of Formatting",["Original Format","Sentence by Sentence"],help="Choose 2nd option to display sentence wise!")
        
        st.markdown("Select techniques to be applied")
        L=st.checkbox('Lemmatization',value=True)
        S=st.checkbox('Stemming')
        Lo=st.checkbox('LowerCase')
        C=st.checkbox('Contractions') 
        St=st.checkbox('StopWords') 
        P=st.checkbox('Punctuations')
        st.form_submit_button("Submit")
with Medium:
    if radio_selection_1=="Text Input":
        val=st.text_input("Enter the text to standardise... (It can be a paragraph or a one liner)")
        flag=0
        val=nltk.sent_tokenize(val)
        if flag==0:
            if Lo:
                for x in range(len(val)):
                    current=val[x] 
                    val1=current.split()
                    val1=list(map(str.lower,val1))
                    val1=" ".join(val1)
                    val[x]=val1 
            if St:
                for x in range(len(val)):
                    current=val[x] 
                    val1=current.split()
                    temp=""
                    for item in val1:
                        if item not in stopwords.words('ENGLISH'):
                            temp+=item
                            temp+=" "
                    val[x]=temp
            if P:
                for x in range(len(val)):
                    current=val[x] 
                    temp=""
                    for item in current:
                        if item not in punct:
                            temp+=item
                    val[x]=temp
            if L:
                b=WordNetLemmatizer()
                for x in range(len(val)):
                    current=val[x] 
                    val1=current.split()
                    val1=list(map(b.lemmatize,val1))
                    val1=" ".join(val1)
                    val[x]=val1 

            if S:
                b=PorterStemmer()
                for x in range(len(val)):
                    current=val[x] 
                    val1=current.split()
                    val1=list(map(b.stem,val1))
                    val1=" ".join(val1)
                    val[x]=val1 

            if C:
                for x in range(len(val)):
                    current=val[x] 
                    val1=current.split()
                    val1=list(map(contractions.fix,val1))
                    val1=" ".join(val1)
                    val[x]=val1
            if len(val)==0:
                st.text("Please enter something to preprocess. I am waiting :(")
            else:
                st.text("Here is your preprocessed text.. Hurray!!!!")
            tempo=val
            checker=0
            if radio_selection=="Original Format":
                val=" ".join(val) 
                val_copy=val
                if len(val)>=101:
                    val_copy=val[:100]+"......."
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.text("")
                st.markdown(f'<p style="color:cornflowerblue;font-size:20px;border-radius:2%;">{val_copy}</p>', unsafe_allow_html=True) 
                if len(val)!=0:
                    st.download_button('Download the preprocessed text', val)
                    checker=1

            else:
                count=1
                no=[]
                sentence=[]
                for item in val:
                    no.append(count)
                    sentence.append(item) 
                    count+=1
                df=pd.DataFrame({"No":no,"Sentence":sentence})
                df = df.reset_index(drop=True)
                st.markdown('##')
                if len(val)!=0:
                    st.dataframe(df)
                    csv = convert_df(df)
                    st.download_button(
                        label="Download the table as a csv file !",
                        data=csv,
                        file_name='text_preprocess',
                        mime='text/csv',
                    )
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")
            st.text("")

            if len(val)!=0:
                with st.expander("Expand/Minimize to access insights into the textual data!"):
            
                    st.markdown('##')

                    length=[]

                    col11,col22=st.columns(2)
                    with col11:
                        st.markdown("Words per Sentence Distribution")
                        for item in tempo:
                            length.append(len(item.split()))
                        fig = plt.figure(figsize=(10, 4))
                        sns.histplot(length,bins=10)
                        plt.xlabel("Words per Sentence")
                        plt.ylabel("Frequency")
                        st.pyplot(fig)      


                    with col22:
                        st.markdown("Chars per Sentence Distribution")
                        length1=[]
                        for item in tempo:
                            length1.append(len(item))
                        fig = plt.figure(figsize=(10, 4))
                        sns.histplot(length1,bins=10)
                        plt.xlabel("Chars per Sentence")
                        plt.ylabel("Frequency")
                        st.pyplot(fig) 
                    dic={}   
                    for item in tempo:
                        tempo1=item.split()
                        for x in tempo1:
                            if x in dic:
                                dic[x]+=1
                            else:
                                dic[x]=1
                    words=dic.keys()
                    values=dic.values()
                    df1=pd.DataFrame({"Words":words,"Frequency":values}) 
                    df1=df1.sort_values(by="Frequency",ascending=False).head(10)
                    fig = plt.figure(figsize=(10, 4))
                    sns.barplot(x=df1["Frequency"],y=df1["Words"])
                    st.markdown("Top 10 frequent words in the corpus!")
                    st.pyplot(fig)
    else:
            def Lo_(val):
                print(val)
                if val is not None and val not in [np.nan]:
                    val=nltk.sent_tokenize(val)
                    for x in range(len(val)):
                        current=val[x]
                        val1=current.split()
                        val1=list(map(str.lower,val1))
                        val1=" ".join(val1)
                        val[x]=val1 
                    return ".".join(val)
                else:
                    return ""

            def St_(val):
                if val is not None and val not in [np.nan]:
                    val=nltk.sent_tokenize(val)
                    for x in range(len(val)):
                        current=val[x] 
                        val1=current.split()
                        temp=""
                        for item in val1:
                            if item not in stopwords.words('ENGLISH'):
                                temp+=item
                                temp+=" "
                        val[x]=temp
                    return ".".join(val)
                else:
                    return ""
            def P_(val):
                if val is not None and val not in [np.nan]:
                    val=nltk.sent_tokenize(val)
                    for x in range(len(val)):
                        current=val[x] 
                        temp=""
                        for item in current:
                            if item not in punct:
                                temp+=item
                        val[x]=temp
                    return ".".join(val)
                else:
                    return ""
            def L_(val):
                print(val)
                if val is not None and val not in [np.nan]:
                    val=str(val)
                    print(val)
                    val=nltk.sent_tokenize(val)
                    b=WordNetLemmatizer()
                    for x in range(len(val)):
                        current=val[x] 
                        val1=current.split()
                        val1=list(map(b.lemmatize,val1))
                        val1=" ".join(val1)
                        val[x]=val1 
                    return ".".join(val)
                else:
                    return ""
                    

            def S_(val):
                if val is not None and val not in [np.nan]:
                    val=nltk.sent_tokenize(val)
                    b=PorterStemmer()
                    for x in range(len(val)):
                        current=val[x] 
                        val1=current.split()
                        val1=list(map(b.stem,val1))
                        val1=" ".join(val1)
                        val[x]=val1 
                    return ".".join(val)
                else:
                    return ""
                
                

            def C_(val):
                if val is not None and val not in [np.nan]:
                    val=nltk.sent_tokenize(val)
                    for x in range(len(val)):
                        current=val[x] 
                        val1=current.split()
                        val1=list(map(contractions.fix,val1))
                        val1=" ".join(val1)
                        val[x]=val1
                    return ".".join(val)
                else:
                    return ""
            encoding_list = ['unicode_escape','ascii','utf-8', 'big5', 'big5hkscs', 'cp037', 'cp273', 'cp424', 'cp437', 'cp500', 'cp720', 'cp737'
                 , 'cp775', 'cp850', 'cp852', 'cp855', 'cp856', 'cp857', 'cp858', 'cp860', 'cp861', 'cp862'
                 , 'cp863', 'cp864', 'cp865', 'cp866', 'cp869', 'cp874', 'cp875', 'cp932', 'cp949', 'cp950'
                 , 'cp1006', 'cp1026', 'cp1125', 'cp1140', 'cp1250', 'cp1251', 'cp1252', 'cp1253', 'cp1254'
                 , 'cp1255', 'cp1256', 'cp1257', 'cp1258', 'euc_jp', 'euc_jis_2004', 'euc_jisx0213', 'euc_kr'
                 , 'gb2312', 'gbk', 'gb18030', 'hz', 'iso2022_jp', 'iso2022_jp_1', 'iso2022_jp_2'
                 , 'iso2022_jp_2004', 'iso2022_jp_3', 'iso2022_jp_ext', 'iso2022_kr', 'latin_1', 'iso8859_2'
                 , 'iso8859_3', 'iso8859_4', 'iso8859_5', 'iso8859_6', 'iso8859_7', 'iso8859_8', 'iso8859_9'
                 , 'iso8859_10', 'iso8859_11', 'iso8859_13', 'iso8859_14', 'iso8859_15', 'iso8859_16', 'johab'
                 , 'koi8_r', 'koi8_t', 'koi8_u', 'kz1048', 'mac_cyrillic', 'mac_greek', 'mac_iceland', 'mac_latin2'
                 , 'mac_roman', 'mac_turkish', 'ptcp154', 'shift_jis', 'shift_jis_2004', 'shift_jisx0213', 'utf_32'
                 , 'utf_32_be', 'utf_32_le', 'utf_16', 'utf_16_be', 'utf_16_le', 'utf_7', 'utf_8_sig']
        
            @st.cache(allow_output_mutation=True)
            def read_file(files):
                val=files
                try:
                    df=pd.read_csv(val,engine='python',encoding='utf-8,',encoding_errors='ignore')
                    return df
                except Exception as e:
                    try:
                        for item in encoding_list:
                            try:
                            
                                df=pd.read_csv(val,encoding=item,engine='python',encoding_errors='ignore')
                                return df
                            except:
                                pass
                        df=pd.read_csv(files)

                    except Exception as e:
                        try:
                            df=pd.read_excel(files)
                            return df 
                        except Exception as e:
                            print(e)
                            st.write(e)


            def multiselect():
                st.session_state.multiselect=[]


            file=st.file_uploader("Upload the file to standardise",type=['csv','xls','xlsx'])
            st.text("")
            st.text("")
            st.text("")
            if file:
                df=read_file(file)
                print(df)
                columns=df.columns 
                columns_=[]
                for ite in columns:
                    if df[ite].dtype=="object":
                        columns_.append(ite)
                    else:
                        pass 
                
                cols=st.multiselect("Select the columns to standardise",list(columns_),key='multiselect')
                with st.spinner("Preprocessing your data with "+str(df.shape[0])+" rows.. Kindly hold on...."):
                    if cols!=list():
                        for item in cols:
                            if Lo:
                                df[item]=df[item].apply(Lo_)
                            if St:
                                df[item]=df[item].apply(St_)
                            if P:
                                df[item]=df[item].apply(P_)
                            if L:

                                df[item]=df[item].apply(L_)
                            if S:
                                df[item]=df[item].apply(S_)
                            if C:
                                df[item]=df[item].apply(C_)
                        st.text("")
                        st.text("")
                        st.text("")
                        st.text("")
                        st.text("")
                        st.text("We have succesfully preprocessed the columns of your choice!")
                        df_processed=df.to_csv(index=False).encode('utf-8')
                        down=st.download_button(
                                label="Download the preprocessed data !",
                                data=df_processed,
                                file_name='text_preprocess.csv',
                                mime='text/csv',
                                on_click=multiselect
                            )
                           
                        
            

                    

                
    



    
    
    





    

