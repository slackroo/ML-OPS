import streamlit as st
import pandas as pd
import numpy as np
import joblib

hyperparams = ["n_est", "max_depth"]
metrics = ['roc_auc', 'average_precision', 'accuracy', 'precision', 'recall', 'f1']
metric_names = ['AUC', 'Av. Precision', "Accuracy", "Precision", "Recall", "F1"]
plots = ['cm', 'pr', 'roc']
plot_names = ['Confusion Matrix', 'Precision Recall Curve', 'ROC Curve']


# @st.cache
def load_runs():
    df = pd.read_csv("results.csv")
    df = df.drop("cm", axis=1)
    dftrain = df[df.data == 'train'].drop("data", axis=1).set_index('run_id')
    dftest = df[df.data == 'test'].drop("data", axis=1).set_index('run_id')
    return dftrain, dftest


def load_model(modelid):
    return joblib.load(modelid)


st.title('Run Details')

# Create a text element and let the reader know the data is loading.
data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data_train, data_test = load_runs()
# Notify the reader that the data was successfully loaded.
data_load_state.text('Loading data...done!')

qparam = st.experimental_get_query_params()
if "option" in qparam:
    option = qparam['option'][0]
else:
    option = st.selectbox(
        'Which run would you like to see?',
        list(data_train.index))

st.header(f"Run: {option}")

test_dict = dict(data_test.loc[option])
train_dict = dict(data_train.loc[option])

t1, t2 = st.tabs(['Metrics and Plots', 'Inference'])

with t1:
    for p in hyperparams:
        st.write(p, test_dict[p])

    st.subheader("Test Set")

    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        cols[i].metric(metric_names[i], round(test_dict[metrics[i]], 2))

    cols = st.columns(len(plots))
    for i, p in enumerate(plots):
        plotfile = f"plots/test_{option}_{p}.png"
        cols[i].image(plotfile, plot_names[i])

    st.subheader("Training Set")

    cols = st.columns(len(metrics))
    for i, m in enumerate(metrics):
        cols[i].metric(metric_names[i], round(train_dict[metrics[i]], 2))

    cols = st.columns(len(plots))
    for i, p in enumerate(plots):
        plotfile = f"plots/train_{option}_{p}.png"
        cols[i].image(plotfile, plot_names[i])

with t2:
    mappings = {
        'sex': {'female': 0, 'male': 1},
        'cp': {'Mild rare': 0, 'Mild frequent': 1, 'Strong Rare': 2, 'Strong frequent': 3},
        'fps': {'True': 1, 'False': 0},
        'exang': {'Yes': 1, 'No': 0}
    }

    col1, col2 = st.columns(2)

    cp_labels = ('Mild rare', 'Mild frequent', 'Strong Rare', 'Strong frequent')
    cps = (0, 1, 2, 3)
    cpmap = dict(zip(cp_labels, cps))
    cpval = col1.radio("Which Classification of chest pain patient is experiencing", cp_labels)

    sex_labels = ('female', 'male')
    sexval = col2.radio("Sex of Patient?", sex_labels)

    age = st.slider("Age", 0, 100, 40)
    trestbps = st.slider("Resting BP(mm Hg on admission to hospital)", 92, 220, 50)
    thalach = st.slider("Max Heart rate achieved ", 70, 210, 97)
    oldpeak = st.slider("ST depression induced by exercise relative to rest", 0.0, 10.0, 4.4)

    col1, col2 = st.columns(2)

    thal_labels = ('Normal', 'Fixed defect', 'Reversible defect', 'Dont_know')
    thals = (1, 2, 3, 0)
    thalmap = dict(zip(thal_labels, thals))
    thalval = col1.selectbox("Heart condition thal", thal_labels)

    exang_labels = ('Yes', 'No')
    exangs = (0, 1)
    exangmap = dict(zip(exang_labels, exangs))
    exangval = col2.selectbox("Was exercise induced angina", exang_labels)

    col1, col2 = st.columns(2)

    ca_labels = ('0', '1', '2', '3')
    cas = (0, 1, 2, 3)
    camap = dict(zip(ca_labels, cas))
    caval = col1.selectbox("number of major vessels (0-3) colored by flourosopy", ca_labels)

    FPS_labels = ('True', 'False')
    FPSs = (1, 0)
    FPSmap = dict(zip(FPS_labels, FPSs))
    FPSval = col2.selectbox("FPS (fasting blood sugar > 120 mg/dl) ", FPS_labels)

    col1, col2 = st.columns(2)

    slope_labels = ('0', '1','2')
    slopes = (0, 1, 2)
    slopemap = dict(zip(slope_labels, slopes))
    slopeval = col1.selectbox("the slope of the peak exercise ST segment ", slope_labels)

    restecg_labels = ('0', '1','2')
    restecgs = (0, 1, 2)
    restecgmap = dict(zip(restecg_labels, restecgs))
    restecgval = col2.selectbox("resting electrocardiographic results", restecg_labels)

    cholestrol = st.slider("serum cholestoral in mg/dl", 100, 800, 150)

    data_to_predict = dict(
        cp=cpmap[cpval],
        ca=camap[caval],
        trestbps=trestbps,
        thal=thalmap[thalval],
        age=age,
        oldpeak=oldpeak,
        thalach=thalach,
        exang=exangmap[exangval],
        sex=sexval,
        slope=slopemap[slopeval],
        restecg=restecgmap[restecgval],
        chol=cholestrol
    )

    for k in data_to_predict:
        if k in ['sex', 'FPSed']:
            data_to_predict[k] = mappings[k][data_to_predict[k]]

    rehydrated = load_model(f"models/{option}.joblib")
    inputX = np.array(list(data_to_predict.values())).reshape(1, -1)
    pred = rehydrated.predict(inputX)[0]
    proba = rehydrated.predict_proba(inputX)
    lifemap = {0: 'No_Heart_Disease', 1: 'Heart_Disease_present'}
    st.write(f"We predict **{lifemap[pred]}**. (Probability of having Heart disease: {proba[0][1]}).")
