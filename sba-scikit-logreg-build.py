import pandas as pd
from sklearn.linear_model import LogisticRegression
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import onnx

# Load data
df_sba_case = pd.read_csv('Data/SBAcase.csv')

# Features and target
predictors = ['RealEstate', 'Portion', 'Recession']
df_train = df_sba_case[df_sba_case['Selected'] == 1].copy()

X_train = df_train[predictors]
y_train = df_train['Default']

# Train
sk_logreg = LogisticRegression(max_iter=1000)
sk_logreg.fit(X_train, y_train)

# Export
initial_type = [('float_input', FloatTensorType([None, len(predictors)]))]

# IMPORTANT: tell skl2onnx to output probabilities
onnx_model = convert_sklearn(
    sk_logreg,
    initial_types=initial_type,
    options={id(sk_logreg): {'zipmap': False}}  # disables zipmap → outputs a single tensor of probabilities
)

with open("sba_sk_logreg.onnx", "wb") as f:
    f.write(onnx_model.SerializeToString())