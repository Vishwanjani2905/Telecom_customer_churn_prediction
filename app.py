from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import pickle
import json

app = Flask(__name__)
app.secret_key = 'telecom_churn_secret_key_2024'

# ── Load model ────────────────────────────────────────────────────────────────
model             = None
expected_features = None

try:
    raw = pickle.load(open('model.sav', 'rb'))
    print(f"✅ model.sav loaded — type: {type(raw)}")

    # Case 1: direct sklearn estimator
    if hasattr(raw, 'predict') and hasattr(raw, 'predict_proba'):
        model = raw
    # Case 2: dict wrapping the model
    elif isinstance(raw, dict):
        for key in ('model', 'classifier', 'clf', 'estimator', 'rf', 'pipeline'):
            if key in raw and hasattr(raw[key], 'predict'):
                model = raw[key]
                print(f"   Extracted model from dict key '{key}'")
                break
        if model is None:
            for v in raw.values():
                if hasattr(v, 'predict') and hasattr(v, 'predict_proba'):
                    model = v
                    break
    # Case 3: list / tuple
    elif isinstance(raw, (list, tuple)):
        for item in raw:
            if hasattr(item, 'predict') and hasattr(item, 'predict_proba'):
                model = item
                break

    if model is not None:
        print(f"✅ Model ready — {type(model).__name__}")
        if hasattr(model, 'feature_names_in_'):
            expected_features = model.feature_names_in_.tolist()
            print(f"   feature_names_in_: {len(expected_features)} features")
        else:
            print("   No feature_names_in_ — will infer from dataset sample.")
    else:
        print("⚠️  Could not extract usable estimator — Demo mode.")

except FileNotFoundError:
    print("⚠️  model.sav not found — Demo mode (rule-based predictions).")
except Exception as e:
    print(f"❌ Error loading model: {e}")

# ── Load dataset ──────────────────────────────────────────────────────────────
df_original = None
try:
    df_original = pd.read_csv('Telco-Customer-Churn.csv')
    df_original['TotalCharges'] = pd.to_numeric(df_original['TotalCharges'], errors='coerce')
    df_original.dropna(inplace=True)
    df_original['Churn'] = df_original['Churn'].map({'Yes': 1, 'No': 0})
    print(f"✅ Dataset loaded — {len(df_original)} rows")
except FileNotFoundError:
    print("⚠️  Telco-Customer-Churn.csv not found — charts will use sample data.")

# ── Constants ─────────────────────────────────────────────────────────────────
USD_TO_INR = 83   # 1 USD ≈ ₹83

# SeniorCitizen is shown as Yes/No in the form; mapped to 0/1 before modelling
CATEGORICAL_FEATURES = {
    'gender':           ['Male', 'Female'],
    'SeniorCitizen':    ['No', 'Yes'],
    'Partner':          ['Yes', 'No'],
    'Dependents':       ['Yes', 'No'],
    'PhoneService':     ['Yes', 'No'],
    'MultipleLines':    ['No phone service', 'No', 'Yes'],
    'InternetService':  ['DSL', 'Fiber optic', 'No'],
    'OnlineSecurity':   ['No', 'Yes', 'No internet service'],
    'OnlineBackup':     ['No', 'Yes', 'No internet service'],
    'DeviceProtection': ['No', 'Yes', 'No internet service'],
    'TechSupport':      ['No', 'Yes', 'No internet service'],
    'StreamingTV':      ['No', 'Yes', 'No internet service'],
    'StreamingMovies':  ['No', 'Yes', 'No internet service'],
    'Contract':         ['Month-to-month', 'One year', 'Two year'],
    'PaperlessBilling': ['Yes', 'No'],
    'PaymentMethod':    ['Electronic check', 'Mailed check',
                         'Bank transfer (automatic)', 'Credit card (automatic)'],
}
NUMERIC_FEATURES = ['tenure', 'MonthlyCharges', 'TotalCharges']


# ── Infer feature list from dataset when model doesn't expose it ──────────────
_inferred_features = None

def get_expected_features():
    global _inferred_features
    if expected_features:
        return expected_features
    if _inferred_features is not None:
        return _inferred_features
    if df_original is None:
        return None
    sample = df_original.drop(columns=['Churn', 'customerID'], errors='ignore').head(20).copy()
    sample['SeniorCitizen'] = sample['SeniorCitizen'].astype(int)
    cat_cols = [c for c in CATEGORICAL_FEATURES if c != 'SeniorCitizen' and c in sample.columns]
    enc = pd.get_dummies(sample, columns=cat_cols)
    _inferred_features = enc.columns.tolist()
    print(f"   Inferred {len(_inferred_features)} features from dataset sample.")
    return _inferred_features


# ── Preprocessing ─────────────────────────────────────────────────────────────
def preprocess_input(input_data: dict) -> pd.DataFrame:
    df = pd.DataFrame([input_data])

    # Convert SeniorCitizen Yes/No → 0/1
    df['SeniorCitizen']  = df['SeniorCitizen'].map({'Yes': 1, 'No': 0}).fillna(0).astype(int)
    df['TotalCharges']   = pd.to_numeric(df['TotalCharges'],   errors='coerce').fillna(0)
    df['MonthlyCharges'] = pd.to_numeric(df['MonthlyCharges'], errors='coerce').fillna(0)
    df['tenure']         = pd.to_numeric(df['tenure'],         errors='coerce').fillna(0)

    cat_cols = [c for c in CATEGORICAL_FEATURES if c != 'SeniorCitizen' and c in df.columns]
    df_enc   = pd.get_dummies(df, columns=cat_cols)

    feats = get_expected_features()
    if feats:
        for col in feats:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[[c for c in feats if c in df_enc.columns]]
        for col in feats:
            if col not in df_enc.columns:
                df_enc[col] = 0
        df_enc = df_enc[feats]

    return df_enc


# ── Rule-based demo prediction ────────────────────────────────────────────────
def demo_predict(data: dict):
    score = 0
    if data.get('Contract') == 'Month-to-month':    score += 30
    elif data.get('Contract') == 'One year':         score += 12
    else:                                             score += 2
    if data.get('InternetService') == 'Fiber optic': score += 15
    elif data.get('InternetService') == 'DSL':        score += 5
    if data.get('PaymentMethod') == 'Electronic check':  score += 18
    elif data.get('PaymentMethod') == 'Mailed check':    score += 10
    else:                                                 score += 3
    if data.get('OnlineSecurity') == 'No': score += 10
    if data.get('TechSupport') == 'No':    score += 8
    if data.get('SeniorCitizen') in ('Yes', 1, '1'): score += 6
    if data.get('Partner') == 'No':        score += 5
    score += max(0, (float(data.get('MonthlyCharges', 65)) - 50) * 0.3)
    score -= min(40, float(data.get('tenure', 12)) * 0.7)
    prob = min(0.97, max(0.05, score / 100))
    return int(prob >= 0.5), [round(1 - prob, 4), round(prob, 4)]


# ── SHAP-style factors ────────────────────────────────────────────────────────
def get_shap_factors(data: dict, prediction: int) -> list:
    factors = []
    contract = data.get('Contract', '')
    if contract == 'Month-to-month':
        factors.append({'label': 'Month-to-month contract',  'value': 0.32, 'direction': 'negative'})
    elif contract == 'One year':
        factors.append({'label': '1-year contract',           'value': 0.18, 'direction': 'positive'})
    else:
        factors.append({'label': '2-year contract',           'value': 0.35, 'direction': 'positive'})

    payment = data.get('PaymentMethod', '')
    if payment == 'Electronic check':
        factors.append({'label': 'Electronic check payment',  'value': 0.24, 'direction': 'negative'})
    elif payment in ('Bank transfer (automatic)', 'Credit card (automatic)'):
        factors.append({'label': 'Automatic payment method',  'value': 0.15, 'direction': 'positive'})
    else:
        factors.append({'label': 'Manual payment method',     'value': 0.09, 'direction': 'negative'})

    inet = data.get('InternetService', '')
    if inet == 'Fiber optic':
        factors.append({'label': 'Fiber optic internet',      'value': 0.18, 'direction': 'negative'})
    elif inet == 'DSL':
        factors.append({'label': 'DSL internet',              'value': 0.07, 'direction': 'negative'})

    if data.get('OnlineSecurity') == 'No':
        factors.append({'label': 'No online security',        'value': 0.14, 'direction': 'negative'})
    else:
        factors.append({'label': 'Has online security',       'value': 0.10, 'direction': 'positive'})

    if data.get('TechSupport') == 'No':
        factors.append({'label': 'No tech support',           'value': 0.12, 'direction': 'negative'})
    else:
        factors.append({'label': 'Has tech support',          'value': 0.09, 'direction': 'positive'})

    tenure = float(data.get('tenure', 12))
    t_eff  = min(0.30, tenure * 0.005)
    if t_eff > 0.10:
        factors.append({'label': f'Long tenure ({int(tenure)} mo)', 'value': t_eff,  'direction': 'positive'})
    else:
        factors.append({'label': f'Short tenure ({int(tenure)} mo)', 'value': 0.15, 'direction': 'negative'})

    monthly = float(data.get('MonthlyCharges', 65))
    if monthly > 80:
        factors.append({'label': 'High monthly charges',      'value': 0.12, 'direction': 'negative'})
    elif monthly < 35:
        factors.append({'label': 'Low monthly charges',       'value': 0.08, 'direction': 'positive'})

    if data.get('SeniorCitizen') in ('Yes', 1, '1'):
        factors.append({'label': 'Senior citizen customer',   'value': 0.08, 'direction': 'negative'})

    factors.sort(key=lambda x: x['value'], reverse=True)
    return factors[:5]


# ── Survival curve ────────────────────────────────────────────────────────────
def get_survival_curve(churn_prob: float) -> list:
    return [{'month': m,
             'retention': round(max(0.05, 1 - churn_prob * pow(m / 24, 0.6)) * 100, 1)}
            for m in [1, 3, 6, 12, 18, 24, 36]]


def get_churn_window(churn_prob: float) -> str:
    if churn_prob >= 0.80: return '15–30 days'
    if churn_prob >= 0.65: return '30–60 days'
    if churn_prob >= 0.50: return '60–90 days'
    if churn_prob >= 0.35: return '90–180 days'
    return 'Unlikely near-term'


# ── CLV in INR ────────────────────────────────────────────────────────────────
def get_clv(monthly_usd: float, churn_prob: float) -> dict:
    exp_months = max(1.0, (1 - churn_prob) * 48)
    clv_inr    = monthly_usd * exp_months * USD_TO_INR
    mth_inr    = monthly_usd * USD_TO_INR

    if   churn_prob >= 0.65 and clv_inr >= 200000: priority = 'Critical'
    elif churn_prob >= 0.65:                        priority = 'Medium'
    elif clv_inr >= 200000:                         priority = 'High'
    else:                                           priority = 'Low'

    return {
        'expected_clv_inr': round(clv_inr),
        'monthly_inr':      round(mth_inr),
        'expected_months':  round(exp_months, 1),
        'priority':         priority,
    }


# ── Recommendations ───────────────────────────────────────────────────────────
def generate_recommendations(data: dict, prediction: int) -> list:
    recs = []
    if prediction == 1:
        recs.append({'text': '🚨 High churn risk — immediate retention action required.', 'level': 'critical'})
    if data.get('Contract') == 'Month-to-month':
        recs.append({'text': '📝 Offer 1–2 year contract with 10% discount to improve retention.', 'level': 'high'})
    if data.get('PaymentMethod') == 'Electronic check':
        recs.append({'text': '💳 Move to automatic payment (bank transfer / credit card).', 'level': 'high'})
    if data.get('OnlineSecurity') in ('No', 'No internet service'):
        recs.append({'text': '🔒 Add Online Security — significantly increases customer stickiness.', 'level': 'medium'})
    if data.get('TechSupport') in ('No', 'No internet service'):
        recs.append({'text': '🛠️ Offer 3-month free Tech Support trial.', 'level': 'medium'})
    if float(data.get('tenure', 0)) < 12:
        recs.append({'text': '🎯 New customer — assign onboarding rep for first 3 months.', 'level': 'medium'})
    elif float(data.get('tenure', 0)) > 36:
        recs.append({'text': '⭐ Long-term customer — enroll in loyalty rewards program.', 'level': 'low'})
    if float(data.get('MonthlyCharges', 0)) > 80:
        recs.append({'text': '💰 High monthly charges — offer loyalty discount (10–15%).', 'level': 'medium'})
    if data.get('SeniorCitizen') in ('Yes', 1, '1'):
        recs.append({'text': '👴 Senior citizen — consider specialised support plans.', 'level': 'low'})
    if not recs:
        recs.append({'text': '✅ Customer appears stable — continue regular engagement.', 'level': 'low'})
    return recs


# ── Chart data ────────────────────────────────────────────────────────────────
def get_chart_data():
    if df_original is None:
        return {
            'tenure':  {'labels':['0–12','13–24','25–36','37–48','49–60','61–72'],
                        'churned':[320,245,180,112,87,54],'retained':[680,890,1020,1100,780,580]},
            'charges': {'labels':['<₹1,660','₹1,660-3,320','₹3,320-4,980','₹4,980-6,640','₹6,640-8,300','>₹8,300'],
                        'churn_rate':[8,12,22,32,41,48]},
            'payment': {'labels':['Electronic check','Mailed check','Bank transfer','Credit card'],
                        'churn_rate':[45,19,16,15]},
            'internet':{'labels':['Fiber optic (41%)','DSL (19%)','No internet (7%)'],
                        'churn_rate':[41,19,7]},
        }

    df_original['tenure_group'] = pd.cut(
        df_original['tenure'], bins=[0,12,24,36,48,60,72],
        labels=['0–12','13–24','25–36','37–48','49–60','61–72'])
    tg = df_original.groupby('tenure_group', observed=True)['Churn'] \
                    .value_counts().unstack(fill_value=0)

    df_original['charge_bin'] = pd.cut(
        df_original['MonthlyCharges'], bins=[0,20,40,60,80,100,200],
        labels=['<₹1,660','₹1,660-3,320','₹3,320-4,980','₹4,980-6,640','₹6,640-8,300','>₹8,300'])
    cb   = df_original.groupby('charge_bin',     observed=True)['Churn'].mean().mul(100).round(1)
    pm   = df_original.groupby('PaymentMethod',  observed=True)['Churn'].mean().mul(100).round(1)
    inet = df_original.groupby('InternetService',observed=True)['Churn'].mean().mul(100).round(1)

    return {
        'tenure': {'labels':[str(l) for l in tg.index.tolist()],
                   'churned': tg.get(1, pd.Series([0]*len(tg))).tolist(),
                   'retained':tg.get(0, pd.Series([0]*len(tg))).tolist()},
        'charges':{'labels':cb.index.tolist(),'churn_rate':cb.values.tolist()},
        'payment':{'labels':pm.index.tolist(),'churn_rate':pm.values.tolist()},
        'internet':{'labels':[f"{k} ({v:.0f}%)" for k,v in inet.items()],
                    'churn_rate':inet.values.tolist()},
    }


# ── Routes ────────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    if df_original is not None:
        stats = {
            'total':          len(df_original),
            'active':         int((df_original['Churn'] == 0).sum()),
            'churned':        int((df_original['Churn'] == 1).sum()),
            'avg_monthly_inr':round(df_original['MonthlyCharges'].mean() * USD_TO_INR),
            'avg_tenure':     round(df_original['tenure'].mean(), 1),
            'mtm_pct':        round((df_original['Contract']=='Month-to-month').mean()*100, 1),
            'echeck_pct':     round((df_original['PaymentMethod']=='Electronic check').mean()*100, 1),
        }
    else:
        stats = {'total':7043,'active':5174,'churned':1869,
                 'avg_monthly_inr':5375,'avg_tenure':32.4,'mtm_pct':55.0,'echeck_pct':34.0}
    return render_template('index.html', stats=stats, model_loaded=(model is not None))


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    result = None
    if request.method == 'POST':
        try:
            input_data = {}
            for feat in CATEGORICAL_FEATURES:
                input_data[feat] = request.form.get(feat, '')
            for feat in NUMERIC_FEATURES:
                input_data[feat] = float(request.form.get(feat, 0) or 0)

            if model is not None:
                enc           = preprocess_input(input_data)
                prediction    = int(model.predict(enc)[0])
                probabilities = model.predict_proba(enc)[0].tolist()
            else:
                prediction, probabilities = demo_predict(input_data)

            churn_prob = probabilities[1]
            result = {
                'risk_label':      'High' if prediction == 1 else 'Low',
                'risk_pct':        round(churn_prob * 100, 1),
                'confidence':      round(max(probabilities) * 100, 1),
                'is_high_risk':    prediction == 1,
                'shap_factors':    get_shap_factors(input_data, prediction),
                'survival':        get_survival_curve(churn_prob),
                'churn_window':    get_churn_window(churn_prob),
                'clv':             get_clv(input_data['MonthlyCharges'], churn_prob),
                'recommendations': generate_recommendations(input_data, prediction),
                'input_data':      input_data,
                'demo_mode':       model is None,
            }
        except Exception as e:
            import traceback
            result = {'error': str(e), 'detail': traceback.format_exc()}

    return render_template('predict.html', result=result,
                           categorical_features=CATEGORICAL_FEATURES,
                           numeric_features=NUMERIC_FEATURES)


@app.route('/insights')
def insights():
    chart_data = get_chart_data()
    if df_original is not None:
        summary = {
            'avg_tenure':     round(df_original['tenure'].mean(), 1),
            'avg_monthly_inr':round(df_original['MonthlyCharges'].mean() * USD_TO_INR),
            'mtm_pct':        round((df_original['Contract']=='Month-to-month').mean()*100, 1),
            'echeck_pct':     round((df_original['PaymentMethod']=='Electronic check').mean()*100, 1),
        }
    else:
        summary = {'avg_tenure':32.4,'avg_monthly_inr':5375,'mtm_pct':55.0,'echeck_pct':34.0}
    return render_template('insights.html', chart_data=json.dumps(chart_data), summary=summary)


@app.route('/about')
def about():
    return render_template('about.html', model_loaded=(model is not None))


@app.route('/api/predict', methods=['POST'])
def api_predict():
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({'error': 'POST JSON body required'}), 400
        if model is not None:
            enc        = preprocess_input(data)
            prediction = int(model.predict(enc)[0])
            probs      = model.predict_proba(enc)[0].tolist()
        else:
            prediction, probs = demo_predict(data)
        churn_prob = probs[1]
        clv = get_clv(float(data.get('MonthlyCharges', 65)), churn_prob)
        return jsonify({
            'prediction':        'High' if prediction == 1 else 'Low',
            'churn_probability': round(churn_prob * 100, 1),
            'confidence':        round(max(probs) * 100, 1),
            'churn_window':      get_churn_window(churn_prob),
            'expected_clv_inr':  clv['expected_clv_inr'],
            'priority':          clv['priority'],
            'demo_mode':         model is None,
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    print("\n" + "=" * 50)
    print("  Telecom Churn Prediction App")
    print("  http://127.0.0.1:5000")
    print("=" * 50 + "\n")
    app.run(debug=True, port=5000)