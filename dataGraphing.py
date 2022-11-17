import datetime as dt
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

## Loading data
columnNames = ['repository', 'year', 'domain', 'criteria_1', 'criteria_2', 'criteria_3', 'criteria_4', 'paper_count']
dataset = pd.read_csv('Stats.csv', names=columnNames).reset_index()

## Preprocessing
dataset['year'] = pd.to_numeric(dataset.year, errors='coerce')
dataset['paper_count'] = pd.to_numeric(dataset.paper_count, errors='coerce')
dataset = dataset.loc[(dataset['year'].notnull()) &
            (dataset['paper_count'].notnull())]
dataset.criteria_2 = dataset.criteria_2.fillna('')

dataset = dataset.dropna(axis=1)
dataset['year'] = dataset['year'].astype(np.int32)
dataset['id'] = dataset['domain'] + dataset['criteria_2']
criterion = ['', 'fairness', 'privacy', 'ethic*']
dataset = dataset[dataset['criteria_2'].isin(criterion)]
dataset['log_paper_count'] = dataset['paper_count'].apply(lambda x: np.abs(np.log(x)) if x > 0 else 0.0000001)
dataset['paper_count'] = dataset['paper_count'].apply(lambda x: x if x > 0 else 0.0000001)

## Data Filtering

cs_exclusive_dataset = dataset.loc[(dataset.domain == 'cs ex.') & (dataset.criteria_2 == '')]
cross_domain_dataset = dataset.loc[(dataset.domain == 'cs') & (dataset.criteria_2 == '')]
cs_exclusive_keywords_dataset = dataset.loc[(dataset.domain == 'cs ex.') & (dataset.criteria_2.isin(criterion[1:]))]
cs_exclusive_ethic_dataset = dataset.loc[(dataset.domain == 'cs ex.') & (dataset.criteria_2 == 'ethic*')]
cs_exclusive_fairness_dataset = dataset.loc[(dataset.domain == 'cs ex.') & (dataset.criteria_2 == 'fairness')]
cs_exclusive_privacy_dataset = dataset.loc[(dataset.domain == 'cs ex.') & (dataset.criteria_2 == 'privacy')]


## Plot setup
from dash import Dash, html, dcc
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
import scipy.stats
from kapteyn import kmpfit


def confidence_interval(data, confidence=0.90):
    return scipy.stats.t.interval(alpha=confidence, df=len(data)-1, loc=np.mean(data), scale=scipy.stats.sem(data)) 

app = Dash(__name__)

# Scatterplot Data collected

fig = go.Figure(layout=dict(title=dict(text="Machine Learning Papers Published per Year Across Domains and with Different Keyword")))
fig.add_trace(go.Scatter(x=cs_exclusive_dataset['year'], y=cs_exclusive_dataset['paper_count'],
    name='Machine Learning (Computer Science)', mode="markers", fillcolor="rgba(111, 178, 236, 0.5)", line_color="rgb(90, 108, 236)"))
fig.add_trace(go.Scatter(x=cross_domain_dataset['year'], y=cross_domain_dataset['paper_count'],
    name='Machine Learning (Any Domain)', mode="markers", fillcolor="rgba(63, 205, 177,0.5)", line_color="rgb(63, 205, 177)"))
fig.add_trace(go.Scatter(x=cs_exclusive_ethic_dataset['year'], y=cs_exclusive_ethic_dataset['paper_count'],
    name='Keyword \"Ethic*\"', mode="markers", fillcolor="rgba(155, 201, 105, 0.5)", line_color="rgb(155, 201, 105)"))
fig.add_trace(go.Scatter(x=cs_exclusive_fairness_dataset['year'], y=cs_exclusive_fairness_dataset['paper_count'],
    name='Keyword \"fairness\"',mode="markers",  fillcolor="rgba(222, 100, 73, 0.5)", line_color="rgb(222, 100, 73)"))
fig.add_trace(go.Scatter(x=cs_exclusive_privacy_dataset['year'], y=cs_exclusive_privacy_dataset['paper_count'],
    name='Keyword \"privacy\"', mode="markers", fillcolor="rgba(176, 80, 206, 0.5)", line_color="rgb(176, 80, 206)"))

# Exponential trend modelling
def modelExpTrend(fig, x, y, trend_name, color):

  def exponenial_func(x, a, b, c):
      return a*np.exp(b*x)+c
  def model(p, x):
    a, b, c = p
    return a*np.exp(b*x)+c
  x_linear = np.array(list(range(x.shape[0])))

  y = y.values
  f = kmpfit.simplefit(model, [.1, 1e-6, .1], x_linear, y)
  # confidence band
  a, b, c = f.params
  dfdp = [np.exp(b*x_linear), a*x_linear*np.exp(b*x_linear), 1]
  y_hat, y_hat_CI_pos, y_hat_CI_neg = f.confidence_band(x_linear, dfdp, 0.90, model)


  fig.add_traces([
    go.Scatter(x=x, y=y_hat_CI_pos, showlegend=False, mode="lines", fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color="rgba(0, 0, 0, 0)"),
    go.Scatter(x=x, y=y_hat, fill='tonexty', name=trend_name, mode="lines",
      fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color=f"rgb({color[0]}, {color[1]}, {color[2]})"),
    go.Scatter(x=x, y=y_hat_CI_neg, fill='tonexty',showlegend=False, mode="lines", fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color="rgba(0,0,0,0)")])

# Linear trend modelling
def modelLinearTrend(fig, x, y, trend_name, color):
  x_linear = np.array(list(range(x.shape[0])))

  model = LinearRegression().fit(x_linear.reshape(-1,1),y)
  y_hat = model.predict(x_linear.reshape(-1,1))

  CI = confidence_interval(y_hat)
  CI_range = (CI[1]-CI[0])/2.0
  y_hat_CI_pos = [np.exp(y_hat_i+CI_range) for y_hat_i in y_hat]
  y_hat_CI_neg = [np.exp(y_hat_i-CI_range) for y_hat_i in y_hat]
  y_hat = [np.exp(y_hat_i) for y_hat_i in y_hat]

  fig.add_traces([
    go.Scatter(x=x, y=y_hat_CI_pos, showlegend=False, mode="lines", fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color="rgba(0, 0, 0, 0)"),
    go.Scatter(x=x, y=y_hat, fill='tonexty', name=trend_name, mode="lines",
      fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color=f"rgb({color[0]}, {color[1]}, {color[2]})"),
    go.Scatter(x=x, y=y_hat_CI_neg, fill='tonexty',showlegend=False, mode="lines", fillcolor=f"rgba({color[0]}, {color[1]}, {color[2]}, 0.2)", line_color="rgba(0,0,0,0)")])

# modelExpTrend(fig, cs_exclusive_dataset['year'], cs_exclusive_dataset['paper_count'], trend_name= 'Computer Science Trend',color=(111, 178, 236))
# modelExpTrend(fig, cross_domain_dataset['year'], cross_domain_dataset['paper_count'], trend_name= 'Any Domain Trend',color=(63, 145, 217))
# modelExpTrend(fig, cs_exclusive_ethic_dataset['year'][:-4], cs_exclusive_ethic_dataset['paper_count'][:-4], trend_name= 'Ethic Trend',color=(155, 201, 105))
# modelExpTrend(fig, cs_exclusive_fairness_dataset['year'][:-3], cs_exclusive_fairness_dataset['paper_count'][:-3], trend_name= 'Fairness Trend',color=(222, 100, 73))
# modelExpTrend(fig, cs_exclusive_privacy_dataset['year'], cs_exclusive_privacy_dataset['paper_count'], trend_name= 'Privacy Trend',color=(176, 80, 206))

modelLinearTrend(fig, cs_exclusive_dataset['year'], cs_exclusive_dataset['log_paper_count'], trend_name= 'Computer Science Trend',color=(90, 108, 236))
modelLinearTrend(fig, cross_domain_dataset['year'], cross_domain_dataset['log_paper_count'], trend_name= 'Any Domain Trend',color=(63, 205, 177))
modelLinearTrend(fig, cs_exclusive_ethic_dataset['year'][:-4], cs_exclusive_ethic_dataset['log_paper_count'][:-4], trend_name= 'Ethic Trend',color=(155, 201, 105))
modelLinearTrend(fig, cs_exclusive_fairness_dataset['year'][:-3], cs_exclusive_fairness_dataset['log_paper_count'][:-3], trend_name= 'Fairness Trend',color=(222, 100, 73))
modelLinearTrend(fig, cs_exclusive_privacy_dataset['year'], cs_exclusive_privacy_dataset['log_paper_count'], trend_name= 'Privacy Trend',color=(176, 80, 206))


fig.update_yaxes(type='log', range=[0,5], title="Paper published on arXiv")
fig.update_xaxes(dtick=1, title="Year of publication")

app.layout = html.Div(children=[
    html.H1(children='Hello Dash'),

    html.Div(children='''
        Dash: A web application framework for your data.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)