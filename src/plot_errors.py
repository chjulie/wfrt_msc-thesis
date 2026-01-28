import marimo

__generated_with = "0.17.7"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt

    from utils.data_constants import ERROR_DATA_DIR
    return ERROR_DATA_DIR, pd, plt


@app.cell
def _(ERROR_DATA_DIR, pd):
    file_name = 'errors-reg_nwp-20230101_20230104.csv'
    error_df = pd.read_csv(f"{ERROR_DATA_DIR}/{file_name}").drop(columns=['Unnamed: 0'])
    return (error_df,)


@app.cell
def _(error_df):
    error_df
    return


@app.cell
def _(error_df):
    lead_times = error_df.lead_time.unique()
    lead_times
    return (lead_times,)


@app.cell
def _(error_df):
    temp = error_df[error_df.field == '2t'].groupby('lead_time')['rmse'].agg('mean')
    temp
    return


@app.cell
def _(error_df):
    precip_df = error_df[(error_df.field == 'tp') & (error_df.lead_time == 12)]
    precip_df
    precip_df.isnull().sum()
    #precip = error_df[error_df.field == 'tp'].groupby('lead_time')['rmse'].agg('mean')
    #precip
    return


@app.cell
def _(error_df):
    wind = error_df[error_df.field == '10ff'].groupby('lead_time')['rmse'].agg('mean')
    wind
    return


@app.cell
def _(error_df, lead_times, plt):
    fig, ax = plt.subplots(figsize=(6,4))

    ax.plot(lead_times, error_df[error_df.field == '2t'].groupby('lead_time')['rmse'].agg('mean'), '.-', color='red', label='2t')
    ax.plot(lead_times, error_df[error_df.field == 'tp'].groupby('lead_time')['rmse'].agg('mean'), '.-', color='blue', label='tp')
    ax.plot(lead_times, error_df[error_df.field == '10ff'].groupby('lead_time')['rmse'].agg('mean'), '.-',color='yellow', label='10ff')

    ax.legend(loc='upper right')
    ax.set_xlabel('Lead time [h]')
    ax.set_ylabel('RMSE')

    plt.show()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
