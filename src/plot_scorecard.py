import marimo

__generated_with = "0.18.0"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib import rc
    import numpy as np
    import pandas as pd
    from datetime import datetime
    return mo, pd, plt, rc


@app.cell
def _(plt, rc):
    rc("text", usetex=False)
    #rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": "12"})
    rc("lines", linewidth=2)
    plt.rcParams["axes.facecolor"] = "w"
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Objectives
    Compare model stages and validate more training leads to lower RMSE (in comparison to training objective) averaged over the domain and over the validation period per lead time. Sanity check for model quality, for added-value from rollout training and to check for divergence

    # Method
    - For each variable, take the spatial mean over the domain and temporal mean across the validation period
    - Plot mean RMSE vs lead time for each model

    # Hypothesis
    - RMSE increase with lead time
    - stage-d4 < stage-d3 < stage-d2 < stage-c << bris for all variables and all lead time
    """)
    return


@app.cell
def _(pd):
    # Open csv files
    folder = "data/error_data"
    df_bris = pd.read_csv(f"{folder}/scorecard-bris-20220701_20221231.csv").drop(columns=['Unnamed: 0'])
    df_bris
    return df_bris, folder


@app.cell
def _(df_bris):
    # group by lead time and field, take mean rmse
    gr_bris = df_bris.groupby(by=['lead_time', 'field'])['rmse'].agg('mean')
    gr_bris
    return (gr_bris,)


@app.cell
def _(folder, pd):
    df_stage_c = pd.read_csv(f"{folder}/scorecard-stage-c-20220701_20221231.csv").drop(columns=['Unnamed: 0'])
    df_stage_d2 = pd.read_csv(f"{folder}/scorecard-stage-d2-20220701_20221231.csv").drop(columns=['Unnamed: 0'])
    df_stage_d3 = pd.read_csv(f"{folder}/scorecard-stage-d3-20220701_20221231.csv").drop(columns=['Unnamed: 0'])
    df_stage_d4 = pd.read_csv(f"{folder}/scorecard-stage-d4-20220701_20221231.csv").drop(columns=['Unnamed: 0'])
    return df_stage_c, df_stage_d2, df_stage_d3, df_stage_d4


@app.cell
def _(df_stage_c, df_stage_d2, df_stage_d3, df_stage_d4):
    gr_stage_c = df_stage_c.groupby(by=['lead_time', 'field'])['rmse'].agg('mean')
    gr_stage_d2 = df_stage_d2.groupby(by=['lead_time', 'field'])['rmse'].agg('mean')
    gr_stage_d3 = df_stage_d3.groupby(by=['lead_time', 'field'])['rmse'].agg('mean')
    gr_stage_d4 = df_stage_d4.groupby(by=['lead_time', 'field'])['rmse'].agg('mean')
    return gr_stage_c, gr_stage_d2, gr_stage_d3, gr_stage_d4


@app.cell
def _(df_bris):
    lead_times = df_bris['lead_time'].unique()
    lead_times
    return (lead_times,)


@app.cell
def _(df_bris):
    fields = df_bris['field'].unique()
    fields
    return (fields,)


@app.cell
def _(gr_bris):
    gr_bris.index
    return


@app.cell
def _(gr_bris):
    gr_bris.loc[:,'v_50']
    return


@app.cell
def _(
    df_stage_c,
    df_stage_d2,
    df_stage_d3,
    df_stage_d4,
    gr_stage_c,
    gr_stage_d2,
    gr_stage_d3,
    gr_stage_d4,
):
    models = {
        #'bris' : {
        #    'data' : gr_bris,
        #},
        'stage-c' : {
            'data_grouped' : gr_stage_c,
            'data' : df_stage_c,
            'color': 'cornflowerblue',
        },
        'stage-d2' : {
            'data_grouped' : gr_stage_d2,
            'data' : df_stage_d2,
            'color': 'orange',
        
        },
        'stage-d3' : {
            'data_grouped' : gr_stage_d3,
            'data' : df_stage_d3,
            'color': 'mediumseagreen',
        },
        'stage-d4' : {
            'data_grouped' : gr_stage_d4,
            'data' : df_stage_d4,
            'color': 'crimson',
        },
    }
    return (models,)


@app.cell
def _(fields, lead_times, models, plt):
    for field in fields:
        fig, ax = plt.subplots(figsize=(10,8))
    
        for model in models.keys():
            df = models[model]['data_grouped']
            ax.plot(lead_times, df.loc[:,field], '.-', color=models[model]['color'], label=model)
    
        ax.legend(loc='upper left')
        ax.grid()
        ax.set_xticks(lead_times)
        ax.set_ylabel('RMSE')
        ax.set_xlabel('Lead time')
        ax.set_title(f"Model stages comparison for field {field}")
        plt.savefig(f"reports/plots/scorecard/rmse_val_{field}.png", bbox_inches='tight', dpi=100)

    return (field,)


@app.cell
def _(mo):
    mo.md(r"""
    # Explore variability of prediction
    Same as before, but add thin line to get an idea of the spread of distribution
    """)
    return


@app.cell
def _(df_bris, df_stage_c, field):
    d = df_bris.initial_date.unique()[0]
    print(d)
    print(df_stage_c[(df_stage_c.initial_date == '2022-07-01T00:00:00') & (df_stage_c['field'] == field)])

    return


@app.cell
def _(fields, lead_times, models, plt):
    def _():
        for field in fields:
            for model in models.keys():
                fig, ax = plt.subplots(figsize=(10,8))
                print(model)
                df_gr = models[model]['data_grouped']
                df = models[model]['data'] 
                ax.plot(lead_times, df_gr.loc[:,field], '.-', label=model, color=models[model]['color'])
                for initial_date in df.initial_date.unique():
                    try: 
                        ax.plot(lead_times, df[(df.initial_date == str(initial_date)) & (df['field'] == field)]['rmse'], linewidth=0.5, alpha=0.5, color=models[model]['color'])
                    except ValueError as e:
                        continue
    
                ax.legend(loc='upper left')
                ax.grid()
                ax.set_xticks(lead_times)
                ax.set_ylabel('RMSE')
                ax.set_xlabel('Lead time')
                ax.set_title(f"RMSE vs Lead time for model {model} and field {field}")
                plt.savefig(f"reports/plots/scorecard/rmse_val_{model}_{field}.png", bbox_inches='tight', dpi=100)
        return 


    _()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
