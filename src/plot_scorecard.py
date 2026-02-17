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
    return mo, np, pd, plt, rc


@app.cell
def _(plt, rc):
    rc("text", usetex=False)
    # rc("font", **{"family": "serif", "serif": ["Computer Modern Roman"], "size": "12"})
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
def _():
    ## Define analysis
    period = "test" # validation, test
    metric = "rmse" # "rmse", "mse"
    return metric, period


@app.cell
def _(pd):
    # Open csv files
    folder = "data/error_data"
    df_bris = pd.read_csv(f"{folder}/scorecard-bris-20220701_20221231.csv").drop(
        columns=["Unnamed: 0"]
    )
    fields = df_bris["field"].unique()
    lead_times = df_bris["lead_time"].unique() # TODO: wrong for test period
    df_bris
    return df_bris, fields, folder, lead_times


@app.cell
def _(df_bris):
    # group by lead time and field, take mean rmse
    gr_bris = df_bris.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")
    gr_bris
    return (gr_bris,)


@app.cell
def _(gr_bris):
    print(gr_bris["rmse"].loc[:,'10u'])
    return


@app.cell
def _(folder, pd):
    def load_validation_data():
        df_stage_c = pd.read_csv(f"{folder}/scorecard-stage-c-20220701_20221231.csv").drop(
            columns=["Unnamed: 0"]
        )
        df_stage_d2 = pd.read_csv(
            f"{folder}/scorecard-stage-d2-20220701_20221231.csv"
        ).drop(columns=["Unnamed: 0"])
        df_stage_d3 = pd.read_csv(
            f"{folder}/scorecard-stage-d3-20220701_20221231.csv"
        ).drop(columns=["Unnamed: 0"])
        df_stage_d4 = pd.read_csv(
            f"{folder}/scorecard-stage-d4-20220701_20221231.csv"
        ).drop(columns=["Unnamed: 0"])

        gr_stage_c = df_stage_c.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")
        gr_stage_d2 = df_stage_d2.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")
        gr_stage_d3 = df_stage_d3.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")
        gr_stage_d4 = df_stage_d4.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")

        return df_stage_c, df_stage_d2, df_stage_d3, df_stage_d4, gr_stage_c, gr_stage_d2, gr_stage_d3, gr_stage_d4
    return


@app.cell
def _(folder, pd):
    def load_test_data():
        df_nwp = pd.read_csv(f"{folder}/scorecard-nwp_reg-24h-20230101_20231231.csv").drop(
            columns=["Unnamed: 0"]
        )
        df_dl = pd.read_csv(f"{folder}/scorecard-stage-c-20230101_20231231.csv").drop(
            columns=["Unnamed: 0"]
        )

        gr_nwp = df_nwp.groupby(by=["lead_time", "field"])[["rmse"]].agg("mean")
        gr_dl = df_dl.groupby(by=["lead_time", "field"])[["rmse","mse"]].agg("mean")

        return df_nwp, df_dl, gr_nwp, gr_dl
    return


app._unparsable_cell(
    r"""
    if period=='validation':
        df_stage_c, df_stage_d2, df_stage_d3, df_stage_d4, gr_stage_c, gr_stage_d2, gr_stage_d3, gr_stage_d4 = load_validation_data()
        models = {
            #'bris' : {
            #    'data' : gr_bris,
            # },
            \"stage-c\": {
                \"data_grouped\": gr_stage_c,
                \"data\": df_stage_c,
                \"color\": \"cornflowerblue\",
            },
            \"stage-d2\": {
                \"data_grouped\": gr_stage_d2,
                \"data\": df_stage_d2,
                \"color\": \"orange\",
            },
            \"stage-d3\": {
                \"data_grouped\": gr_stage_d3,
                \"data\": df_stage_d3,
                \"color\": \"mediumseagreen\",
            },
            \"stage-d4\": {
                \"data_grouped\": gr_stage_d4,
                \"data\": df_stage_d4,
                \"color\": \"crimson\",
            },
        }
    elif period=='test':
        df_nwp, df_dl, gr_nwp, gr_dl = load_test_data()
        models = {
            \"nwp_reg\": {
                \"data_grouped\": gr_nwp,
                \"data\": df_nwp,
                \"color\": \"#A61166\",
            },
            \"dl_reg\": {
                \"data_grouped\": gr_dl[gr_dl.loc[]],
                \"data\": df_dl,
                \"color\": \"#F26419\"
            }
        }
    else:
        raise NotImplementedError
    """,
    name="_"
)


@app.cell
def _(models):
    print(models.keys())
    return


@app.cell
def _(fields, lead_times, metric, models, period, plt):
    for field in fields:
        fig, ax = plt.subplots(figsize=(10, 8))

        for model in models.keys():
            df = models[model]["data_grouped"]
            ax.plot(
                lead_times,
                df[metric].loc[[6,12,18,24], field],
                ".-",
                color=models[model]["color"],
                label=model,
            )

        ax.legend(loc="upper left")
        ax.grid()
        ax.set_xticks(lead_times)
        ax.set_ylabel(metric)
        ax.set_xlabel("Lead time")
        ax.set_title(f"Model stages comparison for field {field}")
        plt.savefig(
            f"reports/plots/scorecard/{metric}_{period}_{field}.png",
            bbox_inches="tight",
            dpi=100,
        )
        plt.close()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Explore variability of prediction
    Same as before, but add thin line to get an idea of the spread of distribution
    """)
    return


@app.cell
def _(fields, lead_times, metric, models, plt):
    def _():
        for field in fields:
            for model in models.keys():
                fig, ax = plt.subplots(figsize=(10, 8))
                print(model)
                df_gr = models[model]["data_grouped"]
                df = models[model]["data"]
                ax.plot(
                    lead_times,
                    df_gr[metric].loc[:, field],
                    ".-",
                    label=model,
                    color=models[model]["color"],
                )
                for initial_date in df.initial_date.unique():
                    try:
                        ax.plot(
                            lead_times,
                            df[
                                (df.initial_date == str(initial_date))
                                & (df["field"] == field)
                            ]["rmse"],
                            linewidth=0.5,
                            alpha=0.5,
                            color=models[model]["color"],
                        )
                    except ValueError as e:
                        continue

                ax.legend(loc="upper left")
                ax.grid()
                ax.set_xticks(lead_times)
                ax.set_ylabel("RMSE")
                ax.set_xlabel("Lead time")
                ax.set_title(f"RMSE vs Lead time for model {model} and field {field}")
                plt.savefig(
                    f"reports/plots/scorecard/rmse_val_{model}_{field}.png",
                    bbox_inches="tight",
                    dpi=100,
                )
        return

    _()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Process scorecard data
    Process dataframes to get raw data to plot in excel sheet
    """)
    return


@app.cell
def _(np):
    scorecard_fields = [
        'z_50',
        'z_100',
        'z_250',
        'z_500',
        'z_850',
        't_50',
        't_100',
        't_250',
        't_500',
        't_850',
        'u_50',
        'u_100',
        'u_250',
        'u_500',
        'u_850',
        'v_50',
        'v_100',
        'v_250',
        'v_500',
        'v_850',
        '2t',
        '10u',
        '10v',
        'tp', 
    ]
    scorecard_lt = np.arange(6,25,6)
    return scorecard_fields, scorecard_lt


@app.cell
def _(scorecard_lt):
    print(scorecard_lt)
    return


@app.cell
def _(gr_dl, gr_nwp, metric, np, scorecard_fields, scorecard_lt):
    processed_scorecard = np.zeros((len(scorecard_fields), len(scorecard_lt)))

    for i,f in enumerate(scorecard_fields):
        for j, lt in enumerate(scorecard_lt):
            nwp_val = gr_nwp[metric].loc[lt, f]
            dl_val = gr_dl[metric].loc[lt, f]
            norm_diff = (nwp_val - dl_val) / nwp_val
            processed_scorecard[i,j] = norm_diff

    np.savetxt('reports/data/processed_scorecard.csv',processed_scorecard, delimiter=',', fmt='%4f')
    return (processed_scorecard,)


@app.cell
def _(processed_scorecard):
    print(processed_scorecard)
    return


@app.cell
def _(plt, processed_scorecard):
    plt.pcolormesh(processed_scorecard)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
