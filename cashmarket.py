class CashMarket(bt.analyzers.Analyzer):
    """
    Analyzer returning cash and market values
    """

    def start(self):
        super(CashMarket, self).start()

    def create_analysis(self):
        self.rets = {}
        self.vals = 0.0

    def notify_cashvalue(self, cash, value):
        self.vals = (cash, value)
        self.rets[self.strategy.datetime.datetime()] = self.vals

    def get_analysis(self):
        return self.rets
    
import quanstats as qs
def tearsheet(scene, results):
   
    # Get the stats auto ordered nested dictionary
    value = results[0].analyzers.getbyname("cash_market").get_analysis()

    columns = [
        "Date",
        "Cash",
        "Value",
    ]

    if scene["save_tearsheet"]:
        # Save tearsheet
        df = pd.DataFrame(value)
        df = df.T
        df = df.reset_index()
        df.columns = columns

        df_value = df.set_index("Date")["Value"]
        df_value.index = pd.to_datetime(df_value.index)
        df_value = df_value.sort_index()
        value_returns = qs.utils.to_returns(df_value)
        value_returns = pd.DataFrame(value_returns)

        value_returns["diff"] = value_returns["Value"].diff().dropna()
        value_returns["diff"] = value_returns["diff"].abs().cumsum()
        value_returns = value_returns.loc[value_returns["diff"] > 0, "Value"]
        value_returns.index = pd.to_datetime(value_returns.index.date)

        # Get the benchmark
        benchmark = None
        bm_title = None
        bm = scene["benchmark"]
        if bm:
            df_benchmark = yf.download(
                bm,
                start=value_returns.index[0],
                end=value_returns.index[-1],
                auto_adjust=True,
            )["Close"]

            df_benchmark = qs.utils.rebase(df_benchmark)
            benchmark = qs.utils.to_returns(df_benchmark)
            benchmark.name = bm
            benchmark.index = pd.to_datetime(benchmark.index.date)
            bm_title = f"  (benchmark: {bm})"



        # Set up file path.
        Path(scene["save_path"]).mkdir(parents=True, exist_ok=True)
        dir = Path(scene["save_path"])
        filename = (
                scene["save_name"]
                + "-"
                + scene["batchname"]
                + "-"
                + scene["batch_runtime"].replace("-", "").replace(":", "").replace(" ", "_")
                + ".html"
        )
        filepath = dir / filename

        title = f"{scene['batchname']}{bm_title if bm_title is not None else ''}"
        qs.reports.html(
            value_returns,
            benchmark=benchmark,
            title=title,
            output=filepath,
        )