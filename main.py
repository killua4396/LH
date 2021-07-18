# 导入一些库
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import xgboost as xgb
from sklearn.decomposition import PCA


# 股票策略模版
# 初始化函数,全局只运行一次
def init(context):
    # 定义一些全局变量
    # g.scaler用于数据标准化
    g.scaler = StandardScaler()
    # g.model用于储存训练后的模型
    g.model = XGBClassifier()
    # g.dr用于储存股息率数据
    g.dr = {}
    # 用于存储下一个调仓日期
    context.next_date = None

    # 设置调仓周期
    context.cycle = 10

    # 设置基准收益：沪深300指数
    set_benchmark('000300.SH')
    # 打印日志
    log.info('回测开始')

    # 设置股票每笔交易的手续费为万分之二(手续费在买卖成交后扣除,不包括税费,税费在卖出成交后扣除)
    set_commission(PerShare(type='stock', cost=0.0002))

    # 设置股票交易滑点0.5%,表示买入价为实际价格乘1.005,卖出价为实际价格乘0.995
    set_slippage(PriceSlippage(0.003))

    # 设置日级最大成交比例25%,分钟级最大成交比例50%
    # 日频运行时，下单数量超过当天真实成交量25%,则全部不成交
    # 分钟频运行时，下单数量超过当前分钟真实成交量50%,则全部不成交
    set_volume_limit(0.25, 0.5)

    # 设置最大持仓数
    context.stock_hold_count = 7

    # 剔除新股与次新股
    get_old_stock()

    # 设置起始资金
    context.cash = 1000000

    log.info('初始参数设置成功')

    # 获取数据并进行处理，create_train_data方法见后
    data = create_train_data(start_date='20170101', end_date='20210101', stock_list=g.s_list)
    data = data.dropna()
    log.info('数据获取完成')

    # 选取出需要进行标准化的列
    d_temp = data[['factor_market_cap', 'factor_dividend_rate',
                   'factor_opt_profit_growth_ratio', 'factor_diluted_eps_growth_ratio',
                   'factor_weighted_roe', 'factor_opt_profit_div_income', 'factor_bbiboll']]
    # 进行标准化操作
    g.scaler.fit(d_temp)
    d_temp = g.scaler.transform(d_temp)
    # 将标准化后的数据重新装入DataFrame并重新填充列名
    d_temp = pd.DataFrame(d_temp)
    d_temp.columns = ['factor_market_cap', 'factor_dividend_rate',
                      'factor_opt_profit_growth_ratio', 'factor_diluted_eps_growth_ratio',
                      'factor_weighted_roe', 'factor_opt_profit_div_income', 'factor_bbiboll']

    # 将不需要标准化的列与标准化后的列拼接在一起
    data = data[["factor_date", "factor_symbol", "close", "close_10", "quote_rate(%)"]]
    data = pd.concat([data, d_temp], axis=1)
    log.info('数据标准化成功')

    # 定义函数进行市值中性化
    from sklearn.linear_model import LinearRegression
    def Neutralization_MC(df):
        x = df['factor_market_cap'].values.reshape(-1, 1)
        cols = df.columns
        for col in cols:
            if col != "factor_market_cap":
                y = df[col]
                lr = LinearRegression()
                lr.fit(x, y)
                y_predict = lr.predict(x)
                df[col] = y - y_predict
            else:
                continue
        return df

    # 为数据打上标签(十日涨幅大于6%则被认定为是好股票)
    data['is_good_stock'] = data['quote_rate(%)'].apply(lambda x: x > 3.75)

    def toInt(x):
        if x:
            return 1;
        else:
            return 0;

    data['is_good_stock'] = data['is_good_stock'].apply(toInt)
    log.info('股票Label标记成功')

    # 获取X,y
    # del用于删除一些模型训练不需要的列，例如日期，股票编码，只留下因子数据
    del data['factor_date']
    del data['factor_symbol']
    del data['close']
    del data['close_10']
    del data['quote_rate(%)']
    # label_y就是该股票是否为好股票
    y = data['is_good_stock']
    del data['is_good_stock']
    # X为因子数据
    X = data
    log.info('X,y分离成功')

    # 进行PCA降维
    g.pca = PCA(n_components=5)
    new_X = g.pca.fit_transform(X)
    log.info('PCA降维成功')

    # 分离训练集与测试集
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(new_X, y, test_size=0.1)

    log.info('开始进行模型训练')
    # 进行模型训练
    other_params = {'learning_rate': 0.01, 'n_estimators': 575, 'max_depth': 8, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.9, 'colsample_bytree': 0.8, 'gamma': 0.5, 'reg_alpha': 0.1, 'reg_lambda': 0.1}
    g.model = XGBClassifier(**other_params)
    g.model.fit(X_train, y_train)
    log.info('模型训练完成')

    # 由于股息率这个数据不是一直都有的，只有有时候分红的时候会有股息率
    # 为了获取完整的每日数据，需要填充股息率即，获取训练数据中股息率的均值并填充到每日数据之中
    log.info('开始获取股息率数据')
    # 获取股息率数据用于填充
    info_dividend_rate = pd.DataFrame()
    for stock in g.s_list:
        q = query(factor.symbol, factor.date, factor.dividend_rate).filter(
            factor.date > '20170101', factor.symbol == stock)
        info_temp = get_factors(q)
        info_dividend_rate = pd.concat([info_dividend_rate, info_temp])
    # 获取股息率均值
    info_dividend_rate = info_dividend_rate.dropna().groupby(["factor_symbol"]).agg('mean').reset_index()
    # 定义一个类型，用于存储每只股票对应的股息率均值
    for index, row in info_dividend_rate.iterrows():
        t = {row['factor_symbol']: row['factor_dividend_rate']}
        g.dr.update(t)
    log.info('股息率数据获取完成')

    log.info('开始获取当日数据')
    # 获取当日数据,get_nowday_info见后
    info = get_nowday_info()
    log.info('当日数据获取成功')

    # 存储一下股票编号，用于模型预测之后提取出股票代码
    stock = info['factor_symbol']
    stock = stock.tolist()

    # 股票编号不是因子数据，所以删除
    del info['factor_symbol']

    # 将今日数据进行标准化
    info = g.scaler.transform(info)
    # 将今日数据进行pca降维
    info = g.pca.transform(info)
    # 代入模型获得预测结果
    result = g.model.predict(info)
    # 得到最终预测出的股票编号
    r = []
    for i in range(len(result)):
        if result[i] == 1:
            r.append(stock[i])
    log.info('股票池获取成功')
    log.info(r)

    # 设置要操作的股票
    context.security = r

    # 调用方法获取下一个调仓日
    log.info("获取下个调仓日")
    current_date = pd.Timestamp(get_datetime()).normalize()
    # get_next_date见后
    context.next_date = get_next_date(current_date, context)
    # 打印下一个调仓日
    log.info("下个调仓日为:" + context.next_date.strftime("%Y-%m-%d"))
    log.info('init结束')


# 定义一个函数获取下一个调仓日
def get_next_date(current_date, context):
    # 获取今日日期
    current_date = pd.Timestamp(current_date).normalize()
    # 获取所有的交易日日期
    date_index = get_all_trade_days()
    # 在今日日期的基础上加上调仓周期获得下个调仓日
    trade_date = date_index[date_index.searchsorted(current_date, side='left') + context.cycle]
    # 返回下一个调仓日
    return trade_date


## 开盘时运行函数
def handle_bar(context, bar_dict):
    # 获取今日日期
    current_date = pd.Timestamp(get_datetime()).normalize()
    # 判断是否是调仓日，若该日是调仓日，则进行调仓
    if current_date == context.next_date:
        log.info('开始调整沪深三百')
        get_old_stock();
        log.info('调整成功')
        log.info("该日为调仓日，进行调仓")
        # 每次调仓之前上次买入失败的股票数据清空
        context.buy_failed_symbols = dict()
        context.sell_failed_symbols = dict()
        context.empty_failed_symbols = list()
        # 调用方法获取今日数据
        info = get_nowday_info()
        log.info('当日数据获取成功')

        # 存储一下股票编号，用于模型预测之后提取出股票代码
        stock = info['factor_symbol']
        stock = stock.tolist()

        # 股票编号不是因子数据，所以删除
        del info['factor_symbol']

        # 进行标准化处理
        info = g.scaler.transform(info)
        # 进行pca降维
        info = g.pca.transform(info)

        # 代入模型获得预测结果
        result = g.model.predict(info)

        # 得到最终预测出的股票编号
        r = []
        for i in range(len(result)):
            if result[i] == 1:
                r.append(stock[i])
        log.info('股票池获取成功')
        log.info(r)

        # 调仓
        context.security = r
        log.info('调仓结束')

        # 获取下一个调仓日并打印
        context.next_date = get_next_date(current_date, context)
        log.info("下个调仓日为:" + context.next_date.strftime("%Y-%m-%d"))



# 获取今日基本数据
def get_nowday_info():
    # 获取今日时间
    time = get_datetime().strftime("%Y%m%d")
    # 定义一个DataFrame用于存储今日基本数据
    info = pd.DataFrame()
    # 对于基础股票池中的每只股票都获取今日数据，最终拼接在一起
    for stock in g.s_list:
        q = query(factor.symbol, factor.market_cap, factor.dividend_rate, factor.opt_profit_growth_ratio,
                  factor.diluted_eps_growth_ratio, factor.weighted_roe, factor.opt_cost_div_income, factor.bbiboll
                  ).filter(factor.date == time, factor.symbol == stock)
        info_temp = get_factors(q)
        info = pd.concat([info, info_temp], axis=0)
    info = info.reset_index(drop=True)
    # 由于股息率数据不完整，所以填充今日股息率数据
    info["factor_dividend_rate"] = info['factor_symbol'].apply(lambda x: g.dr[x])
    # 去除空值，除了股息率之外仍会有少数股票的少数因子残缺，不做其他操作了，直接删除该行
    info = info.dropna()
    # 返回今日数据
    return info


# 创建一个函数用于获取数据，输入三个数据
# start_date起始日期，end_date结束日期，stock_list要获取的股票编号
def create_train_data(start_date, end_date, stock_list):
    log.info('开始获取数据')
    # 使用mindgo方法获取收盘数据
    # 此时data里面存储的是stock_list里面的股票从起始日期到结束日期每日的收盘价
    data = get_price(stock_list, start_date, end_date, '1d', ['close'], skip_paused=False,
                     is_panel=1).to_frame().reset_index()
    # 改变DataFrame的列名
    data.columns = ["date", "number", "close"]
    log.info('收盘数据获取成功')

    # 定义数据平移函数，就是将数据平移十日，为了获取该股票十日后的收盘价
    def shift(df):
        df["close_10"] = df["close"].apply(lambda x: x)
        df = df.set_index(["date", "number", 'close']).shift(-10).reset_index()
        df = df.dropna()
        return df

    # 定义一个Dataframe存储数据，后面的数据都存在这个new_data里面了
    new_data = pd.DataFrame()

    # 对于股票池里的每个股票获取收盘价并且平移10日，获取十日后收盘价
    for stock in stock_list:
        temp = shift(data[data["number"] == stock])
        new_data = pd.concat([new_data, temp])
    new_data = new_data.reset_index(drop=True)
    log.info('平移拼接成功')

    # 计算股票十日涨幅，存储到列quote_rate(%)中
    f = lambda x: (x['close_10'] - x['close']) / x['close'] * 100
    new_data["quote_rate(%)"] = new_data.apply(f, axis=1)
    new_data.columns = ["factor_date", "factor_symbol", "close", "close_10", "quote_rate(%)"]
    new_data = new_data.set_index(["factor_date", "factor_symbol"])

    # 定义函数用于均值填充
    from sklearn.impute import SimpleImputer
    def fillNA(df):
        item = df.columns[2]
        my_imputer = SimpleImputer()
        head = df[['factor_date', 'factor_symbol']]
        body = df[df.columns[2]]
        imputed_body = pd.DataFrame(my_imputer.fit_transform([body]))
        df = head.join(imputed_body.T)
        df.columns = ['factor_date', 'factor_symbol', item]
        return df

    # 定义函数用于删除极值
    def del_extremum(df):
        item = df.columns[2]
        a = df[item].mean() + df[item].std() * 3
        b = df[item].mean() - df[item].std() * 3
        c = df[item]
        c[c >= a] = np.nan
        c.fillna(a, inplace=True)
        c[c <= b] = np.nan
        c.fillna(b, inplace=True)
        df[item] = c
        return df

    # 定义函数用于添加因子数据
    def Add_factors(data, factor_item, stock_list):
        # 定义一个Dataframe暂时存储一下数据
        result = pd.DataFrame()
        # 对于股票池里的每只股票都做一遍这个操作
        for stock in stock_list:
            # 获取单个因子数据
            q = query(factor.date, factor.symbol, factor_item).filter(factor.symbol == stock, factor.date > '20170101',
                                                                      factor.date < '20210101')
            df = get_factors(q)
            df = df.reset_index(drop=True)
            df["factor_date"] = df['factor_date'].apply(lambda x: pd.to_datetime(x))
            # 进行均值填充
            df = fillNA(df)
            # 删除极值
            df = del_extremum(df)
            # 改变一下DataFrame的index，并将结果与原先的data拼接起来
            df = df.set_index(["factor_date", "factor_symbol"])
            result = pd.concat([result, df])
        data = data.join(result)
        # 返回数据
        return data

    # 要添加的因子的列表
    F_list = [factor.market_cap, factor.dividend_rate, factor.opt_profit_growth_ratio,
              factor.diluted_eps_growth_ratio, factor.weighted_roe, factor.opt_profit_div_income, factor.bbiboll]
    # 定义一个count用于后面输出添加因子进度(没什么实际作用)
    count = 0
    log.info('添加因子开始')
    # 对因子列表中的因子进行逐个添加操作
    for f in F_list:
        new_data = Add_factors(new_data, f, stock_list)
        count += 1
        log.info('获取数据进度：' + str(round(count / len(F_list) * 100)) + "%")
    new_data = new_data.reset_index()
    # 返回因子添加完毕后的数据
    return new_data

def get_old_stock():
    g.s_list = get_index_stocks("000300.SH")
    threshold_date = get_datetime().strftime('%Y-%m-%d %H:%M:%S')
    threshold_date = pd.to_datetime(threshold_date)
    threshold_date = threshold_date - pd.DateOffset(years=1)
    remove_list = []
    for i in range(len(g.s_list)):
        if threshold_date < pd.to_datetime(get_security_info(g.s_list[i]).start_date):
            remove_list.append(g.s_list[i])
    for stock in remove_list :
        g.s_list.remove(stock)
    
