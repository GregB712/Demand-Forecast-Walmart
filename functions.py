import pandas as pd
import plotly.express as px
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
# from pandas.core.common import SettingWithCopyWarning
import warnings
import plotly.graph_objects as go
import numpy as np

from keras.layers import LSTM, Dense, Dropout
from keras.models import Sequential
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import EarlyStopping

from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error, mean_absolute_percentage_error


# warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore")

# Available templates for plots
# ['ggplot2', 'seaborn', 'simple_white', 'plotly',
#      'plotly_white', 'plotly_dark', 'presentation', 'xgridoff',
#      'ygridoff', 'gridon', 'none']

def evaluateDept(df):
    """
    It takes the dataframe, groups it by date, department, and store, sums the weekly sales, and then
    plots the results
    
    :param df: the dataframe to be evaluated
    """
    # Dept to avoid: 39, 77, 43
    # Validation of the result with excel.
    df = df[['Date', 'Dept', 'Store', 'Weekly_Sales']]
    aggregated_df = df.groupby(['Date','Dept','Store'], as_index=False).sum()
    aggregated_df = aggregated_df[['Date', 'Dept', 'Weekly_Sales']]
    aggregated_df = aggregated_df.groupby(['Date','Dept'], as_index=False).sum()
    aggregated_df['Date'] = pd.to_datetime(aggregated_df['Date'])
    aggregated_df = aggregated_df.sort_values(by=['Date','Dept'])

    fig = px.line(aggregated_df, x='Date', y='Weekly_Sales', color='Dept',title = 'Sales by Week and Department', 
              labels={'Date':'Week','Weekly_Sales':'Sales'}
              #,template='plotly_dark'
              )
    fig.show()

def combine_columns(df, columns):
    """
    It takes a dataframe and a list of columns, and returns a dataframe with a new column called
    'eventsInd' that is the logical OR of the columns in the list
    
    :param df: the dataframe
    :param columns: a list of column names
    :return: A dataframe with a new column called eventsInd.
    """    
    expr = ' | '.join(columns)
    for col in columns:
        df[col] = df[col].apply(lambda x: False if x == 0 else True)
    df['eventsInd'] = df.eval(expr)
    # df = df.drop(columns=columns)
    # Use the eval function to evaluate the expression
    return df

def createIndicators(df, listOfIndicators):
    """
    For each indicator in the list of indicators, create a new column in the dataframe that is 1 if the
    indicator is not 0, and 0 if the indicator is 0.
    
    :param df: the dataframe
    :param listOfIndicators: a list of the columns that you want to create indicators for
    :return: A dataframe with the new columns added.
    """
    i = 1
    for indicator in listOfIndicators:
            promoInd = 'promoInd' + str(i)
            df[promoInd] = df[indicator].apply(lambda x: 0 if x == 0 else 1)
            i += 1
    return df

def fixDates(timeSeries, aggregationMethods):
    """
    It takes a dataframe with a 'date' column and a 'sales' column, and returns a dataframe with a
    'week' column and a 'sales' column, where the 'week' column contains all the weeks between the
    minimum and maximum dates in the original dataframe, and the 'sales' column contains the sum of the
    sales for each week
    
    :param timeSeries: the dataframe containing the time series data
    :param aggregationMethods: a dictionary of aggregation methods to be applied to the dataframe
    :return: A dataframe with all dates between min and max date in original dataframe
    """
    # Monday as first day of the week
    timeSeries['week'] = timeSeries['date'] - timeSeries['date'].dt.weekday * np.timedelta64(1, 'D')
    
    timeSeries = timeSeries.groupby('week').agg(aggregationMethods).reset_index()

    # Create a new dataframe with all dates between min and max date in original dataframe
    date_range = pd.date_range(start=timeSeries['week'].min(), end=timeSeries['week'].max(), freq='W-MON')
    all_dates = pd.DataFrame({'week': date_range})
    all_dates = all_dates.set_index('week').assign(sales=0)

    # convert the 'week' column in the original dataframe to datetime format
    timeSeries["week"] = pd.to_datetime(timeSeries["week"])

    # Merge original dataframe with new dataframe on 'week' column
    timeSeries = pd.merge(all_dates, timeSeries, on='week', how='left').fillna(0)

    # Sort the dataframe by 'week' column in ascending order
    timeSeries = timeSeries.sort_index()

    # Rename and drop columns resulted from merging 
    timeSeries = timeSeries.rename(columns={'sales_y':'sales'}).drop(columns=['sales_x'])

    return timeSeries

def oosDetection(timeSeries):
    """
    If the sales value is 0, then the out-of-stock indicator is 1, otherwise it is 0
    
    :param timeSeries: the time series dataframe
    :return: A dataframe with a new column called 'oosInd' that is 1 if the sales value is 0 and 0
    otherwise.
    """
    timeSeries['oosInd'] = timeSeries['sales'].apply(lambda x: 1 if x == 0 else 0)
    return timeSeries

def plotTS(timeSeries, indList, hodilay, title='Sales Plot', y_axis_cols=['sales']):
    """
    It takes a time series dataframe, a list of indicators, a holiday indicator, a title, and a list of
    y-axis values (defaults to sales) and plots the time series with the indicators as bars
    
    :param timeSeries: the dataframe containing the time series data
    :param indList: a list of the indicators you want to plot
    :param hodilay: The name of the column that contains the holiday indicator
    :param title: The title of the plot
    :param y_axis_cols: the name of the column in the dataframe that contains the sales data
    """
    if len(indList) != 0:
        timeSeries['holInd'] = timeSeries[hodilay].apply(lambda x: 0 if x == 0 else 1)
        timeSeries = createIndicators(timeSeries, indList)
        timeSeries = timeSeries.sort_values(by=['week'])
        
        indList.append('oosInd')
        indList.append('holInd')

        fig = px.line(timeSeries, x='week', y=y_axis_cols, labels={'week':'Week','sales':'Sales'},
                    template='seaborn',
                    title = title,
                    # hover_data=indList
                    )

        fig.add_bar(x=timeSeries[timeSeries['holInd'] == 1]['week'], y=timeSeries[timeSeries['holInd'] == 1]['sales'], name='Holiday')
        fig.add_bar(x=timeSeries[timeSeries['promoInd1'] == 1]['week'], y=timeSeries[timeSeries['promoInd1'] == 1]['sales'], name='Promo1')
        fig.add_bar(x=timeSeries[timeSeries['promoInd2'] == 1]['week'], y=timeSeries[timeSeries['promoInd2'] == 1]['sales'], name='Promo2')
        fig.add_bar(x=timeSeries[timeSeries['promoInd3'] == 1]['week'], y=timeSeries[timeSeries['promoInd3'] == 1]['sales'], name='Promo3')
        fig.add_bar(x=timeSeries[timeSeries['promoInd4'] == 1]['week'], y=timeSeries[timeSeries['promoInd4'] == 1]['sales'], name='Promo4')
        fig.add_bar(x=timeSeries[timeSeries['promoInd5'] == 1]['week'], y=timeSeries[timeSeries['promoInd5'] == 1]['sales'], name='Promo5')
        fig.add_bar(x=timeSeries[timeSeries['oosInd'] == 1]['week'], y=timeSeries[timeSeries['oosInd'] == 0]['sales'], name='OOS')

        fig.show()
    else:
        timeSeries = timeSeries.sort_values(by=['week'])
        fig = px.line(timeSeries, x='week', y=y_axis_cols, labels={'week':'Week','sales':'Sales'},
                    template='seaborn')
        fig.show()

def plotTS_2(timeSeries, indList, indListNames, title='Sales Plot', y_axis_cols=['sales']):
    """
    This function takes a time series dataframe, a list of indicator columns, a list of indicator names,
    a title, and a list of y-axis columns, and plots a line graph of the time series dataframe with a
    bar graph for each indicator column
    
    :param timeSeries: the dataframe that contains the time series data
    :param indList: a list of indicator columns to plot
    :param indListNames: A list of names for the indicators
    :param title: The title of the plot, defaults to Sales Plot (optional)
    :param y_axis_cols: the column(s) you want to plot on the y-axis
    """
    timeSeries = timeSeries.sort_values(by=['week'])
    fig = px.line(timeSeries, x='week', y=y_axis_cols, labels={'week':'Week','sales':'Sales'},
                template='seaborn', title=title)
    for indicator, indicatorName in zip(indList, indListNames):
        fig.add_bar(x=timeSeries[timeSeries[indicator] == 1]['week'], y=timeSeries[timeSeries[indicator] == 1]['sales'], name=indicatorName)
    fig.show()

def find_outliers_IQR(df):
    """
    It takes a dataframe as input, and returns a dataframe of outliers
    
    :param df: the dataframe
    :return: A dataframe with the outliers
    """
    q1=df.quantile(0.25)
    q3=df.quantile(0.75)
    IQR=q3-q1
    outliers = df[((df<(q1-1.5*IQR)) | (df>(q3+1.5*IQR)))]
    return outliers

def plotCorrMatrix(timeSeries):
    """
    It takes a dataframe as input, calculates the correlation matrix, and plots it as a heatmap
    
    :param timeSeries: the dataframe containing the time series data
    """
    timeSeries_corr = timeSeries.corr()

    layout = go.Layout(
        autosize=False,
        width=500,
        height=500,

        xaxis= go.layout.XAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True),

        yaxis= go.layout.YAxis(linecolor = 'black',
                            linewidth = 1,
                            mirror = True),

        margin=go.layout.Margin(
            l=50,
            r=50,
            b=100,
            t=100,
            pad = 4
        )
    )

    fig = go.Figure(layout=layout)
    fig.add_trace(
        go.Heatmap(
            x = timeSeries_corr.columns,
            y = timeSeries_corr.index,
            z = np.array(timeSeries_corr),
            text=timeSeries_corr.values,
            texttemplate='%{text:.2f}'
        )
    )
    fig.show()

def outlierCorrections(timeSeries, col='sales', zero_condition = '', smoothing_level=0.35):
    """
    It takes a time series, and for each row, it fits a simple exponential smoothing model on the time
    series up to that row, and then forecasts the value for that row. 
    
    The function returns a list of corrected values
    
    :param timeSeries: The time series dataframe
    :param col: The column to be corrected, defaults to sales (optional)
    :param zero_condition: This is the column name of the column that has the zero values. If you don't
    have any zero values, you can leave this blank
    :param smoothing_level: This is the alpha parameter in the Holt-Winters equation. It is used to
    define the weight of data in the model
    :return: The corrected sales values
    """

    if col == 'sales':
        _timeSeries = timeSeries.copy()
        # Select rows where sales column is less than a certain percentage of the mean sales

        _timeSeries = combine_columns(_timeSeries, ['holInd', 'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'])

        # Select rows to be corrected
        # outliers = find_outliers_IQR(_timeSeries["sales"])
        q1=_timeSeries[col].quantile(0.25)
        q3=_timeSeries[col].quantile(0.75)
        IQR=q3-q1
        
        # rows_to_correct = _timeSeries.loc[(_timeSeries['sales'] < threshold*mean_sales) & (_timeSeries['eventsInd'] == False) & (_timeSeries['oosInd'] != 1)]
        rows_to_correct = _timeSeries.loc[((_timeSeries[col]<(q1-1.5*IQR)) | (_timeSeries[col]>(q3+1.5*IQR)) | (_timeSeries['oosInd']==1)) & (_timeSeries['eventsInd'] == False)]
        # print(rows_to_correct)

        # Create an empty list to store corrected sales values
        corrected_col = [0 for _ in range(rows_to_correct.shape[0])]

        # Iterate over selected rows
        j = 0
        for i, row in rows_to_correct.iterrows():
            # Fit the Simple Exponential Smoothing model on the time series
            train_data = _timeSeries[col].iloc[:i]

            if len(train_data) < 3:
                train_data[1] = train_data[0]
                train_data[2] = train_data[1]
            train_data = pd.Series(train_data)
            if train_data.empty:
                print('EMPTY TRAIN DATA')
                return []
            model = SimpleExpSmoothing(train_data).fit(smoothing_level=smoothing_level)
            # Forecast the sales value for the current row
            forecast = model.forecast(1)
            # Append the forecasted value to the list of corrected sales values
            corrected_col[j] = forecast.to_numpy()[0]
            j += 1

            # Create new dataframe with corrected values
            corrected_rows = rows_to_correct.copy()
            corrected_rows[col] = corrected_col

            # Update original dataframe
            _timeSeries.update(corrected_rows)
    else:
        _timeSeries = timeSeries.copy()

        # Select rows to be corrected
        # outliers = find_outliers_IQR(_timeSeries["sales"])
        q1=_timeSeries[col].quantile(0.25)
        q3=_timeSeries[col].quantile(0.75)
        IQR=q3-q1

        if zero_condition != '':
            rows_to_correct = _timeSeries.loc[((_timeSeries[col]<(q1-1.5*IQR)) | (_timeSeries[col]>(q3+1.5*IQR)) | (_timeSeries[zero_condition]==0))]
        else:
            rows_to_correct = _timeSeries.loc[((_timeSeries[col]<(q1-1.5*IQR)) | (_timeSeries[col]>(q3+1.5*IQR)))]
        
        # print(rows_to_correct.shape[0])

        # Create an empty list to store corrected sales values
        corrected_col = [0 for i in range(rows_to_correct.shape[0])]

        # Iterate over selected rows
        j = 0
        for i, row in rows_to_correct.iterrows():
            # Fit the Simple Exponential Smoothing model on the time series
            train_data = _timeSeries[col].iloc[:i]
            if len(train_data) < 3:
                train_data[1] = train_data[0]
                train_data[2] = train_data[1]
            train_data = pd.Series(train_data)
            if train_data.empty:
                print('EMPTY TRAIN DATA')
                return []
            model = SimpleExpSmoothing(train_data).fit(smoothing_level=smoothing_level)
            # Forecast the sales value for the current row
            forecast = model.forecast(1)
            # Append the forecasted value to the list of corrected sales values
            corrected_col[j] = forecast.to_numpy()[0]
            j += 1

            # Create new dataframe with corrected values
            corrected_rows = rows_to_correct.copy()
            corrected_rows[col] = corrected_col

            # Update original dataframe
            _timeSeries.update(corrected_rows)

    return _timeSeries[col]

def calendarFeatures(timeSeries):
    """
    It takes a time series dataframe and adds columns for week of year, month, and year
    
    :param timeSeries: The time series dataframe
    :return: The timeSeries dataframe with the new columns added.
    """
    timeSeries['week'] = pd.to_datetime(timeSeries['week'])
    timeSeries['woy'] = timeSeries['week'].dt.isocalendar().week
    timeSeries['month'] = timeSeries['week'].dt.month
    timeSeries['year'] = timeSeries['week'].dt.year

    return timeSeries

def outlierCorColWrapper(timeSeries, listCols, plot=True):
    """
    It takes a time series dataframe, a list of columns to correct, and a boolean to plot the results.
    It then loops through the list of columns, creates a new column with the corrected values, and plots
    the results if the boolean is set to True
    
    :param timeSeries: the dataframe you want to correct
    :param listCols: list of columns to be corrected
    :param plot: if True, will plot the original and corrected time series, defaults to True (optional)
    :return: The timeSeries dataframe with the new columns added.
    """
    for col in listCols:
        newColName = 'corrected_' + col
        timeSeries[newColName] = outlierCorrections(timeSeries, col, col)
        if plot:
            plotTS(timeSeries, [], hodilay='', title = col +' by Week', y_axis_cols = [col, newColName])
    return timeSeries

def averageSalesPerClass(timeSeries, column, method):
    """
    It takes a time series dataframe, a column name, and a method name, and returns a dataframe with the
    average sales per class of the column
    
    :param timeSeries: The dataframe that contains the time series data
    :param column: The column you want to aggregate by
    :param method: The aggregation method to be used
    :return: A dataframe with the average sales per class
    """
    timeSeries = timeSeries[['sales', column]]
    if method == 'mean':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).mean()
    elif method == 'median':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).median()
    elif method == 'std':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).std()
    elif method == 'var':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).var()
    elif method == 'mad':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).mad()
    elif method == 'max':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).max()
    elif method == 'min':
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).min()
    else:
        print('The Specified method does not exist, average will be used as the aggregation method')
        aggregated_timeSeries = timeSeries.groupby([column], as_index=False).mean()

    return aggregated_timeSeries

def combineAggregatedInfo(timeSeries, column, new_col_name, method):
    """
    It takes a time series dataframe, a column name, a new column name, and a method (mean, median, or
    sum) and returns a dataframe with the new column added
    
    :param timeSeries: the dataframe that you want to add the new column to
    :param column: the column you want to aggregate by
    :param new_col_name: the name of the new column that will be created
    :param method: 'mean' or 'median'
    :return: A dataframe with the new column added.
    """
    df = averageSalesPerClass(timeSeries, column, method)
    timeSeries = pd.merge(timeSeries, df, on=column, how='outer')

    timeSeries = timeSeries.rename(columns={'sales_x':'sales', 'sales_y':new_col_name})

    return timeSeries, df

def MovingAverage(timeSeries, method, rolling_window=4, alpha=0.1):
    """
    The function takes a time series, a method, a rolling window and an alpha as input and returns a
    time series with the moving average of the selected method
    
    :param timeSeries: The time series dataframe
    :param method: The type of moving average you want to use. The options are:
    :param rolling_window: The number of periods to include in the average, defaults to 4 (optional)
    :param alpha: The smoothing factor
    :return: the timeSeries dataframe with the added columns of the moving averages.
    """
    if method == 'simple':
        timeSeries['sma'] = timeSeries.sales.rolling(rolling_window, min_periods=1).mean()
    elif method == 'cumulative':
        timeSeries['cma'] = timeSeries.sales.expanding().mean()
    elif method == 'exponential':
        timeSeries['ema'] = timeSeries.sales.ewm(alpha=alpha, adjust=False).mean()
    else:
        print('No selected method all the available methods will be used!')
        timeSeries['sma'] = timeSeries.sales.rolling(rolling_window, min_periods=1).mean()
        timeSeries['cma'] = timeSeries.sales.expanding().mean()
        timeSeries['ema'] = timeSeries.sales.ewm(alpha=alpha, adjust=False).mean()
    return timeSeries

def timeShiftTs(timeSeries, windows):
    """
    > The function takes a time series and a list of windows and returns the time series with the
    windows shifted
    
    :param timeSeries: the time series dataframe
    :param windows: a list of integers, each integer represents the number of days to shift the time
    series
    :return: The time series with the shifted values.
    """
    for window in windows:
        name = 'shiftedSls_' + str(window)
        timeSeries[name].shift(window)
    return timeSeries

def TS_Stats(timeSeries):
    """
    It takes a time series and returns a time series with the following features: momentum, normalized
    sales, kurtosis, and skew
    
    :param timeSeries: the time series dataframe
    :return: the timeSeries dataframe.
    """
    timeSeries.loc[0, 'momentum'] = 1
    for i in range(1, len(timeSeries)):
        timeSeries.loc[i, 'momentum'] = timeSeries.loc[i, 'sales'] / timeSeries.loc[i-1, 'sales']

    maxSls = timeSeries['sales'].max()
    for i in range(0, len(timeSeries)):
        timeSeries.loc[i, 'norm'] = timeSeries.loc[i, 'sales'] / maxSls

    for i in range(0, len(timeSeries)):
        timeSeries.loc[i, 'kurtosis'] = timeSeries.loc[:i, 'sales'].kurtosis()
        timeSeries.loc[i, 'skew'] = timeSeries.loc[:i, 'sales'].skew()
    timeSeries['kurtosis'] = timeSeries['kurtosis'].bfill()
    timeSeries['skew'] = timeSeries['skew'].bfill()

    return timeSeries

def checkNanValues(df):
    missing_values = df.isna().sum()
    if missing_values == 0:
        print('No missing values found!')
    else:
        print('Missing Values:')
        print(df.columns[df.isnull().any()])
        print('-----------------')

def countNaNs(df):
    """
    It counts the number of missing values in each column of the dataframe and prints the columns which
    have missing values.
    
    :param df: The dataframe to be analyzed
    """
    # Count the number of missing values in each column
    missing_counts = df.isnull().sum()

    # Select the columns which have missing values
    missing_cols = missing_counts.loc[missing_counts != 0]

    # Print the missing count of each column
    print(missing_cols)


def preprocess(df_train, df_test, selected_features):
    """
    The function takes in the training and testing dataframes, and the selected features, and returns
    the training and testing data in the format required by the RNN.
    
    :param df_train: The training dataframe
    :param df_test: The test dataframe
    :param selected_features: The features we wish to use for the training
    :return: x_train_feat, x_test_feat, y_train, y_test, train_feats
    """
    df_train = df_train.sort_values(by=['week'])
    df_test = df_test.sort_values(by=['week'])

    # Select only the columns we wish to use for the training
    df_train = df_train[selected_features]

    df_test = df_test[selected_features]

    # Convert the 'week' column to datetime
    df_train['week'] = pd.to_datetime(df_train['week'])
    df_test['week'] = pd.to_datetime(df_test['week'])

    # Get the minimum date
    min_date_train = df_train['week'].min()
    min_date_test = df_test['week'].min()

    # Convert date to number of weeks
    df_train['week'] = (df_train['week'] - min_date_train).dt.days//7
    df_test['week'] = (df_test['week'] - min_date_test).dt.days//7

    # Create empty lists to store the train and test data
    x_train_feat = []
    y_train = []
    x_test_feat = []
    y_test = []

    # Loop through the dataframe and append the data to the lists
    for i in range(len(df_train)):
        x_train_feat.append(df_train.iloc[i, 1:-1])
        y_train.append(df_train.iloc[i, -1])

    for i in range(len(df_test)):
        x_test_feat.append(df_test.iloc[i, 1:-1])
        y_test.append(df_test.iloc[i, -1])

    train_feats = df_train.columns

    # Convert the lists to numpy arrays
    x_train_feat = np.array(x_train_feat)
    y_train = np.array(y_train)
    x_test_feat = np.array(x_test_feat)
    y_test = np.array(y_test)

    # Scale the data
    scaler = MinMaxScaler()
    x_train_feat = scaler.fit_transform(x_train_feat)
    x_test_feat = scaler.transform(x_test_feat)

    # Reshape the data for the RNN
    x_train_feat = x_train_feat.reshape((x_train_feat.shape[0], 1, x_train_feat.shape[1]))
    x_test_feat = x_test_feat.reshape((x_test_feat.shape[0], 1, x_test_feat.shape[1]))

    return x_train_feat, x_test_feat, y_train, y_test, train_feats


def create_model(x_train_feat_shape):
    """
    It creates a model with a single LSTM layer with 150 neurons, followed by a dense layer with 120
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 100
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 50
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 50
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 50
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 50
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer with 50
    neurons, followed by a dropout layer with a dropout rate of 0.2, followed by a dense layer
    
    :param x_train_feat_shape: (number of samples, number of time steps, number of features)
    :return: A model
    """
    # Create the model
    model = Sequential()
    model.add(LSTM(150, activation='relu', input_shape=(x_train_feat_shape[1], x_train_feat_shape[2])))
    model.add(Dense(120))
    model.add(Dropout(0.2))
    model.add(Dense(100))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(50))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')

    return model


def model_train(model, x_train_feat, y_train, x_test_feat):
    """
    It takes a model, training data, and test data as input, and returns the trained model and
    predictions
    
    :param model: the model to train
    :param x_train_feat: the training data
    :param y_train: The training data for the target variable
    :param x_test_feat: the test data
    :return: The model and the predictions
    """
    # fit the model
    # model.fit(x_train_feat, y_train, epochs=100, verbose=0)
    early_stop = EarlyStopping(monitor='loss', patience=2, verbose=1)
    history = model.fit(x_train_feat, y_train, epochs=100, batch_size=32, callbacks=[early_stop], verbose=0, shuffle=False)

    # make predictions
    y_pred = model.predict(x_test_feat)

    return model, y_pred

def kpis(y_test, y_pred):
    """
    It prints the MAE, MSE, R2_Score and MAPE of the predicted values and the actual values
    
    :param y_test: The actual values of the target variable
    :param y_pred: The predicted values
    """
    print("MAE: ", mean_absolute_error(y_test, y_pred))
    print("MSE: ", mean_squared_error(y_test, y_pred))
    print("R2_Score: ", r2_score(y_test, y_pred))
    print("MAPE: ", mean_absolute_percentage_error(y_test, y_pred))

