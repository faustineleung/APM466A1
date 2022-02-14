import pandas as pd
from scipy import optimize
import matplotlib.pyplot as plt
import math
import numpy as np

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Get present value of future cash flows
def get_pv(cpn,fv,days_to_maturity,x):
    t = (days_to_maturity/365)*2

    # Calculate face value of bond
    pv = fv/((1+x/2)**t)

    # Calculate coupon pyaments of bond backtracking coupon periods from maturity
    while t > 0:
        pv += cpn/((1+x/2)**t)
        t -= 1

    return pv

# Get the number of days that have passed since last coupon payment
def get_cpn_days(d):
    while d > 0:
        d -= 365/2
    return abs(d)

# Calculate dirty price by adding clean price and accrued interest
def get_dirty_price(days_to_maturity,cpn,clean_price):
    days_per_period = 365/2
    accrued_int = cpn * (get_cpn_days(days_to_maturity) / days_per_period)

    return clean_price + accrued_int

# Plot yield curve and get yields
def get_ytm(df):
    # Initialize estimates to be used for Newton's method to find yield
    x0_dict = {0.5: 0.0041, 1: 0.0082, 2: 0.0129, 3: 0.0139, 5: 0.0164}

    df_ytm = pd.DataFrame()
    df_ytm['Maturity'] = df['Maturity Date']

    # Iterate through days of data
    for col in df.columns:
        ytm_col = []
        if '2022' in col:
            col_date = pd.to_datetime(col)

            # Iterate through each bond
            for index, row in df.iterrows():
                cpn = float(row['Coupon'].strip('%')) / 2
                fv = 100
                days_to_maturity = (row['Maturity Date'] - col_date).days
                clean_price = row[col]
                # Get expression for yield
                get_ytm = lambda x: get_pv(cpn, fv, days_to_maturity,x) - get_dirty_price(days_to_maturity,cpn,clean_price)

                # Find estimate to use for Newton's method using closest years to maturity in dictionary
                x0_diff = lambda list_val: abs(list_val - days_to_maturity/365)
                x0_key = min(list(x0_dict.keys()), key=x0_diff)
                x0 = x0_dict[x0_key]

                # Find yield using Newton's method
                ytm_col.append(optimize.newton(get_ytm, x0) * 100)
            df_ytm[col] = ytm_col

    # Plot yield curve
    df_ytm = df_ytm.set_index('Maturity')
    df_ytm.plot(title="5-year CAN Yield Curve")
    plt.xlabel("Maturity Date")
    plt.ylabel("Yield (%)")
    plt.xticks(rotation=70)
    plt.savefig('yield.pdf')

    return df_ytm

# Plot spot curve and get spot rates
def get_spot(df):
    spot_dataframes = []
    col_names = []

    # Iterate through days of data
    for col in df.columns:
        if '2022' in col:
            col_names.append(col)

            # Initialize dataframe to contain spot rates for each day
            df_spot = pd.DataFrame()
            spot_col = []
            time_col = []
            col_date = pd.to_datetime(col)

            # Iterate through each bond
            for index, row in df.iterrows():
                days_to_maturity = (row['Maturity Date'] - col_date).days
                years_to_maturity = days_to_maturity/365
                clean_price = row[col]
                cpn = float(row['Coupon'].strip('%')) / 2
                fv = 100

                # If bond matures in less than 6 months
                if years_to_maturity < 0.5:
                    # Calculate spot rate, treating like zero-coupon bond
                    spot = (-math.log(get_dirty_price(days_to_maturity,cpn,clean_price)/(fv+cpn)))/years_to_maturity
                # If bond matures in more than 6 months
                else:
                    # Sum the present value of all coupon payments before bond matures
                    sum = 0
                    for t in range(len(spot_col)):
                        sum += cpn*math.exp(-(spot_col[t]/100)*time_col[t])

                    # Calculate spot rate
                    spot = (-math.log((get_dirty_price(days_to_maturity,cpn,clean_price)-sum)/(fv+cpn)))/years_to_maturity

                time_col.append(years_to_maturity)
                spot_col.append(spot*100)

            # Add time and spot rates to dataframe
            df_spot['Time'] = time_col
            df_spot['Spot'] = spot_col
            spot_dataframes.append(df_spot)

    # Plot first day of data
    ax = spot_dataframes[0].plot(x='Time', y='Spot', label=col_names[0], title="5-year CAN Spot Curve")
    plt.ylabel("Spot Rate (%)")

    # Plot spots for subsequent days on same graph
    for i in range(1,len(spot_dataframes)):
        spot_dataframes[i].plot(x='Time', y='Spot', ax=ax, label=col_names[i])

    # Save plot
    plt.xlim(left=1)
    plt.savefig('spot.pdf')

    return spot_dataframes,col_names

# Plot 1yr forward curve and get forward rates
def get_forward(spot_dataframes,col_names):
    forward_dataframes = []

    # Iterate through list of dataframes containing spot rates for each day
    for i in range(len(spot_dataframes)):
        # Initialize empty dataframe to contain forward rates for each day
        df_forward = pd.DataFrame()
        time_col = []
        f_col = []

        df = spot_dataframes[i]
        time = list(df['Time'])
        spot = list(df['Spot'])

        # Calculate 1yr forward rates for 1yr to 4yr
        r_1 = np.interp(1,time,spot)
        for i in range(2,6):
            r_i = np.interp(i,time,spot)
            time_col.append(i-1)
            f_col.append((r_i*i-r_1)/(i-1))

        # Add time and forward rates to list of dataframes
        df_forward['Time'] = time_col
        df_forward['Forward'] = f_col
        forward_dataframes.append(df_forward)

    # Plot first day of data
    ax = forward_dataframes[0].plot(x='Time', y='Forward', label=col_names[0], title="1-year CAN Forward Curve")
    plt.ylabel("Forward Rate (%)")

    # Plot forward rates for subsequent days on same graph
    for i in range(1, len(forward_dataframes)):
        forward_dataframes[i].plot(x='Time', y='Forward', ax=ax, label=col_names[i])

    # Save plot
    plt.xlim(left=1)
    plt.savefig('forward.pdf')

    return forward_dataframes

# Find covariance matrix for yield
def get_cov_ytm(df_ytm):
    # Add yield for each day to list
    mat_ytm = []
    for col in df_ytm.columns:
        col_date = pd.to_datetime(col)
        r_col = []
        for i in range(1,6):
            maturity = (col_date + pd.offsets.DateOffset(years=i)).timestamp()
            x = [j.timestamp() for j in list(df_ytm.index)]
            y = list(df_ytm[col])
            r_maturity = np.interp(maturity,x,y)
            r_col.append(r_maturity)
        mat_ytm.append(r_col)

    # Create matrix of log-returns of yield
    mat_ytm = np.array(mat_ytm).T
    mat_ytm = np.log(mat_ytm)
    mat_ytm = np.diff(mat_ytm)

    # Find covariance matrix
    mat_ytm = mat_ytm.T
    mean_ytm = np.mean(mat_ytm,axis=0)
    n_ytm = mat_ytm.shape[0]
    cov_ytm = np.dot((mat_ytm-mean_ytm).T, (mat_ytm-mean_ytm))/(n_ytm-1)

    return cov_ytm

# Find covariance matrix for forward rate
def get_cov_forward(forward_dataframes):
    # Add each day of forward rates to list
    mat_forward = []
    for i in range(len(forward_dataframes)):
        df = forward_dataframes[i]
        mat_forward.append(list(df['Forward']))

    # Create matrix of log-return of forward rates
    mat_forward = np.array(mat_forward).T
    mat_forward = np.log(mat_forward)
    mat_forward = np.diff(mat_forward)

    # Find covariance matrix
    mat_forward = mat_forward.T
    mean_forward = np.mean(mat_forward, axis=0)
    n_forward = mat_forward.shape[0]
    cov_forward = np.dot((mat_forward - mean_forward).T, (mat_forward - mean_forward)) / (n_forward - 1)

    return cov_forward

# Find eigenvectors/values
def get_eigen(cov_ytm,cov_forward):
    # Eigenvectors/values for yield covariance matrix
    eval_ytm, evect_ytm = np.linalg.eig(cov_ytm)

    # Eigenvectors/values for forward rate covariance matrix
    eval_forward, evect_forward = np.linalg.eig(cov_forward)

    return eval_ytm,evect_ytm,eval_forward,evect_forward


if __name__ == "__main__":
    # Import 11 selected bonds
    df = pd.read_csv('A1Data.csv', header='infer')

    # Change format of date columns
    df['Maturity Date'] = pd.to_datetime(df['Maturity Date'])
    df['Issue Date'] = pd.to_datetime(df['Issue Date'])

    # 4a
    df_ytm = get_ytm(df)

    # 4b
    spot_dataframes,col_names = get_spot(df)

    # 4c
    forward_dataframes = get_forward(spot_dataframes,col_names)

    # 5
    cov_ytm = get_cov_ytm(df_ytm)
    cov_forward = get_cov_forward(forward_dataframes)

    # 6
    eval_ytm,evect_ytm,eval_forward,evect_forward = get_eigen(cov_ytm, cov_forward)

