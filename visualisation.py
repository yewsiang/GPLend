
import matplotlib.pyplot as plt
import numpy as np

def test():
    #dummy data
    port_std = np.array([[20,30,15,10,20,30,50,50,30,20,15,20],
            [20,30,15,10,20,30,50,50,30,20,15,20],
            [20,30,15,10,20,30,50,50,30,20,15,20],
            [20,30,15,10,20,30,50,50,30,20,15,20]])

    port_avg = np.array([[300,310,320,330,340,350,360,370,380,390,300,310],
            [400,390,380,370,360,350,340,330,320,310,300,290],
            [500,510,520,530,540,550,560,570,580,590,600,610],
            [700,710,720,730,740,750,760,770,780,790,700,710]])

    time = [1,2,3,4,5,6,7,8,9,10,11,12]

    #test call
    plot_portfolio_performance(time, port_avg, port_std)

    #test call
    plot_portfolio_model_compare(time, port_avg, port_std)

def get_plot_data_from_performance(performance):
    """
    Generates a list of mean and std deviation for a performance.

    Outputs
    -------
    t: Time period
    mean: [D, T] array, where D is the types of data to show (eg profits, fund utilization)
        and T is the time period. Contains the mean of the data.
    std: [D, T] array, where D is the types of data to show (eg profits, fund utilization)
        and T is the time period. Contains the std deviation of the data.
    """
    _, num_months, D = performance.shape
    mean, std = np.zeros((D, num_months)), np.zeros((D, num_months))
    
    # Profits
    profits      = performance[:,:,0]
    mean_profits = np.mean(profits, axis=0)
    std_profits  = np.std(profits, axis=0)
    mean[2,:]    = mean_profits
    std[2,:]     = std_profits
    
    # Cumulative profits
    cum_profits      = np.cumsum(profits, axis=1)
    mean_cum_profits = np.mean(cum_profits, axis=0)
    std_cum_profits  = np.std(cum_profits, axis=0)
    mean[0,:]        = mean_cum_profits
    std[0,:]         = std_cum_profits

    # Funds remaining
    funds      = performance[:,:,1]
    mean_funds = np.mean(funds, axis=0)
    std_funds  = np.std(funds, axis=0)
    mean[1,:]  = mean_funds
    std[1,:]   = std_funds

    # Profit percentage
    profit_margin      = performance[:,:,4]
    mean_profit_margin = np.mean(profit_margin, axis=0)
    std_profit_margin  = np.std(profit_margin, axis=0)
    mean[3,:]          = mean_profit_margin
    std[3,:]           = std_profit_margin

    # Represent in thousands
    mean[0:2,:] = mean[0:2,:] / 1000
    std[0:2,:]  = std[0:2,:] / 1000
    # Represent in percentage
    mean[3,:] = mean[3,:] * 100
    std[3,:]  = std[3,:] * 100

    t = np.arange(num_months)

    return t, mean, std

def plot_portfolio_performance(performance):
    """
    Generate a plot of the portfolio performance over time given
    loan decisions made based on strategy chosen from varying models.

    Parameters
    ----------
    performance: performance statistics of the portfolio over time.
    """
    t, mean, std = get_plot_data_from_performance(performance)

    # Draw plots
    fig = plt.figure(figsize=(25,12))

    add_suplot_with_mean_and_std(fig, (2, 2, 1, "Cumulative profits", "Time (in months)", "Money (thousands)"),
                                 t, mean[0], std[0], line_col="red", fill_col="r")
    add_suplot_with_mean_and_std(fig, (2, 2, 2, "Funds remaining in the portfolio", "Time (in months)", "Money (thousands)"),
                                 t, mean[1], std[1], line_col="blue", fill_col="b")
    add_suplot_with_mean_and_std(fig, (2, 2, 3, "Profits during the month", "Time (in months)", "Money ($)"),
                                 t, mean[2], std[2], line_col="green", fill_col="g")
    add_suplot_with_mean_and_std(fig, (2, 2, 4, "Profit percentage", "Time (in months)", "Percentage (%)"),
                                 t, mean[3], std[3], line_col="black", fill_col="k")
    plt.tight_layout()
    plt.show()

def plot_portfolio_performance_comparisons(performances):
    _, num_months, D = performances[0].shape
    t = np.arange(num_months)
    fig = plt.figure(figsize=(25,12))
    means, stds = [], []

    for performance in performances:
        _, mean, std = get_plot_data_from_performance(performance)
        means.append(mean)
        stds.append(std)
    line_cols = ["red", "blue", "green", "black"]
    fill_cols = ["r", "b", "g", "k"]

    add_subplots_with_multiple_lines_mean_and_std(fig, 
        (2, 2, 1, "Cumulative profits", "Time (in months)", "Money (thousands)"), t, 
        [mean[0] for mean in means], [std[0] for std in stds], line_cols, fill_cols)
    add_subplots_with_multiple_lines_mean_and_std(fig,
        (2, 2, 2, "Funds remaining in the portfolio", "Time (in months)", "Money (thousands)"), t, 
        [mean[1] for mean in means], [std[1] for std in stds], line_cols, fill_cols)
    add_subplots_with_multiple_lines_mean_and_std(fig, 
        (2, 2, 3, "Profits during the month", "Time (in months)", "Money ($)"), t, 
        [mean[2] for mean in means], [std[2] for std in stds], line_cols, fill_cols)
    add_subplots_with_multiple_lines_mean_and_std(fig, 
        (2, 2, 4, "Profit percentage", "Time (in months)", "Percentage (%)"), t, 
        [mean[3] for mean in means], [std[3] for std in stds], line_cols, fill_cols)

    plt.show()

def add_suplot_with_mean_and_std(fig, subplot_info, t, mean, std, line_col='black', fill_col='k', ymin=None, ymax=None):
    col, row, plot_no, title, xlabel, ylabel = subplot_info
    sub = fig.add_subplot(col, row, plot_no)
    sub.set_title(title)
    sub.set_xlabel(xlabel)
    sub.set_ylabel(ylabel)
    sub.plot(t, mean, line_col)
    sub.fill_between(t, mean - std, mean + std, alpha=0.1, color=fill_col)
    if ymin and ymax:
        sub.set_ylim([ymin, ymax])
    
    # Commented out first
    """
    ymin = np.min(mean[0]-std[0])
    ymax = np.max(mean[0]+std[0])

    for i in range(1,4):
        model_min = np.min(mean[i]-std[i])
        model_max = np.max(mean[i]+std[i])
        if(model_min < ymin):
            ymin = model_min
        if(model_max > ymax):
            ymax = model_max

    #give some buffer space
    ymax = ymax + 100
    ymin = ymin - 100
    """

def add_subplots_with_multiple_lines_mean_and_std(fig, subplot_info, t, means, stds, line_cols, fill_cols):
    col, row, plot_no, title, xlabel, ylabel = subplot_info
    sub = fig.add_subplot(col, row, plot_no)

    for i in range(len(means)):
        mean = means[i]
        std  = stds[i]
        line_col = line_cols[i]
        fill_col = fill_cols[i]
        
        sub.set_title(title)
        sub.set_xlabel(xlabel)
        sub.set_ylabel(ylabel)
        sub.plot(t, mean, line_col)
        sub.fill_between(t, mean - std, mean + std, alpha=0.1, color=fill_col)

#plot on the same axis
def plot_portfolio_model_compare(t, mean, std):
    """
    Generate a plot comparing portfolio performances over time given
    loan decisions made based on strategy chosen from varying models.

    Parameters
    ----------
    time: int array
    port_min: int array
        lower bound
    port_max: int array
        upper bound
    port_avg: int array
        mean performance
    """
    fig = plt.figure(figsize=(25,12))

    plt.plot(t, mean[0],'red',label = "Model Name 1")
    plt.fill_between(t, mean[0]-std[0], mean[0]+ std[0],alpha=0.1, color = 'r')
    plt.plot(t, mean[1],'blue',label = "Model Name 2")
    plt.fill_between(t, mean[1]-std[1], mean[1]+ std[1],alpha=0.1, color = 'b')
    plt.plot(t, mean[2],'green',label = "Model Name 3")
    plt.fill_between(t, mean[2]-std[2], mean[2]+ std[2],alpha=0.1, color = 'g')
    plt.plot(t, mean[3],'black',label = "Model Name 4")
    plt.fill_between(t, mean[3]-std[3], mean[3]+ std[3],alpha=0.1, color = 'k')
    plt.xlabel("Time (in months)")
    plt.legend(loc = "upper right")
    plt.ylabel("$")
    plt.title("Comparison of Portfolio Performance")

    plt.show()
