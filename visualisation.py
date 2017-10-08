
import matplotlib.pyplot as plt
import numpy as np

def test():
    #dummy data
    port_min = np.array([[100,110,120,130,140,150,160,170,180,190,200,210],
                [200,190,180,170,160,150,140,130,120,110,100,90],
                [300,310,320,330,340,350,360,370,380,390,400,410],
                [500,510,520,530,540,550,560,570,580,590,500,610]])

    port_max = [[500,510,520,530,540,550,560,570,580,590,500,510],
                [700,690,680,670,660,650,640,630,620,610,600,590],
                [800,810,820,830,840,850,860,870,880,890,800,810],
                [900,910,920,930,940,950,960,970,980,990,900,910]]

    port_avg = np.array([[300,310,320,330,340,350,360,370,380,390,300,310],
                [400,390,380,370,360,350,340,330,320,310,300,290],
                [500,510,520,530,540,550,560,570,580,590,600,610],
                [700,710,720,730,740,750,760,770,780,790,700,710]])

    time = [1,2,3,4,5,6,7,8,9,10,11,12]

    #test call
    print(type(port_avg[0]))
    plot_portfolio_performance(time,port_avg,port_min)

    #test call
    plot_portfolio_model_compare(time,port_avg)

def plot_portfolio_performance(performance):
    """
    Generate a plot of the portfolio performance over time given
    loan decisions made based on strategy chosen from varying models.

    Parameters
    ----------
    performance: performance statistics of the portfolio over time.
    """
    _, num_months, D = performance.shape
    mean, stddev = np.zeros((D, num_months)), np.zeros((D, num_months))
    
    # Profits
    profits = performance[:,:,0]
    mean_profits   = np.mean(profits, axis=0)
    stddev_profits = np.std(profits, axis=0)
    mean[2,:]   = mean_profits
    stddev[2,:] = stddev_profits
    
    # Cumulative profits
    cum_profits        = np.cumsum(profits, axis=1)
    mean_cum_profits   = np.mean(cum_profits, axis=0)
    stddev_cum_profits = np.std(cum_profits, axis=0)
    mean[0,:]   = mean_cum_profits
    stddev[0,:] = stddev_cum_profits

    # Funds remaining
    funds = performance[:,:,1]
    mean_funds   = np.mean(funds, axis=0)
    stddev_funds = np.std(funds, axis=0)
    mean[1,:]   = mean_funds
    stddev[1,:] = stddev_funds

    # Profit percentage
    profit_margin = performance[:,:,4]
    mean_profit_margin = np.mean(profit_margin, axis=0)
    stddev_profit_margin = np.std(profit_margin, axis=0)
    mean[3,:]   = mean_profit_margin
    stddev[3,:] = stddev_profit_margin

    plot_mean_and_std(np.arange(num_months), mean, stddev)

#multiple plots 2x2 comparison
def plot_mean_and_std(t, mean, std):
    """
    Generate multiple plots with mean and std deviation.

    Parameters
    ----------
    t: int array
    mean: int array
    std: int array
    """

    fig = plt.figure(figsize=(25,12))

    #2x2 with 4 plots
    sub1 = fig.add_subplot(2,2,1)
    sub1.set_title('Cumulative profits')
    sub1.set_xlabel("Time (in months)")
    sub1.set_ylabel("$")
    sub1.plot(t,mean[0],'red')
    sub1.fill_between(t, mean[0]-std[0], mean[0]+ std[0],alpha=0.1, color = 'r')

    sub2 = fig.add_subplot(2,2,2)
    sub2.set_title('Funds remaining in the portfolio')
    sub2.set_xlabel("Time (in months)")
    sub2.set_ylabel("$")
    sub2.plot(t,mean[1],'blue')
    sub2.fill_between(t, mean[1]-std[1], mean[1]+ std[1],alpha=0.1, color = 'b')

    sub3 = fig.add_subplot(2,2,3)
    sub3.set_title('Profits during the month')
    sub3.set_xlabel("Time (in months)")
    sub3.set_ylabel("$")
    sub3.plot(t,mean[2],'green')
    sub3.fill_between(t, mean[2]-std[2], mean[2]+ std[2],alpha=0.1, color = 'g')

    sub4 = fig.add_subplot(2,2,4)
    sub4.set_title('Profit percentage')
    sub4.set_xlabel("Time (in months)")
    sub4.set_ylabel("%")
    sub4.plot(t,mean[3],'black')
    sub4.fill_between(t, mean[3]-std[3], mean[3]+ std[3],alpha=0.1, color = 'k')

    plt.tight_layout()
    plt.show()

#plot on the same axis
def plot_portfolio_model_compare(t, port_avg):
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

    plt.plot(t,port_avg[0],'black',label = "Model Name 1")
    plt.plot(t,port_avg[1],'red',label = "Model Name 2")
    plt.plot(t,port_avg[2],'blue',label = "Model Name 3")
    plt.plot(t,port_avg[3],'green',label = "Model Name 4")
    plt.xlabel("Time (in months)")
    plt.ylabel("$")
    plt.title("Comparison of Mean Portfolio Performance")

    plt.show()
