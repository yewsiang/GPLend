import matplotlib.pyplot as plt
import numpy as np
#loan_data = pd.read_csv('dataset/LoanStats3a.csv')
#print(loan_data)

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

#multiple plots 2x2 comparison
def plot_portfolio_performance(t, mean, std):
    """
    Generate a plot of the portfolio performance over time given
    loan decisions made based on strategy chosen from varying models.

    Parameters
    ----------
    t: int array
    mean: int array
    std: int array
    """
    fig = plt.figure(figsize=(25,12))

    ymin = np.min(mean[0]-std[0])
    ymax = np.max(mean[0]+std[0])

    for i in range(1,4):
        model_min = np.min(mean[i]-std[i])
        model_max = np.min(mean[i]+std[i])
        if(model_min < ymin):
            ymin = model_min
        if(model_max > ymax):
            ymax = model_max

    #2x2 with 4 plots
    sub1 = fig.add_subplot(2,2,1)
    sub1.set_title('Model Name 1')
    sub1.set_xlabel("Time (in months)")
    sub1.set_ylabel("$")
    sub1.plot(t,mean[0],'red')
    sub1.fill_between(t, mean[0]-std[0], mean[0]+ std[0],alpha=0.1, color = 'r')
    sub1.set_ylim([ymin,ymax])

    sub2 = fig.add_subplot(2,2,2)
    sub2.set_title('Model Name 2')
    sub2.set_xlabel("Time (in months)")
    sub2.set_ylabel("$")
    sub2.plot(t,mean[1],'blue')
    sub2.fill_between(t, mean[1]-std[1], mean[0]+ std[0],alpha=0.1, color = 'b')
    sub2.set_ylim([ymin,ymax])

    sub3 = fig.add_subplot(2,2,3)
    sub3.set_title('Model Name 3')
    sub3.set_xlabel("Time (in months)")
    sub3.set_ylabel("$")
    sub3.plot(t,mean[2],'green')
    sub3.fill_between(t, mean[2]-std[2], mean[2]+ std[2],alpha=0.1, color = 'g')
    sub3.set_ylim([ymin,ymax])

    sub4 = fig.add_subplot(2,2,4)
    sub4.set_title('Model Name 4')
    sub4.set_xlabel("Time (in months)")
    sub4.set_ylabel("$")
    sub4.plot(t,mean[3],'black')
    sub4.fill_between(t, mean[3]-std[3], mean[3]+ std[3],alpha=0.1, color = 'k')
    sub4.set_ylim([ymin,ymax])

    plt.tight_layout()
    plt.show()

#test call
print(type(port_avg[0]))
plot_portfolio_performance(time,port_avg,port_min)

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

#test call
plot_portfolio_model_compare(time,port_avg)
