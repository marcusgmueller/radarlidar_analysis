from datetime import datetime, timedelta
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import numpy as np
from RadarLidarWindSpeed import RadarLidarWindSpeed
import matplotlib.dates as mdates



def plotRoutine(plotFilePath, dateBegin, dateEnd):



    #-----wind speed full height-----#

    grid = list(range(0,13000,26))



    #now = datetime.now()
    #dateEnd = datetime(now.year, now.month, now.day,23,59)
    #dateBegin = datetime(now.year, now.month, now.day)

    #run analysis
    analysis = RadarLidarWindSpeed(dateBegin, dateEnd, grid)
    analysis.importDataset()


    analysis.calculateFusion()
    analysis.calculateDifferences()
    analysis.calculateAvailability()






    #height coverage plot
    result = analysis.getHeightProfile()
    heightGrid = analysis.heightGrid
    plt.figure(figsize=(20,10))
    plt.plot(result['radar Coverage'].tolist(),heightGrid, 'go-', label='Radar')
    plt.plot(result['lidar Coverage'].tolist(),heightGrid, 'rs-', label='Lidar')
    plt.plot(result['total Coverage'].tolist(),heightGrid, 'b*-', label='Total')
    axes = plt.axes()
    #axes.set_ylim([0, 12000])
    axes.set_xlim([0, 100])
    plt.xlabel("coverage [%]", fontsize=16)
    plt.ylabel("height [m]", fontsize=16)
    plt.legend(fontsize=16)
    plt.title('data coverage by height: '+dateBegin.strftime("%Y-%m-%d"), fontsize=16)
    plt.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"coverageHeightPlot_fullheight_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
    plt.close()




    #plot data overview
    df = analysis.dataframe
    df.reset_index(level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    diff = df.pivot(index="height", columns="time", values="speedDifference")
    fusion = df.pivot(index="height", columns="time", values="Fusion")
    availability = df.pivot(index="height", columns="time", values="availability")
    X,Y = np.meshgrid(analysis.hours, analysis.heightGrid)



    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

    fig.suptitle("data overview "+dateBegin.strftime("%Y-%m-%d"), fontsize=16)

    # Availability
    col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
    axes[0].set_title("data availability ")
    axes[0].set_ylabel("height AGL [m]")

    # Difference
    im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-10, vmax=10)#+-10
    axes[1].set_title("Difference: Radar - Lidar ")
    axes[1].set_ylabel("height AGL [m]")

    # Fusion
    im = axes[2].pcolor(X,Y,fusion,cmap='viridis', vmin=0, vmax=32)#vmax=32, cmap=viridis
    axes[2].set_title("Radar/Lidar fusion ")
    axes[2].set_xlabel("Time UTC [h]")
    axes[2].set_ylabel("height AGL [m]")
    #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
    # cbar speed
    #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    #cbar = fig.colorbar(im, cax=cb_ax)
    #cbar.set_label('Horizontal wind speed [m/s]')
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Horizontal wind speed [m/s]')



    # cbar difference
    cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cbar3 = fig.colorbar(im3, cax=cb_ax3)
    cbar3.set_label('Difference [m/s]')

    # cbar availability
    cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
    cbar2.set_ticks([0,1,2,3])
    cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
    cbar2.set_label('data availability')

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

    #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
    plt.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"merge_windspeed_fullheight_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300,bbox_inches='tight')
    plt.close(fig)
    #plt.show()
    analysis.exportNCDF(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"windspeed_fullheight_"+str(dateBegin.strftime("%Y%m%d"))+".nc")


    #-----wind speed boundary layer-----#

    grid = list(range(0,3000,10))



    #now = datetime.now()
    #dateEnd = datetime(now.year, now.month, now.day,23,59)
    #dateBegin = datetime(now.year, now.month, now.day)

    #run analysis
    analysis = RadarLidarWindSpeed(dateBegin, dateEnd, grid)
    analysis.importDataset()


    analysis.calculateFusion()
    analysis.calculateDifferences()
    analysis.calculateAvailability()






    #height coverage plot
    result = analysis.getHeightProfile()
    heightGrid = analysis.heightGrid
    #plt.figure(figsize=(20,10))
    fig1 = plt.figure(figsize=(20,10))
    ax1 = plt.axes()
    ax1.plot(result['radar Coverage'].tolist(),heightGrid, 'go-', label='Radar')
    ax1.plot(result['lidar Coverage'].tolist(),heightGrid, 'rs-', label='Lidar')
    ax1.plot(result['total Coverage'].tolist(),heightGrid, 'b*-', label='Total')
    #axes.set_ylim([0, 12000])
    ax1.set_xlim([0, 100])
    plt.xlabel("coverage [%]", fontsize=16)
    plt.ylabel("height [m]", fontsize=16)
    plt.legend(fontsize=16)
    plt.title('data coverage by height: '+dateBegin.strftime("%Y-%m-%d"), fontsize=16)
    fig1.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"coverageHeightPlot_boundarylayer_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)

    plt.close(fig1)


    #plot data overview
    df = analysis.dataframe
    df.reset_index(level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    diff = df.pivot(index="height", columns="time", values="speedDifference")
    fusion = df.pivot(index="height", columns="time", values="Fusion")
    availability = df.pivot(index="height", columns="time", values="availability")
    X,Y = np.meshgrid(analysis.hours, analysis.heightGrid)



    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

    fig.suptitle("data overview "+dateBegin.strftime("%Y-%m-%d"), fontsize=16)

    # Availability
    col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
    axes[0].set_title("data availability ")
    axes[0].set_ylabel("height AGL [m]")

    # Difference
    im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-10, vmax=10)#+-10
    axes[1].set_title("Difference: Radar - Lidar ")
    axes[1].set_ylabel("height AGL [m]")

    # Fusion
    im = axes[2].pcolor(X,Y,fusion,cmap='viridis', vmin=0, vmax=32)#vmax=32, cmap=viridis
    axes[2].set_title("Radar/Lidar fusion ")
    axes[2].set_xlabel("Time UTC [h]")
    axes[2].set_ylabel("height AGL [m]")
    #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
    # cbar speed
    #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    #cbar = fig.colorbar(im, cax=cb_ax)
    #cbar.set_label('Horizontal wind speed [m/s]')
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Horizontal wind speed [m/s]')



    # cbar difference
    cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cbar3 = fig.colorbar(im3, cax=cb_ax3)
    cbar3.set_label('Difference [m/s]')

    # cbar availability
    cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
    cbar2.set_ticks([0,1,2,3])
    cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
    cbar2.set_label('data availability')

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

    #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
    plt.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"merge_windspeed_boundarylayer_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300,bbox_inches='tight')
    plt.close()
    #plt.show()

    analysis.exportNCDF(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"windspeed_boundarylayer_"+str(dateBegin.strftime("%Y%m%d"))+".nc")



    #-----wind direction full height-----#

    grid = list(range(0,13000,26))



    #now = datetime.now()
    #dateEnd = datetime(now.year, now.month, now.day,23,59)
    #dateBegin = datetime(now.year, now.month, now.day)

    #run analysis
    analysis = RadarLidarWindSpeed(dateBegin, dateEnd, grid,'dir')
    analysis.importDataset()


    analysis.calculateFusion()
    analysis.calculateDifferences()
    analysis.calculateAvailability()






    #plot data overview
    df = analysis.dataframe
    df.reset_index(level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    diff = df.pivot(index="height", columns="time", values="speedDifference")
    fusion = df.pivot(index="height", columns="time", values="Fusion")
    availability = df.pivot(index="height", columns="time", values="availability")
    X,Y = np.meshgrid(analysis.hours, analysis.heightGrid)



    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

    fig.suptitle("data overview "+dateBegin.strftime("%Y-%m-%d"), fontsize=16)

    # Availability
    col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
    axes[0].set_title("data availability ")
    axes[0].set_ylabel("height AGL [m]")

    # Difference
    im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-20, vmax=20)#+-10
    axes[1].set_title("Difference: Radar - Lidar ")
    axes[1].set_ylabel("height AGL [m]")

    # Fusion
    im = axes[2].pcolor(X,Y,fusion,cmap='twilight', vmin=0, vmax=360)#vmax=32, cmap=viridis
    axes[2].set_title("Radar/Lidar fusion ")
    axes[2].set_xlabel("Time UTC [h]")
    axes[2].set_ylabel("height AGL [m]")
    #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
    # cbar speed
    #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    #cbar = fig.colorbar(im, cax=cb_ax)
    #cbar.set_label('Horizontal wind speed [m/s]')
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Horizontal wind direction [°]')



    # cbar difference
    cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cbar3 = fig.colorbar(im3, cax=cb_ax3)
    cbar3.set_label('Difference [°]')#Difference [m/s]

    # cbar availability
    cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
    cbar2.set_ticks([0,1,2,3])
    cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
    cbar2.set_label('data availability')

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

    #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
    plt.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"merge_winddirection_fullheight_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300,bbox_inches='tight')
    plt.close()
    #plt.show()




    analysis.exportNCDF(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"winddirection_fullheight_"+str(dateBegin.strftime("%Y%m%d"))+".nc")




    #-----wind direction boundary layer-----#


    grid = list(range(0,3000,10))



    #now = datetime.now()
    #dateEnd = datetime(now.year, now.month, now.day,23,59)
    #dateBegin = datetime(now.year, now.month, now.day)

    #run analysis
    analysis = RadarLidarWindSpeed(dateBegin, dateEnd, grid,'dir')
    analysis.importDataset()


    analysis.calculateFusion()
    analysis.calculateDifferences()
    analysis.calculateAvailability()







    #plot data overview
    df = analysis.dataframe
    df.reset_index(level=0, inplace=True)
    df.reset_index(level=0, inplace=True)
    diff = df.pivot(index="height", columns="time", values="speedDifference")
    fusion = df.pivot(index="height", columns="time", values="Fusion")
    availability = df.pivot(index="height", columns="time", values="availability")
    X,Y = np.meshgrid(analysis.hours, analysis.heightGrid)



    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(10, 10), sharex=True, sharey=False)

    fig.suptitle("data overview "+dateBegin.strftime("%Y-%m-%d"), fontsize=16)

    # Availability
    col_dict={1:"blue", 2:"red", 13:"orange", 7:"green"}
    cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
    im2 = axes[0].pcolor(X,Y,availability,cmap=cm, vmin=0, vmax=3)
    axes[0].set_title("data availability ")
    axes[0].set_ylabel("height AGL [m]")

    # Difference
    im3 = axes[1].pcolor(X,Y,diff,cmap='bwr', vmin=-20, vmax=20)#+-10
    axes[1].set_title("Difference: Radar - Lidar ")
    axes[1].set_ylabel("height AGL [m]")

    # Fusion
    im = axes[2].pcolor(X,Y,fusion,cmap='twilight', vmin=0, vmax=360)#vmax=32, cmap=viridis
    axes[2].set_title("Radar/Lidar fusion ")
    axes[2].set_xlabel("Time UTC [h]")
    axes[2].set_ylabel("height AGL [m]")
    #plt.savefig(plotFilePath+"merge_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300)
    # cbar speed
    #cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    #cbar = fig.colorbar(im, cax=cb_ax)
    #cbar.set_label('Horizontal wind speed [m/s]')
    cb_ax = fig.add_axes([1, 0.1, 0.02, 0.8])
    cbar = fig.colorbar(im, cax=cb_ax)
    cbar.set_label('Horizontal wind direction [°]')



    # cbar difference
    cb_ax3 = fig.add_axes([1.1, 0.1, 0.02, 0.8])
    cbar3 = fig.colorbar(im3, cax=cb_ax3)
    cbar3.set_label('Difference [°]')#Difference [m/s]

    # cbar availability
    cb_ax2 = fig.add_axes([1.2, 0.1, 0.02, 0.8])
    cbar2 = fig.colorbar(im2, cax=cb_ax2, ticks=[0,1,2,3])
    cbar2.set_ticks([0,1,2,3])
    cbar2.set_ticklabels(["no data", "only Radar", "only Lidar","both"])
    cbar2.set_label('data availability')

    xformatter = mdates.DateFormatter('%H:%M')
    plt.gcf().axes[2].xaxis.set_major_formatter(xformatter)

    #fig.subplots_adjust(wspace=0.05, hspace=0.4, right=0.4)
    plt.savefig(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"merge_winddirection_boundarylayer_"+str(dateBegin.strftime("%Y%m%d"))+".png",dpi=300,bbox_inches='tight')
    #plt.show()
    plt.close() 
    analysis.exportNCDF(plotFilePath+str(dateBegin.strftime("%Y/%m/%d/"))+"winddirection_boundarylayer_"+str(dateBegin.strftime("%Y%m%d"))+".nc")