import matplotlib.pyplot as plt
import os
import sys

if len(sys.argv)<=3:
    print(f'Error! Format: python plot_ratio.py <single/multi>') 
    print(f'    If single: python plot_ratio.py single <date> <time> <topic> <y_limit>')
    print(f'    Example: python plot_ratio.py single 2023-04-10 15-58-20 distance 5')
    print(f'    If multi: python plot_ratio.py multi ')
    print(f'    Adjust parametere inside')
    sys.exit
else:
    if sys.argv[1] == "single":
        if len(sys.argv)<=5:
            print(f'Error! Format: python plot_ratio.py single <date> <time> <topic> <y_limit>')
            print(f'Example: python plot_ratio.py single 2023-04-10 15-58-20 distance 5')
            sys.exit
        else:
            for arg in sys.argv:
                print(arg)
            TOPIC = sys.argv[4]
            ylim = float(sys.argv[5])
            OUTPUT_PATH = "./outputs"
            DATE_PATH = sys.argv[2]
            TIME_PATH = sys.argv[3]


            file_path = os.path.join(OUTPUT_PATH, DATE_PATH, TIME_PATH)
            # loss_path = os.path.join(file_path, 'distance_ratio.txt')
            if TOPIC=="loss":
                loss_path = os.path.join(file_path, 'loss.txt')
                plot_path = os.path.join(file_path, 'loss.png')
            else:
                loss_path = os.path.join(file_path, TOPIC+'_ratio.txt')
                plot_path = os.path.join(file_path, 'test_'+TOPIC+'_ratio.png')

            with open(loss_path, 'r') as f:
                lines = f.readlines()

            lines = [float(l[:-1]) for l in lines]

            plt.plot(lines)
            # plt.plot(epochs, ave_ratios)
            plt.xlabel('epoch')
            if TOPIC=="loss":
                plt.ylabel('loss')
                plt.title('loss v.s. epoch')
            elif TOPIC =="distance":
                plt.ylabel('pred_distance/init_distance')
                plt.title('ratio of pred_dist/init_dis v.s. epoch')
            else:
                plt.ylabel('angle_diff/init_angle')
                plt.title('ratio of angle_diff/init_angle v.s. epoch')  

            # plt.legend(['pred_shift_distance / init_shift_distance'])
            # plt.legend(['shift_u only'])
            # plt.legend(['shift_v only'])
            plt.legend(['theta only'])

            plt.ylim([0, ylim])
            # plt.axis('scaled')
            plt.savefig(plot_path)
    else:

        # ------ Subplotting ---------- #

        # TOPICS = ["loss", "loss", "loss", "distance", "distance", "angle"]
        # ylims = [350, 250, 300, 500, 350, 16]
        # OUTPUT_PATH = "./outputs"
        # DATE_PATH = sys.argv[1]
        # TIME_PATHS = ["15-58-20", "15-59-29", "18-18-18"] # or sys.argv[2]~[5]
        # legends = ["shift_u-only", "shift_v-only", "theta"]
        
        # # Subplotting 
        # rows, cols = 2, 3

        TOPICS = ["loss", "distance", "angle"]
        ylims = [500, 100, 100]
        OUTPUT_PATH = "./outputs"
        DATE_PATH = sys.argv[2]
        TIME_PATHS = ["03-00-21", "03-00-21", "03-00-21"] # or sys.argv[2]~[5]
        legends = ["Loss", "distance-ratio", "angle-ratio"]
        
        # Subplotting 
        rows, cols = 1, 3        

        def get_lines(TOPIC, OUTPUT_PATH, DATE_PATH, TIME_PATH):
            file_path = os.path.join(OUTPUT_PATH, DATE_PATH, TIME_PATH)
            # loss_path = os.path.join(file_path, 'distance_ratio.txt')
            if TOPIC=="loss":
                loss_path = os.path.join(file_path, 'loss.txt')
            else:
                loss_path = os.path.join(file_path, TOPIC+'_ratio.txt')

            with open(loss_path, 'r') as f:
                lines = f.readlines()
            lines = [float(l[:-1]) for l in lines]
            return lines


        save_fig_path = os.path.join(OUTPUT_PATH, DATE_PATH, 'all-shifts-loss-ratio.png')


        fig, ax = plt.subplots(rows, cols)
                            # sharex='col', 
                            # sharey='row')


        for row in range(rows):
            for col in range(cols):
                TOPIC = TOPICS[row*cols + col]
                print('topic', TOPIC)
                lines = get_lines(TOPIC, OUTPUT_PATH, DATE_PATH, TIME_PATHS[col])  
                print(col)
                # print(lines)   
                ax[col].plot(lines)
                if TOPIC=="loss":
                    ylabel_ = 'loss'
                    # title_ = 'loss v.s. epoch'
                elif TOPIC =="distance":
                    ylabel_ = 'ratio (distance)'
                    # title_ = 'ratio of pred_dist/init_dis v.s. epoch'
                else:
                    ylabel_ = 'ratio (angle) '
                    # title_ = 'ratio of angle_diff/init_angle v.s. epoch'  

                if row==0:  
                    title_ = legends[col]
                    ax[col].set(xlabel='epoch', ylabel= ylabel_, title=title_)       
                else:   
                    ax[col].set(xlabel='epoch', ylabel= ylabel_) #,title=title_)            
                # ax[row,col].legend([legends[col]])
                ax[col].set_ylim([0, ylims[row*cols+col]])
                ax[col].set_box_aspect(1.0)
                # ax[row, col].text(0.5, 0.5, 
                #                   str((row, col)),
                #                   color="green",
                #                   fontsize=18, 
                #                   ha='center')

        fig.tight_layout(pad=0.5)
        plt.savefig(save_fig_path)