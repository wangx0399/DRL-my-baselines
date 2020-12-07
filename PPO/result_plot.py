import pandas as pd
import numpy as np
import seaborn as sns
sns.set()
import matplotlib
import matplotlib.pyplot as plt
from os.path import join as joindir


# move to compute average
def move_avg(data, parallel_num=5, sm=1):  # sm=3, 5, 7, 9, 11, ...
    if sm > 1:
        y = np.ones(sm) * 1.0 / sm
        if parallel_num > 1:
            smooth_data = []
            for i in range(parallel_num):
                d = data[data['parallel_run']==i]['mean_ep_reward']
                dd = np.convolve(d, y, 'same')
                x = (sm+1)//2
                dd[-x:-1] = d.values[-x:-1]
                dd[-1] = d.values[-1]
                dd[0:x] = d.values[0:x]
                smooth_data = np.hstack((smooth_data, dd))
        else:
            d = data['mean_ep_reward']
            dd = np.convolve(data['mean_ep_reward'], y, 'same')
            x = (sm + 1) // 2
            dd[-x:-1] = d.values[-x:-1]
            dd[-1] = d.values[-1]
            dd[0:x] = d.values[0:x]
            smooth_data = dd
    else:
        smooth_data = data['mean_ep_reward']
    return smooth_data


# 1 for PPO_simple
# 0 for PPO  5 parallel run
# 10 for PPO and PPO_origin comparison
point = 2
env = 'Pendulum-v0'
RESULT_DIR = '/home/wangxu/PycharmProjects/torchprojects/result/'

if point == 10:
    df0 = pd.read_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(env)))
    df1 = pd.read_csv(joindir(RESULT_DIR, 'ppo-origin-record-{}.csv'.format(env)))  # no index columns
    df0['mean_ep_reward'] = move_avg(df0, parallel_num=5, sm=11)
    df1['mean_ep_reward'] = move_avg(df1, parallel_num=5, sm=11)
    df0['isorigin'] = 0
    df1['isorigin'] = 1
    df = pd.concat([df0, df1], ignore_index=True)
    sns.set(style='darkgrid', font_scale=1.0)
    sns.tsplot(data=df, time='episode', value='mean_ep_reward', unit='parallel_run', condition='isorigin')#, err_style='ci_band')
    # tsplot(data,  time=    ,value=    ,unit= compute mean ,condition= different Env)
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)
    plt.xlabel('Epoch Iteration Number')
    plt.ylabel('Mean Reward')
    plt.title('PPO for {}'.format(env))
    plt.show()
elif point == 1:
    df = pd.read_csv(joindir(RESULT_DIR, 'ppo-simple-record.csv'))
    df['mean_ep_reward'] = move_avg(df, parallel_num=1, sm=11)
    df['unit'] = 1
    sns.set(style='darkgrid', font_scale=1.0)
    sns.tsplot(data=df, time='episode', value='mean_ep_reward', unit='unit')
    # tsplot(data,  time=    ,value=    ,unit= compute mean ,condition= different Env)
    plt.tight_layout(pad=0.5)
    plt.xlabel('Epoch Iteration Number')
    plt.ylabel('Mean Reward')
    plt.title('PPO simple for Walker2D-v2')
    plt.show()
elif point == 0:
    df = pd.read_csv(joindir(RESULT_DIR, 'ppo-record-{}.csv'.format(env)))
    df['mean_ep_reward'] = move_avg(df, parallel_num=1, sm=11)
    sns.set(style='darkgrid', font_scale=1.0)
    sns.tsplot(data=df, time='episode', value='mean_ep_reward', unit='parallel_run')
    # tsplot(data,  time=    ,value=    ,unit= compute mean ,condition= different Env)
    plt.tight_layout(pad=0.5)
    plt.xlabel('Epoch Iteration Number')
    plt.ylabel('Mean Reward')
    plt.title('ppo=0\ ppo_origin=1 for {}]'.format(env))
    plt.show()
elif point == 2:
    df0 = pd.read_csv(joindir(RESULT_DIR, 'pendulum-record-{}.csv'.format(env)))
    df1 = pd.read_csv(joindir(RESULT_DIR, 'ppo-simple-record-{}.csv'.format(env)))  # no index columns
    df0['mean_ep_reward'] = move_avg(df0, parallel_num=1, sm=5)
    df1['mean_ep_reward'] = move_avg(df1, parallel_num=1, sm=5)
    df0['isorigin'] = 0
    df1['isorigin'] = 1
    df = pd.concat([df0, df1], ignore_index=True)
    sns.set(style='darkgrid', font_scale=1.0)
    sns.tsplot(data=df, time='episode', value='mean_ep_reward', unit='isorigin', condition='isorigin')  # , err_style='ci_band')
    # tsplot(data,  time=    ,value=    ,unit= compute mean ,condition= different Env)
    plt.legend(loc='best').set_draggable(True)
    plt.tight_layout(pad=0.5)
    plt.xlabel('Epoch Iteration Number')
    plt.ylabel('Mean Reward')
    #plt.title('PPO for {}'.format(env))
    plt.show()
else:
    print('which one should point be')
